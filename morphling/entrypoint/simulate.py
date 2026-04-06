from __future__ import annotations

import argparse
import importlib
import json


def _model_from_name(name, transformer_model_config):
    if name == "gpt2":
        return transformer_model_config(
            num_layers=12,
            hidden_dim=768,
            num_heads=12,
        )
    if name == "gpt2-medium":
        return transformer_model_config(
            num_layers=24,
            hidden_dim=1024,
            num_heads=16,
        )
    if name == "gpt2-large":
        return transformer_model_config(
            num_layers=36,
            hidden_dim=1280,
            num_heads=20,
        )
    if name == "custom":
        return transformer_model_config()
    raise ValueError(f"Unsupported model preset: {name}")


def _to_float_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse numeric value: {value}")

    text = value.strip()
    if text.isdigit() or text.replace(".", "", 1).isdigit():
        return float(text)

    units = {
        "B": 1.0,
        "K": 1024.0,
        "M": 1024.0**2,
        "G": 1024.0**3,
        "T": 1024.0**4,
        "P": 1024.0**5,
    }
    if not text:
        raise ValueError("Cannot parse empty numeric string")

    suffix = text[-1].upper()
    if suffix in units:
        return float(text[:-1]) * units[suffix]
    return float(text)


def _build_default_devices(num_devices, device_spec):
    if num_devices <= 0:
        return []

    flops_min = 1e12
    flops_max = 10e12
    bw_min = 1e9
    bw_max = 10e9
    ul_lat_min = 50e-6
    ul_lat_max = 300e-6
    dl_lat_min = 50e-6
    dl_lat_max = 300e-6

    devices = []
    denom = max(1, num_devices - 1)
    for rank in range(num_devices):
        alpha = rank / denom
        flops = flops_min + alpha * (flops_max - flops_min)
        bw = bw_min + alpha * (bw_max - bw_min)
        ul_lat = ul_lat_max - alpha * (ul_lat_max - ul_lat_min)
        dl_lat = dl_lat_max - alpha * (dl_lat_max - dl_lat_min)
        devices.append(
            device_spec(
                rank=rank,
                flops=flops,
                ul_bw_bytes_per_s=bw,
                dl_bw_bytes_per_s=bw,
                ul_lat_s=ul_lat,
                dl_lat_s=dl_lat,
            )
        )
    return devices


def _load_devices_from_json(path, device_spec):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("device-config JSON must be a list of device objects")

    devices = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"device-config item at index {idx} is not an object"
            )

        rank = int(item.get("rank", idx))
        flops_raw = item.get("flops", item.get("flops_bytes_per_s"))
        ul_bw_raw = item.get("ul_bw_bytes_per_s", item.get("ul_bw"))
        dl_bw_raw = item.get("dl_bw_bytes_per_s", item.get("dl_bw"))
        ul_lat_raw = item.get("ul_lat_s", item.get("ul_lat", 0.0))
        dl_lat_raw = item.get("dl_lat_s", item.get("dl_lat", 0.0))

        if flops_raw is None or ul_bw_raw is None or dl_bw_raw is None:
            raise ValueError(
                "Each device needs flops, ul_bw(_bytes_per_s), and dl_bw(_bytes_per_s)"
            )

        devices.append(
            device_spec(
                rank=rank,
                flops=_to_float_number(flops_raw),
                ul_bw_bytes_per_s=_to_float_number(ul_bw_raw),
                dl_bw_bytes_per_s=_to_float_number(dl_bw_raw),
                ul_lat_s=float(ul_lat_raw),
                dl_lat_s=float(dl_lat_raw),
            )
        )

    return devices


def _default_hybrid_groups(devices):
    ranks = [device.rank for device in devices]
    n = len(ranks)
    if n <= 1:
        return [ranks] if ranks else []

    split = max(1, n // 2)
    left = ranks[:split]
    right = ranks[split:]
    if not right:
        return [left]
    return [left, right]


def _parse_topologies(spec, devices, topology_config):
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one topology must be specified")

    topologies = []
    for name in names:
        if name == "allreduce-ring":
            topologies.append(
                topology_config(mode="allreduce", allreduce_algo="ring")
            )
        elif name == "allreduce-tree":
            topologies.append(
                topology_config(mode="allreduce", allreduce_algo="tree")
            )
        elif name == "ps":
            topologies.append(topology_config(mode="ps", num_ps_servers=1))
        elif name == "hybrid":
            topologies.append(
                topology_config(
                    mode="hybrid",
                    allreduce_algo="ring",
                    intra_group_algo="allreduce",
                    inter_group_algo="ps",
                    device_groups=_default_hybrid_groups(devices),
                )
            )
        elif name.startswith("ps-") and name.endswith("server"):
            count = int(name.removeprefix("ps-").removesuffix("server"))
            topologies.append(
                topology_config(mode="ps", num_ps_servers=max(1, count))
            )
        elif name.startswith("ps-") and name.endswith("servers"):
            count = int(name.removeprefix("ps-").removesuffix("servers"))
            topologies.append(
                topology_config(mode="ps", num_ps_servers=max(1, count))
            )
        elif name.startswith("hybrid-"):
            parts = name.split("-")
            if len(parts) != 3:
                raise ValueError(f"Invalid hybrid format: {name}")
            if parts[1] not in {"ar", "ps"} or parts[2] not in {"ar", "ps"}:
                raise ValueError(f"Invalid hybrid format: {name}")
            intra = "allreduce" if parts[1] == "ar" else "ps"
            inter = "allreduce" if parts[2] == "ar" else "ps"
            topologies.append(
                topology_config(
                    mode="hybrid",
                    allreduce_algo="ring",
                    intra_group_algo=intra,
                    inter_group_algo=inter,
                    device_groups=_default_hybrid_groups(devices),
                )
            )
        else:
            raise ValueError(f"Unsupported topology: {name}")
    return topologies


def main():
    parser = argparse.ArgumentParser(
        description="Morphling distributed training topology simulator"
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "custom"],
    )
    _ = parser.add_argument("--num-devices", type=int, default=4)
    _ = parser.add_argument("--num-steps", type=int, default=10)
    _ = parser.add_argument(
        "--topologies",
        type=str,
        default="allreduce-ring,allreduce-tree,ps,hybrid",
    )
    _ = parser.add_argument(
        "--overlap",
        type=str,
        default="none",
        choices=["none", "full"],
    )
    _ = parser.add_argument("--output-json", type=str, default="")
    _ = parser.add_argument("--output-csv", type=str, default="")
    _ = parser.add_argument("--device-config", type=str, default="")

    args = parser.parse_args()

    config_mod = importlib.import_module("morphling.simulator.config")
    runner_mod = importlib.import_module("morphling.simulator.runner")
    output_mod = importlib.import_module("morphling.simulator.output")

    device_spec = getattr(config_mod, "DeviceSpec")
    topology_config = getattr(config_mod, "TopologyConfig")
    transformer_model_config = getattr(config_mod, "TransformerModelConfig")
    comparison_runner = getattr(runner_mod, "ComparisonRunner")
    format_table = getattr(output_mod, "format_comparison_table")
    export_json = getattr(output_mod, "export_json")
    export_csv = getattr(output_mod, "export_csv")

    model_cfg = _model_from_name(args.model, transformer_model_config)
    layers = model_cfg.to_layers()

    if args.device_config:
        devices = _load_devices_from_json(args.device_config, device_spec)
    else:
        devices = _build_default_devices(args.num_devices, device_spec)

    topologies = _parse_topologies(args.topologies, devices, topology_config)
    runner = comparison_runner(
        layers=layers,
        devices=devices,
        topologies=topologies,
        num_steps=args.num_steps,
        overlap_mode=args.overlap,
    )
    results = runner.run()

    print(format_table(results))

    if args.output_json:
        export_json(results, args.output_json)
    if args.output_csv:
        export_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
