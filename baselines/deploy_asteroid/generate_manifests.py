#!/usr/bin/env python3
"""
Baselines-Asteroid K8s Manifest Generator.

Renders Jinja2 templates to create K8s manifests
from hpp_plan.json.

Usage:
    python generate_manifests.py --plan ./hpp_plan.json
    python generate_manifests.py --plan ./hpp_plan.json \
        --image myregistry/baselines:v1.0.0
    python generate_manifests.py --plan ./hpp_plan.json \
        --apply
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        TemplateNotFound,
    )
except ImportError:
    print(
        "Error: Jinja2 not installed. "
        "Run: pip install jinja2"
    )
    sys.exit(1)


def load_plan(path: str) -> Dict[str, Any]:
    """Load hpp_plan.json file."""
    with open(path, "r") as f:
        return json.load(f)


def resolve_k8s_node_names() -> Dict[str, str]:
    """Query kubectl to map IP -> K8s node name."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "nodes",
                "-o",
                "jsonpath="
                "{range .items[*]}"
                "{.status.addresses"
                '[?(@.type=="InternalIP")]'
                ".address}="
                "{.metadata.name}"
                '{"\\n"}{end}',
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "KUBECONFIG": os.path.expanduser(
                    "~/.kube/config"
                ),
            },
        )
        if result.returncode == 0:
            mapping: Dict[str, str] = {}
            for line in (
                result.stdout.strip().split("\n")
            ):
                if "=" in line:
                    ip, name = line.split("=", 1)
                    mapping[ip.strip()] = name.strip()
            return mapping
    except Exception:
        pass
    return {}


def setup_jinja_env(templates_dir: str) -> Environment:
    """Configure Jinja2 environment."""
    return Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def generate_configmap(
    env: Environment,
    plan: Dict[str, Any],
    namespace: str,
) -> str:
    """Generate ConfigMap YAML with hpp_plan.json."""
    try:
        template = env.get_template(
            "configmap.yaml.j2"
        )
    except TemplateNotFound:
        print(
            "Error: configmap.yaml.j2 not found"
        )
        sys.exit(1)

    return template.render(
        namespace=namespace,
        hpp_plan_json=json.dumps(plan, indent=2),
    )


def generate_headless_service(
    env: Environment,
    namespace: str,
    master_port: int,
) -> str:
    """Generate Headless Service YAML."""
    try:
        template = env.get_template(
            "headless_service.yaml.j2"
        )
    except TemplateNotFound:
        print(
            "Error: headless_service.yaml.j2 "
            "not found"
        )
        sys.exit(1)

    return template.render(
        namespace=namespace,
        master_port=master_port,
    )


def _get_master_ip(plan: Dict[str, Any]) -> str:
    """Return rank-0's host IP for MASTER_ADDR.

    With hostNetwork: true, rank 0 listens on its
    host's physical IP, so all ranks must use that
    IP (not the K8s service DNS) as MASTER_ADDR.
    """
    node_mapping = plan.get("node_mapping", {})
    rank0_info = node_mapping.get("0", {})
    ip = rank0_info.get("ip", "")
    if ip:
        return ip
    # Fallback to service DNS if no IP in plan
    return "asteroid-master.default.svc.cluster.local"


def generate_job_manifest(
    env: Environment,
    plan: Dict[str, Any],
    rank: int,
    stage: int,
    namespace: str,
    image: str,
    master_port: int,
    world_size: int,
    k8s_node_map: Dict[str, str] | None = None,
    extra_vars: Dict[str, Any] | None = None,
) -> str:
    """Generate Job YAML for a specific rank."""
    try:
        template = env.get_template(
            "stage_job.yaml.j2"
        )
    except TemplateNotFound:
        print(
            "Error: stage_job.yaml.j2 not found"
        )
        sys.exit(1)

    node_mapping = plan.get("node_mapping", {})
    node_info = node_mapping.get(str(rank), {})

    # Fallback: if rank missing from plan's node_mapping,
    # try the cluster config nodes list (from extra_vars)
    if not node_info and extra_vars:
        cluster_nodes = extra_vars.get(
            "cluster_nodes", []
        )
        for cn in cluster_nodes:
            if cn.get("rank") == rank:
                node_info = cn
                break

    node_hostname = node_info.get(
        "hostname", f"node-{rank}"
    )
    node_ip = node_info.get("ip", "")
    if k8s_node_map and node_ip in k8s_node_map:
        node_hostname = k8s_node_map[node_ip]

    stage_alloc = plan.get(
        "micro_batch_alloc", {}
    ).get(str(stage), {})
    micro_batch_size = stage_alloc.get(
        str(rank), 4
    )

    tpl_vars: Dict[str, Any] = {
        "rank": rank,
        "stage": stage,
        "world_size": world_size,
        "namespace": namespace,
        "master_port": master_port,
        "image": image,
        "node_hostname": node_hostname,
        "memory_mb": node_info.get(
            "memory_mb", 4096
        ),
        "gpu_id": node_info.get("gpu_id", 0),
        "nccl_ifname": node_info.get("nic", "ens33"),
        "master_ip": _get_master_ip(plan),
        "micro_batch_size": micro_batch_size,
        "is_last_stage": (
            stage
            == plan.get("num_stages", 1) - 1
        ),
    }

    if extra_vars:
        tpl_vars.update(extra_vars)

    return template.render(**tpl_vars)


def write_manifest(
    path: Path,
    content: str,
    dry_run: bool = False,
) -> None:
    """Write manifest to file."""
    if dry_run:
        print(f"[DRY RUN] Would write: {path}")
    else:
        path.write_text(content)
        print(f"[CREATED] {path}")


def generate_apply_script(
    output_dir: Path, namespace: str,
) -> str:
    """Generate shell script to apply manifests."""
    return f"""#!/bin/bash
# Apply baselines-asteroid K8s manifests
set -e

MANIFEST_DIR="{output_dir}"
NAMESPACE="{namespace}"

echo "Applying manifests from $MANIFEST_DIR"
echo "Namespace: $NAMESPACE"

if ! kubectl get namespace "$NAMESPACE" \
    &>/dev/null; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

echo "Applying ConfigMap..."
kubectl apply -f "$MANIFEST_DIR/00-configmap.yaml"

echo "Applying Headless Service..."
kubectl apply -f \
    "$MANIFEST_DIR/01-headless-service.yaml"

echo "Applying Worker Jobs..."
for job in "$MANIFEST_DIR"/02-job-rank-*.yaml; do
    echo "  Applying $(basename $job)..."
    kubectl apply -f "$job"
done

echo "All manifests applied!"
echo ""

echo "Waiting for pods to start..."
kubectl wait --for=condition=Ready \
    pod -l app=baselines-asteroid \
    -n "$NAMESPACE" --timeout=300s || true

echo ""
echo "Pod Status:"
kubectl get pods -l app=baselines-asteroid \
    -n "$NAMESPACE" -o wide
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate K8s manifests from "
            "hpp_plan.json"
        ),
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to hpp_plan.json",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help="Jinja2 templates directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for manifests",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="baselines:latest",
        help=(
            "Docker image for worker pods "
            "(default: baselines:latest)"
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="PyTorch distributed port",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print without writing files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply manifests via kubectl",
    )
    parser.add_argument(
        "--image-pull-policy",
        type=str,
        default="IfNotPresent",
        choices=[
            "Always",
            "IfNotPresent",
            "Never",
        ],
        help="Image pull policy",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="asteroid",
        choices=[
            "asteroid",
            "confident",
            "dtfm",
        ],
        help="Training strategy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to asteroid_default.yaml. "
            "Used as fallback for node info "
            "when plan node_mapping is incomplete."
        ),
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    templates_dir = (
        Path(args.templates_dir)
        if args.templates_dir
        else script_dir / "templates"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir / "generated"
    )
    plan_path = Path(args.plan)

    if not plan_path.exists():
        print(
            f"Error: Plan file not found: "
            f"{plan_path}"
        )
        return 1

    if not templates_dir.exists():
        print(
            f"Error: Templates directory not "
            f"found: {templates_dir}"
        )
        return 1

    print("=" * 60)
    print("BASELINES-ASTEROID MANIFEST GENERATOR")
    print("=" * 60)
    print(f"Plan: {plan_path}")
    print(f"Templates: {templates_dir}")
    print(f"Output: {output_dir}")
    print(f"Image: {args.image}")
    print(f"Namespace: {args.namespace}")

    print(f"\nLoading plan from {plan_path}...")
    plan = load_plan(str(plan_path))

    world_size = plan.get("world_size", 1)
    num_stages = plan.get("num_stages", 1)

    print(f"  World size: {world_size}")
    print(f"  Stages: {num_stages}")

    env = setup_jinja_env(str(templates_dir))

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[tuple[str, Path]] = []
    extra_vars = {
        "image_pull_policy": (
            args.image_pull_policy
        ),
        "strategy": args.strategy,
    }

    # Load cluster nodes from config as fallback
    if args.config and Path(args.config).exists():
        if yaml is None:
            print(
                "Warning: pyyaml not installed, "
                "cannot load config fallback"
            )
        else:
            with open(args.config) as cf:
                ast_cfg = yaml.safe_load(cf) or {}
            cluster_nodes = (
                ast_cfg.get("cluster", {})
                .get("nodes", [])
            )
            if cluster_nodes:
                extra_vars["cluster_nodes"] = (
                    cluster_nodes
                )
                print(
                    f"  Loaded {len(cluster_nodes)}"
                    " cluster node(s) from config"
                )

    print("\nGenerating ConfigMap...")
    cm = generate_configmap(
        env, plan, args.namespace,
    )
    cm_path = output_dir / "00-configmap.yaml"
    write_manifest(cm_path, cm, args.dry_run)
    generated.append(("ConfigMap", cm_path))

    print("Generating Headless Service...")
    svc = generate_headless_service(
        env, args.namespace, args.master_port,
    )
    svc_path = (
        output_dir / "01-headless-service.yaml"
    )
    write_manifest(svc_path, svc, args.dry_run)
    generated.append(("Service", svc_path))

    print("Generating Worker Jobs...")
    device_groups = plan.get("device_groups", {})

    k8s_node_map = resolve_k8s_node_names()
    if k8s_node_map:
        print(
            f"  Resolved {len(k8s_node_map)} "
            "K8s node name(s)"
        )

    for stage_str, devices in (
        device_groups.items()
    ):
        stage = int(stage_str)
        for rank in devices:
            job = generate_job_manifest(
                env=env,
                plan=plan,
                rank=rank,
                stage=stage,
                namespace=args.namespace,
                image=args.image,
                master_port=args.master_port,
                world_size=world_size,
                k8s_node_map=k8s_node_map,
                extra_vars=extra_vars,
            )
            job_path = (
                output_dir
                / f"02-job-rank-{rank}.yaml"
            )
            write_manifest(
                job_path, job, args.dry_run,
            )
            generated.append(
                (
                    f"Job (Rank {rank}, "
                    f"Stage {stage})",
                    job_path,
                )
            )

    if not args.dry_run:
        script_content = generate_apply_script(
            output_dir, args.namespace,
        )
        script_path = output_dir / "apply.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        generated.append(
            ("Apply Script", script_path)
        )

    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} files:")
    for name, path in generated:
        print(f"  - {name}: {path.name}")
    print("=" * 60)

    if not args.dry_run:
        print(f"\nTo apply manually:")
        print(
            f"  kubectl apply -f {output_dir}/"
        )
        print(f"\nOr use the apply script:")
        print(
            f"  bash {output_dir}/apply.sh"
        )

    if args.apply and not args.dry_run:
        print("\nApplying manifests...")
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "apply",
                    "-f",
                    str(output_dir) + "/",
                ],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return 1

            print("\nPod status:")
            subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-l",
                    "app=baselines-asteroid",
                    "-n",
                    args.namespace,
                    "-o",
                    "wide",
                ]
            )
        except FileNotFoundError:
            print(
                "Error: kubectl not found in PATH"
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
