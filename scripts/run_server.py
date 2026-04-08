#!/usr/bin/env python3
"""
Run server-only for the DeviceEmulator project.

This script initializes the selected backend (mqtt/rabbitmq/amqp/proxy)
and, if requested, loads the model. It does NOT spawn any device client
processes — it only starts the server and waits for connections.

Example:
  python3 scripts/run_server.py --backend proxy --model_name facebook/opt-125m
  python3 scripts/run_server.py --backend proxy --model_name facebook/opt-125m --cfg ./config/proxy/svr.ini
"""

import argparse
import asyncio
import os
import signal
import time

import torch

import morphling
from morphling.hooks import apply_hooks
from scripts._runtime_common import (
    load_model_and_tokenizer,
    prepare_inputs,
    start_backend,
    wait_for_connections,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run server-only for DeviceEmulator"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="proxy",
        help="Backend to use: proxy, mqtt, rabbitmq, amqp",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config file path for proxy backend (default: config/proxy/svr.ini)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="(Optional) model name to load for the server",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Block size used by some backends",
    )
    parser.add_argument(
        "--no-model",
        dest="load_model",
        action="store_false",
        help="Do not load the model; only start backend",
    )
    parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        help="Start immediately without waiting for any devices",
    )
    parser.add_argument(
        "--enable-cache",
        dest="enable_cache",
        action="store_true",
        help="Enable client-side caching (for proxy backend)",
    )
    parser.add_argument(
        "--enable-hooks",
        dest="enable_hooks",
        action="store_true",
        help="Enable hooks for distributed computation (apply_hooks)",
    )
    parser.add_argument(
        "--enable-verification",
        dest="enable_verification",
        action="store_true",
        help="Enable output verification in hooks (for debugging)",
    )
    parser.set_defaults(
        load_model=True, enable_cache=False, enable_hooks=False, no_wait=False
    )
    return parser.parse_args()


def get_connection_count_safe(backend) -> int:
    if hasattr(backend, "get_connection_count"):
        return backend.get_connection_count()  # type: ignore[attr-defined]
    return 0


def main():
    args = parse_args()
    print(
        f"Starting server-only with backend={args.backend}, load_model={args.load_model}"
    )

    # Optionally load model (on CPU by default)
    model = None
    tokenizer = None
    if args.load_model and args.model_name:
        print("Loading model (this may take a while)...")
        model, tokenizer = load_model_and_tokenizer(
            args.model_name, dtype=None
        )
        print("Model loaded")

    # Initialize backend
    print("Initializing backend...")
    backend = start_backend(
        backend_name=args.backend,
        block_size=args.block_size,
        cfg_path=args.cfg,
        enable_cache=args.enable_cache,
    )
    print("Backend started. Server is now listening for device connections.")

    # Set backend for morphling hooks
    morphling.hooks.autograd._backend = backend
    morphling.hooks.autograd._enable_verification = args.enable_verification

    if args.enable_verification:
        print("✓ Output verification in hooks ENABLED")

    # Wait for minimum devices to connect
    min_devices = 1
    if args.no_wait:
        print("No-wait mode: Starting immediately without waiting for devices")
        connected = get_connection_count_safe(backend)
    else:
        connected = wait_for_connections(backend, min_devices=min_devices)

    if connected < min_devices and not args.no_wait:
        print(
            f"Warning: Only {connected} device(s) connected, but {min_devices} required."
        )
        print("Continuing anyway (new devices can join dynamically)...")
    elif connected == 0 and not args.no_wait:
        print("⚠ No devices connected, but starting in dynamic mode")
        print("Inference will proceed when devices connect")

    if connected > 0 and model is not None and tokenizer is not None:
        print("\n=== Running Text Generation Inference ===")
        # Print current device status
        current_devices = (
            get_connection_count_safe(backend)
            if hasattr(backend, "get_connection_count")
            else connected
        )
        print(f"📊 Current connected devices: {current_devices}")
        print(
            f"💡 Note: Backend (C++ RephrasePartitions) will dynamically allocate tasks based on actual device count"
        )
        print()

        inputs = prepare_inputs(
            tokenizer,
            batch_size=1,
            seq_length=128,
        )
        print("inputs:", inputs)

        # Apply hooks for distributed computation (similar to run_devices.py line 236)
        if args.enable_hooks:
            print(
                "✓ Distributed computation mode: apply_hooks('linear') ENABLED"
            )
            print(
                "  → Linear layer computations will be offloaded to remote devices"
            )
            apply_hooks("linear")
        else:
            print("✗ Local computation mode: apply_hooks('linear') DISABLED")
            print("  → All computations will run locally using PyTorch")

        inputs = inputs.to("cpu")
        model = model.to("cpu")
        model = model.to(torch.float32)

        # Debug: Print input info
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"First 10 token IDs: {inputs['input_ids'][0, :10]}")

        start = time.time()
        outputs = model(**inputs, return_dict=True)
        end = time.time()
        print(f"Inference finished in {end - start:.2f}s")

        # Print device info during inference
        final_devices = (
            get_connection_count_safe(backend)
            if hasattr(backend, "get_connection_count")
            else current_devices
        )
        if final_devices > current_devices:
            print(
                f"✓ Device(s) joined during inference: {current_devices} → {final_devices}"
            )
        else:
            print(
                f"  Device count: {final_devices} (consistent throughout inference)"
            )

        if hasattr(outputs, "logits"):
            print("logits shape:", outputs.logits.shape)

            # Save logits to pt file
            os.makedirs("logits_comparison", exist_ok=True)
            suffix = "with_hooks" if args.enable_hooks else "without_hooks"
            logits_path = os.path.join(
                "logits_comparison", f"logits_{suffix}.pt"
            )
            torch.save(outputs.logits.cpu().detach(), logits_path)
            print(f"✓ Saved logits to {logits_path}")
        print("=== Inference Done ===\n")

    # Graceful shutdown handling
    stop = False

    def _signal_handler(sig, frame):
        nonlocal stop
        print(f"\nReceived signal {sig}. Shutting down...")
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("\n=== Server is running ===")
    print("Press Ctrl+C to stop")

    try:
        while not stop:
            time.sleep(1)
    finally:
        print("Stopping backend...")
        try:
            if hasattr(backend, "stop"):
                backend.stop()  # type: ignore[attr-defined]
            if hasattr(backend, "disconnect"):
                # async disconnect if available
                loop = asyncio.get_event_loop()
                loop.run_until_complete(backend.disconnect())  # type: ignore[attr-defined]
        except Exception as e:
            print("Error while stopping backend:", e)
        print("Server shutdown complete.")
        import sys

        sys.exit(0)


if __name__ == "__main__":
    # import torch lazily to avoid heavy import if not loading model
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    main()
