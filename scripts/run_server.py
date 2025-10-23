#!/usr/bin/env python3
"""
Run server-only for the DeviceEmulator project.

This script initializes the selected backend (mqtt/rabbitmq/amqp/proxy)
and, if requested, loads the model. It does NOT spawn any device client
processes — it only starts the server and waits for connections.

Example:
  python3 scripts/run_server.py --backend proxy --model_name facebook/opt-125m
"""

import argparse
import asyncio
import signal
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

import morphling
from morphling.backend import AutoBackend
from morphling.hooks import apply_hooks
from morphling.entrypoint import ModelConfigArguments, DeviceConfigArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Run server-only for DeviceEmulator")
    parser.add_argument("--backend", type=str, default="proxy",
                        help="Backend to use: proxy, mqtt, rabbitmq, amqp")
    parser.add_argument("--model_name", type=str, default=None,
                        help="(Optional) model name to load for the server")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Block size used by some backends")
    parser.add_argument("--no-model", dest="load_model", action="store_false",
                        help="Do not load the model; only start backend")
    parser.add_argument("--min_devices", type=int, default=1,
                        help="Minimum number of devices to wait for before starting")
    parser.add_argument("--test-matmul", dest="test_matmul", action="store_true",
                        help="Run a test matrix multiplication after devices connect")
    parser.add_argument("--enable-cache", dest="enable_cache", action="store_true",
                        help="Enable client-side caching (for proxy backend)")
    parser.add_argument("--enable-hooks", dest="enable_hooks", action="store_true",
                        help="Enable hooks for distributed computation (apply_hooks)")
    parser.set_defaults(load_model=True, test_matmul=False, enable_cache=False, enable_hooks=False)
    return parser.parse_args()


async def start_backend_async(backend_name: str, block_size: int):
    loop = asyncio.get_event_loop()
    backend = AutoBackend.from_name(backend_name, loop, block_size=block_size)
    await backend.connect()
    return backend


def start_backend_sync(backend_name: str, block_size: int, enable_cache: bool = False):
    # Some backends (like mqtt/proxy) may use synchronous initialization
    backend = None
    if backend_name in ("mqtt", "proxy"):
        backend = AutoBackend.from_name(backend_name)
        # proxy/mqtt backends expose start() instead of async connect
        if hasattr(backend, "initialize"):
            if backend_name == "proxy":
                # ProxySvr.initialize() expects a config file path and optional cache flag
                import os
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "proxy", "svr.ini")

                # Check if initialize accepts cache parameter
                if enable_cache and hasattr(backend, "set_cache_enabled"):
                    backend.set_cache_enabled(True)
                    print(f"✓ Client-side caching ENABLED")
                else:
                    print(f"Client-side caching disabled")

                backend.initialize(config_path)
            else:
                try:
                    backend.initialize()
                except TypeError:
                    # initialize may accept args in some versions; ignore
                    backend.initialize(None)
        if hasattr(backend, "start"):
            backend.start()
    else:
        # Use asyncio for rabbitmq/amqp style backends
        loop = asyncio.get_event_loop()
        backend = loop.run_until_complete(start_backend_async(backend_name, block_size))
    return backend


def wait_for_devices(backend, min_devices: int, timeout: int = 120):
    """Wait for minimum number of devices to connect"""
    print(f"Waiting for at least {min_devices} device(s) to connect...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if hasattr(backend, "get_connection_count"):
                connection_count = backend.get_connection_count()
                print(f"Connected devices: {connection_count}/{min_devices}")

                if connection_count >= min_devices:
                    print(f"✓ {connection_count} device(s) connected!")
                    return connection_count
            else:
                print("Backend does not support connection counting")
                return 0

            time.sleep(2)
        except Exception as e:
            print(f"Error checking connection count: {e}")
            time.sleep(2)

    print(f"Timeout waiting for devices. Connected: {backend.get_connection_count() if hasattr(backend, 'get_connection_count') else 0}/{min_devices}")
    return backend.get_connection_count() if hasattr(backend, 'get_connection_count') else 0


def test_matrix_multiplication(backend):
    """Run a simple matrix multiplication test"""
    try:
        import torch
        print("\n=== Running Matrix Multiplication Test ===")

        # Create test matrices
        mat_a = torch.randn(512, 512)
        mat_b = torch.randn(512, 512)

        print(f"Test matrices: A={mat_a.shape}, B={mat_b.shape}")
        print("Dispatching matrix multiplication to connected devices...")

        # Dispatch the computation
        if hasattr(backend, "dispatch_matmul_async"):
            oid = backend.dispatch_matmul_async(mat_a, mat_b)
            print(f"Task dispatched with oid={oid}")

            print("Waiting for results...")
            result = backend.wait_matmul(oid)

            # Verify result
            expected = torch.matmul(mat_a, mat_b)
            if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
                print("✓ Matrix multiplication test PASSED!")
            else:
                print("✗ Matrix multiplication test FAILED!")
                print(f"Max difference: {torch.max(torch.abs(result - expected))}")
        else:
            print("Backend does not support matrix multiplication dispatch")

    except Exception as e:
        print(f"Error during matrix multiplication test: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    print(f"Starting server-only with backend={args.backend}, load_model={args.load_model}")

    # Optionally load model (on CPU by default)
    model = None
    tokenizer = None
    if args.load_model and args.model_name:
        print("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=None)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print("Model loaded")

    # Initialize backend
    print("Initializing backend...")
    backend = start_backend_sync(args.backend, args.block_size, args.enable_cache)
    print("Backend started. Server is now listening for device connections.")

    # Set backend for morphling hooks
    morphling.hooks.autograd._backend = backend

    # Wait for minimum devices to connect
    connected = wait_for_devices(backend, args.min_devices)

    if connected < args.min_devices:
        print(f"Warning: Only {connected} device(s) connected, but {args.min_devices} required.")
        print("Continuing anyway...")

    # 自动推理任务分发（无需 --test-matmul）
    if connected > 0 and model is not None and tokenizer is not None:
        print("\n=== Running Text Generation Inference ===")
        input_text = ["Hello, my dog is cute. He is a good " * 128]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        print("inputs:", inputs)

        # Apply hooks for distributed computation (similar to run_devices.py line 236)
        if args.enable_hooks:
            print("✓ Distributed computation mode: apply_hooks('linear') ENABLED")
            print("  → Linear layer computations will be offloaded to remote devices")
            apply_hooks("linear")
        else:
            print("✗ Local computation mode: apply_hooks('linear') DISABLED")
            print("  → All computations will run locally using PyTorch")

        inputs = inputs.to("cpu")
        model = model.to("cpu")
        start = time.time()
        outputs = model(**inputs, return_dict=True)
        end = time.time()
        print(f"Inference finished in {end-start:.2f}s")
        #print("outputs:", outputs)
        if hasattr(outputs, "logits"):
            print("logits shape:", outputs.logits.shape)
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
                backend.stop()
            if hasattr(backend, "disconnect"):
                # async disconnect if available
                loop = asyncio.get_event_loop()
                loop.run_until_complete(backend.disconnect())
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