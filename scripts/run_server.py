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
    parser.set_defaults(load_model=True)
    return parser.parse_args()


async def start_backend_async(backend_name: str, block_size: int):
    loop = asyncio.get_event_loop()
    backend = AutoBackend.from_name(backend_name, loop, block_size=block_size)
    await backend.connect()
    return backend


def start_backend_sync(backend_name: str, block_size: int):
    # Some backends (like mqtt/proxy) may use synchronous initialization
    backend = None
    if backend_name in ("mqtt", "proxy"):
        backend = AutoBackend.from_name(backend_name)
        # proxy/mqtt backends expose start() instead of async connect
        if hasattr(backend, "initialize"):
            if backend_name == "proxy":
                # ProxySvr.initialize() expects a config file path
                import os
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "proxy", "svr.ini")
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
    backend = start_backend_sync(args.backend, args.block_size)
    print("Backend started. Server is now listening for device connections.")

    # Graceful shutdown handling
    stop = False

    def _signal_handler(sig, frame):
        nonlocal stop
        print(f"Received signal {sig}. Shutting down...")
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

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


if __name__ == "__main__":
    # import torch lazily to avoid heavy import if not loading model
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    main()
