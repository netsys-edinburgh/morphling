"""Emulator server entrypoint for Morphling.

Starts the proxy backend server (``ProxySvr``) with the configured
checkpoint path, GPU/CPU memory budget, and network listen address, then
serves device connections until interrupted.

The legacy implementation exec'd a standalone C++ ``morphling_server``
binary; that build target was removed (the gRPC server is no longer
needed) so the proxy backend is now the canonical server path, matching
``scripts/run_devices.py`` / ``scripts/_runtime_common.start_backend``.
"""

import os
import time
from typing import Any, cast

from transformers import HfArgumentParser

import morphling
from morphling.backend import AutoBackend
from morphling.common import EmulatorConfig


def _default_proxy_cfg_path() -> str:
    """Path to the proxy server INI shipped with the installed package."""
    return os.path.join(
        os.path.dirname(morphling.__path__[0]),
        "config",
        "proxy",
        "svr.ini",
    )


def main():
    parser = HfArgumentParser(cast(Any, (EmulatorConfig,)))
    # parse_args_into_dataclasses() constructs the EmulatorConfig, whose
    # __post_init__ wires the MORPHLING_CKPT_PATH / MORPHLING_SHM_SIZE /
    # MORPHLING_PIN_SIZE / MORPHLING_GPU_SIZE env vars that the C++ proxy
    # server reads at startup. (The previous parse_args() returned a bare
    # Namespace and never ran __post_init__, so those env vars were never
    # set.)
    (config,) = parser.parse_args_into_dataclasses()

    print(config)

    # The proxy server reads listen_ip/listen_port from svr.ini but honors
    # MORPHLING_PROXY_HOST / MORPHLING_PROXY_PORT overrides (see
    # csrc/core/env_cfg.cpp). Export the entrypoint's --listen_ip /
    # --listen_port so those flags take effect.
    os.environ["MORPHLING_PROXY_HOST"] = config.listen_ip
    os.environ["MORPHLING_PROXY_PORT"] = str(config.listen_port)

    if config.debug:
        os.environ["SPDLOG_LEVEL"] = "DEBUG"

    cfg_path = _default_proxy_cfg_path()

    server = AutoBackend.from_name("proxy")
    server.initialize(cfg_path)
    server.start()

    print(
        f"morphling_emulator: proxy server listening on "
        f"{config.listen_ip}:{config.listen_port} "
        f"(ckpt={config.ckpt_path}); press Ctrl-C to stop.",
        flush=True,
    )

    # server.start() launches the libevent loop on a background thread and
    # returns immediately. Block the main thread so the server keeps
    # serving; on Ctrl-C, returning drops the last reference to `server`,
    # whose destructor performs the ordered CUDA/loop teardown.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("morphling_emulator: shutting down.", flush=True)


if __name__ == "__main__":
    main()
