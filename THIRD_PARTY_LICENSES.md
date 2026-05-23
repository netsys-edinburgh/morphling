# Third-Party Licenses

Morphling (DeviceEmulator) bundles or links against the following
third-party components. Each carries its own license; their respective
notices and license texts are reproduced in the locations listed below.

For the main project license, see [`LICENSE`](LICENSE) (Apache-2.0).

## Vendored sources (in-tree)

| Path | Component | Upstream | License | License file |
|------|-----------|----------|---------|--------------|
| `external/muduo_base/` | muduo base library (subset) | <https://github.com/chenshuo/muduo> | BSD-3-Clause | [`external/muduo_base/LICENSE`](external/muduo_base/LICENSE) |

## Git submodules (`external/`)

| Path | Component | Upstream | License |
|------|-----------|----------|---------|
| `external/protobuf` | Protocol Buffers | <https://github.com/protocolbuffers/protobuf> | BSD-3-Clause |
| `external/rabbitmq-c` | RabbitMQ C client | <https://github.com/alanxz/rabbitmq-c> | MIT |
| `external/redis-plus-plus` | Redis-plus-plus | <https://github.com/sewenew/redis-plus-plus> | Apache-2.0 |

Each submodule retains its upstream license file at the root of its
checkout (`external/<submodule>/LICENSE` or equivalent). Run
`git submodule update --init --recursive` to materialise them.

## Runtime dependencies (system / pip)

Morphling links at runtime against the following commonly available
libraries; their licenses are unchanged and not redistributed in this
repository:

| Component | License (typical) | Notes |
|-----------|-------------------|-------|
| PyTorch (`torch`) | BSD-3-Clause | Pulled via pip |
| Intel MKL (`mkl_rt`) | ISSL (Intel Simplified Software License) | System install |
| libevent | BSD-3-Clause | System install |
| OpenSSL | Apache-2.0 (3.x) | System install |
| CUDA Toolkit / cuBLAS | NVIDIA EULA | System install |
| Google Benchmark | Apache-2.0 | Build-time test dep |
| GoogleTest | BSD-3-Clause | Build-time test dep |

If you redistribute a binary built from this repository, you are
responsible for satisfying the redistribution obligations of every
component above. The owners of this repository make no representation
that this list is exhaustive for your build configuration.
