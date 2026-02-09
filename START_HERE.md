# Protobuf vs FlatBuffers 性能测试 - 快速使用指南

## ⚡ 最快开始方式

```bash
cd /home/yufeng.xia/DeviceEmulator/tests/cpp

# 方式1：使用独立编译脚本（推荐，不依赖整个项目编译）
./simple_build.sh both

# 方式2：只编译 debug 版本
./simple_build.sh debug

# 方式3：只编译 release 版本
./simple_build.sh release
```

## 📋 前置要求

### 检查依赖是否已安装

```bash
# 检查 Protobuf
protoc --version
pkg-config --cflags --libs protobuf

# 检查 FlatBuffers
flatc --version
```

### 如果缺少依赖，安装它们

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install protobuf-compiler libprotobuf-dev flatbuffers-compiler libflatbuffers-dev

# Fedora/RHEL
sudo dnf install protobuf-devel flatbuffers-devel

# macOS (Homebrew)
brew install protobuf flatbuffers
```

## 📁 文件清单

已创建的文件：

```
DeviceEmulator/
├── proto/
│   └── compute_gemm.fbs              # FlatBuffers schema
│
├── tests/cpp/
│   ├── serialization_perf_test.cc    # C++ 测试源代码 (650+ 行)
│   ├── build_perf_test.sh            # CMake 方式构建脚本
│   ├── simple_build.sh               # ⭐ 独立构建脚本（推荐）
│   ├── CMakeLists.txt                # CMake 配置（已更新）
│   └── SERIALIZATION_TEST_README.md  # 详细文档
│
└── 文档
    ├── SERIALIZATION_PERF_TEST_QUICK_START.md
    ├── SERIALIZATION_TEST_CREATION_SUMMARY.md
    ├── SERIALIZATION_TEST_FILES.md
    └── 本文件
```

## 🚀 三种构建方式

### 方式1：独立脚本（推荐 ⭐）

最简单的方式，不依赖整个项目的构建配置：

```bash
cd /home/yufeng.xia/DeviceEmulator/tests/cpp
./simple_build.sh both
```

**优点**：
- 不需要整个项目的 CMake 配置
- 直接调用 gcc 和 protoc
- 自动生成 protobuf 和 flatbuffers 代码
- 编译速度快

### 方式2：CMake 脚本（需要整个项目构建）

```bash
cd /home/yufeng.xia/DeviceEmulator
mkdir -p build && cd build
cmake -DMORPHLING_PYTHON_EXECUTABLE=/usr/bin/python3 ..
make serialization_perf_test_debug serialization_perf_test_release -j$(nproc)
```

**优点**：
- 完整的项目集成
- 自动依赖管理
- IDE 支持

**缺点**：
- 需要整个项目构建成功
- 可能有其他依赖问题

### 方式3：手动编译（用于学习）

```bash
cd /home/yufeng.xia/DeviceEmulator/tests/cpp

# 生成 protobuf 代码
protoc --cpp_out=. -I../../proto ../../proto/global_api.proto
protoc --cpp_out=. -I../../proto ../../proto/morphling.proto

# 生成 flatbuffers 代码
flatc -c --gen-mutable -o . ../../proto/compute_gemm.fbs

# 编译 debug 版本
g++ -std=c++17 -O0 -g3 -Wall -Wextra \
    -I. -I../../proto \
    -o test_debug \
    serialization_perf_test.cc \
    *.pb.cc \
    $(pkg-config --cflags --libs protobuf)

# 编译 release 版本
g++ -std=c++17 -O3 -march=native -DNDEBUG \
    -I. -I../../proto \
    -o test_release \
    serialization_perf_test.cc \
    *.pb.cc \
    $(pkg-config --cflags --libs protobuf)

# 运行测试
./test_debug
./test_release
```

## 📊 预期输出

```
=============================================================================
Protobuf vs FlatBuffers Serialization Performance Test
GEMM Message Size: ~1.18 MB (1179737 bytes)
Number of Iterations: 100
Number of Matrix Payloads: 4
=============================================================================

Test data created: 1179737 bytes
Number of matrix payloads: 4

Running Protobuf serialization tests...
  Iteration 10/100 completed
  Iteration 20/100 completed
  ...
  Iteration 100/100 completed

--- Protobuf Results ---
Total Iterations:           100
Average Serialization:      123.456 us
Average Deserialization:    45.678 us
...

--- FlatBuffers Results ---
Total Iterations:           100
Average Serialization:      98.765 us
Average Deserialization:    12.345 us
...

=== Performance Comparison ===
Serialization Speed:        FlatBuffers is 20.00% faster than Protobuf
Deserialization Speed:      FlatBuffers is 73.00% faster than Protobuf
...

=============================================================================
Test completed
=============================================================================
```

## 🔍 测试参数

| 参数 | 值 |
|------|-----|
| 消息大小 | ~1.18 MB (1,179,737 bytes) |
| 矩阵数量 | 4 个 Payload |
| 单个大小 | ~295 KB |
| 迭代次数 | 100 次/格式 |
| 总测试消息数 | 200 (100 protobuf + 100 flatbuffers) |

## 💾 生成的文件

编译后会生成：

```
tests/cpp/build/
├── global_api.pb.cc          # 生成的 protobuf 代码
├── global_api.pb.h
├── morphling.pb.cc
├── morphling.pb.h
├── compute_gemm_generated.h  # 生成的 flatbuffers 代码
├── serialization_perf_test_debug    # Debug 可执行文件
└── serialization_perf_test_release  # Release 可执行文件
```

## ⚙️ 编译参数说明

### Debug 版本 (`-O0`)
```bash
-std=c++17              # C++17 标准
-O0                     # 无优化
-g3                     # 完整调试符号
-Wall -Wextra           # 所有警告
```

### Release 版本 (`-O3`)
```bash
-std=c++17              # C++17 标准
-O3                     # 最大优化
-march=native           # 针对本机 CPU 优化
-DNDEBUG                # 禁用断言
-Wall -Wpedantic        # 严格警告
```

## 📈 性能基准参考

基于典型结果，FlatBuffers 应该显示：

| 指标 | 改进 |
|------|------|
| 序列化速度 | 15-30% 快速 |
| 反序列化速度 | 3-10x 快速 |
| 消息大小 | 5-20% 更小 |

## 🛠️ 故障排除

### 编译错误：找不到 protobuf

```bash
# 检查 pkg-config
pkg-config --cflags --libs protobuf

# 如果输出为空或错误，重新安装 protobuf
sudo apt-get install --reinstall protobuf-compiler libprotobuf-dev

# 检查库文件
pkg-config --modversion protobuf
```

### 编译错误：找不到 flatbuffers

```bash
# 检查 flatc
which flatc

# 如果找不到，安装 flatbuffers
sudo apt-get install flatbuffers-compiler libflatbuffers-dev

# 验证安装
flatc --version
```

### 运行时错误：找不到 libprotobuf.so

```bash
# 解决方案 1：设置库路径
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 解决方案 2：检查库位置
ldconfig -p | grep protobuf
```

## 📝 性能数据记录

在运行测试后，可以将输出重定向到文件进行记录：

```bash
cd /home/yufeng.xia/DeviceEmulator/tests/cpp

# 运行并保存结果
./simple_build.sh both > perf_results.txt 2>&1

# 查看结果
cat perf_results.txt

# 提取关键指标
grep "Average\|Speedup" perf_results.txt
```

## 🎯 下一步

1. **运行测试获取基准数据**
   ```bash
   ./simple_build.sh both
   ```

2. **分析结果**
   - 比较 debug 和 release 性能
   - 评估迁移 FlatBuffers 的收益
   - 决定是否进行迁移

3. **集成决策**
   - 如果反序列化快 3x 以上，推荐迁移
   - 如果消息大小减少 >10%，推荐迁移
   - 综合考虑迁移成本和收益

4. **实施迁移（如需要）**
   - 更新 MatrixPartition 类定义
   - 修改 `proxy_svr.cc` 和 `proxy_cli.cc`
   - 添加兼容性转换层

## 📚 详细文档

- `SERIALIZATION_PERF_TEST_QUICK_START.md` - 命令行快速参考
- `tests/cpp/SERIALIZATION_TEST_README.md` - 完整技术文档
- `SERIALIZATION_TEST_CREATION_SUMMARY.md` - 项目总结
- `SERIALIZATION_TEST_FILES.md` - 文件清单

## 💡 提示

- 在空闲系统上运行测试以获得最准确的结果
- 关闭其他应用以减少系统噪声
- 运行多次测试并比较结果的一致性
- 使用 `perf` 或 `valgrind` 进行深入分析

---

**快速链接**:
- 开始编译: `cd /home/yufeng.xia/DeviceEmulator/tests/cpp && ./simple_build.sh both`
- 查看详细文档: `cat SERIALIZATION_TEST_README.md`
- 依赖检查: `protoc --version && flatc --version`
