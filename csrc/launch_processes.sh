
#!/bin/bash
unset LD_PRELOAD

# Navigate to the project root directory (one level up from csrc)
cd "$(dirname "$0")/.."

# Set environment variables for MKL and CUDA
export MKLROOT=/opt/intel/oneapi/mkl/latest
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Additional library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/csrc/memory:$LD_LIBRARY_PATH

# Set the log directory and clean up old log files
LOG_DIR="$(pwd)/csrc/intercept/logs"
echo "Cleaning up old log files..."

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

rm -f "$LOG_DIR"/process_*.txt
rm -f "$LOG_DIR"/gpu_process.log

# Set shared memory size and name
total_ram=$(grep MemTotal /proc/meminfo | awk '{print $2}')
shm_size=$((total_ram * 1024 / 2))
shm_name="/sgemm_shm"

if [ -e "/dev/shm$shm_name" ]; then
    rm "/dev/shm$shm_name"
fi

dd if=/dev/zero of="/dev/shm$shm_name" bs=$shm_size count=1
chmod 666 "/dev/shm$shm_name"

# Compile the shared memory library
gcc -g -fPIC -shared -o "csrc/memory/shared_memory.so" \
    "csrc/memory/initialize_memory.c" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    -I"csrc/memory" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread
sleep 1

# Compile the interceptor library
gcc -g -fPIC -shared -o "csrc/intercept/interceptor.so" \
    "csrc/intercept/interceptor.c" \
    "csrc/memory/shared_memory.so" \
    -I"csrc/memory" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -Wl,-rpath="$(pwd)/csrc/memory" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread -lspdlog
sleep 1

# Compile the worker process
gcc -g -o "csrc/intercept/gpu_process" "csrc/worker/worker.c" \
    -I"csrc/memory" \
    -I"${MKLROOT}/include" -I"${CUDA_HOME}/include" \
    "csrc/memory/shared_memory.so" \
    -L"${MKLROOT}/lib/intel64" -L"${CUDA_HOME}/lib64" \
    -Wl,-rpath="$(pwd)/csrc/memory" \
    -lmkl_rt -lcublas -lcudart -ldl -lpthread
sleep 5

# Start GPU process
echo "Starting GPU process..."
"./csrc/intercept/gpu_process" &

# Start Python script(s) with the interceptor
for i in $(seq 1 1); do
    echo "Starting Python script $i with interceptor..."
    LD_PRELOAD="$(pwd)/csrc/intercept/interceptor.so" \
    python3 "tests/cpp/interception_tests/gpt2_training_test.py" &
done
sleep 1

# Wait for all processes to complete
wait

# Clean up any remaining processes
ps aux | grep '[p]ython3.*test_script_3.py\|[g]pu_process' | awk '{print $2}' | xargs -r kill -9

echo "All processes have completed. Logs are available in the '$LOG_DIR' directory."

