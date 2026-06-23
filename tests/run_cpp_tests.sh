#!/bin/bash
# Run all C++ tests for DeviceEmulator
# Usage: ./run_cpp_tests.sh [category]
# Categories: all, unit, cuda, worker
#
# Note: Tests should be built beforehand (e.g., in Docker image).
# This script runs the test executables directly.
#
# Discovery is recursive (issue #58): some CMake categories drop
# binaries under nested subdirectories such as
# `tests/cpp/build/unit/zerocopy/` rather than the top-level build dir.
# A naive `${BUILD_DIR}/test_*` glob misses those, so we use `find` to
# walk the whole build tree.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/cpp/build"

# Default category
CATEGORY="${1:-all}"

echo "========================================"
echo "DeviceEmulator C++ Test Runner"
echo "========================================"
echo "Category: ${CATEGORY}"
echo ""

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found at ${BUILD_DIR}"
    echo "Please build tests first with CMake."
    exit 1
fi

# Recursively discover test binaries matching a `find` predicate string.
# Args: extra `find` predicates appended after the standard -name filter.
# Emits one absolute path per line, deduplicated and sorted.
discover_tests() {
    # -name 'test_*' matches both bare `test_foo` and `test_foo.something`
    # but the -type f / -executable filters drop CMake artefacts (e.g.
    # `test_foo.dir/`) and build scripts. `-not -name '*.cmake'` /
    # `-not -name '*.o'` guard against object files that happen to start
    # with `test_`.
    find "${BUILD_DIR}" \
        -type f \
        -executable \
        -name 'test_*' \
        -not -name '*.cmake' \
        -not -name '*.o' \
        -not -name '*.cpp.o*' \
        "$@" \
        2>/dev/null | sort -u
}

run_tests_from() {
    local tests
    local rc=0
    tests="$(discover_tests "$@")"
    if [ -z "${tests}" ]; then
        echo "(no matching test binaries found)"
        return 0
    fi
    while IFS= read -r test; do
        echo "Running $(basename "${test}") (${test#${BUILD_DIR}/})..."
        if ! "${test}"; then
            echo "FAIL: $(basename "${test}") exited non-zero"
            rc=1
        fi
    done <<<"${tests}"
    return ${rc}
}

echo "========================================"
echo "Running tests..."
echo "========================================"

# Run tests based on category
case "${CATEGORY}" in
    all)
        echo "Running all available tests..."
        run_tests_from
        ;;
    unit)
        echo "Running unit tests..."
        # All `test_*` binaries that are not under a `bench/` subtree. The
        # historical filter (test_worker_base + test_cublas_error15_repro)
        # was an under-coverage artefact — CMake puts many unit tests at
        # the top level of the build dir (test_worker_base, test_dispatch_gate,
        # ...) and others under nested subdirs (unit/zerocopy/...). Matching
        # by name keeps both shapes covered.
        run_tests_from -not -path '*/bench/*'
        ;;
    cuda)
        echo "Running CUDA tests..."
        # Match the historical `test_cublas_*` prefix anywhere in the tree.
        run_tests_from -name 'test_cublas_*'
        ;;
    worker)
        echo "Running worker tests..."
        # Anything under `unit/worker/` plus the legacy top-level
        # `test_worker_*` binaries (test_worker_base today).
        run_tests_from \( -path '*/unit/worker/*' -o -name 'test_worker_*' \)
        ;;
    *)
        echo "Unknown category: ${CATEGORY}"
        echo "Available categories: all, unit, cuda, worker"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Test run complete!"
echo "========================================"
