#!/bin/bash
set -e

RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
BUILD_DIR=${BUILD_DIR:-cmake-build-debug/clang-tidy-build}

set -x
cmake -S . -B "${BUILD_DIR}"
${RUN_CLANG_TIDY} -p "${BUILD_DIR}" -clang-tidy-binary="${CLANG_TIDY}" -header-filter='(.*/benchmarks/.*)' -quiet benchmarks/*.cpp
