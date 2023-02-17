#!/bin/bash
set -e

RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy-15}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy-15}
BUILD_DIR=${BUILD_DIR:-cmake-build-debug/clang-tidy-build}

set -x
CXX=clang++-15 cmake -S . -B "${BUILD_DIR}"
${RUN_CLANG_TIDY} -p "${BUILD_DIR}" -clang-tidy-binary="${CLANG_TIDY}" -header-filter='(.*/benchmarks/.*)' -quiet benchmarks/*.cpp
