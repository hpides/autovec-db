BENCH_ARGS="--benchmark_format=csv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"

LLVM_COMPILER="${AUTOVEC_DB_COMPILER:-clang++}"
GCC_COMPILER="g++-12"

run_benchmarks () {
  NAME=$1
  COMPILER=$2
  BUILD_DIR="./build-${NAME}-release"
  RESULT_DIR="./results-${NAME}"

  mkdir -p ${BUILD_DIR}
  mkdir -p ${RESULT_DIR}

  CXX=${COMPILER} cmake . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
  cmake --build ${BUILD_DIR} -j

  for BENCHMARK in hashing hash_bucket compressed_scan dictionary_scan; do
    ${BUILD_DIR}/${BENCHMARK} ${BENCH_ARGS} | tee ${RESULT_DIR}/${BENCHMARK}.csv
  done
}

run_benchmarks llvm ${LLVM_COMPILER}
run_benchmarks gcc ${GCC_COMPILER}
