BUILD_DIR=./build-gcc-release
BENCH_ARGS="--benchmark_format=csv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"

COMPILER="g++-12"

mkdir -p ${BUILD_DIR}
CXX=${COMPILER} cmake . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
cmake --build ${BUILD_DIR} -j

for BENCHMARK in hashing hash_bucket compressed_scan dictionary_scan; do
  ${BUILD_DIR}/${BENCHMARK} ${BENCH_ARGS} | tee ${BENCHMARK}.csv
done
