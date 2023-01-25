### To build the image:
# docker build -t hpides/autovec-db .
# docker login
# docker push hpides/autovec-db

### To use the image:
# git clone [...]autovec-db.git
# enroot import docker://hpides/autovec-db
# enroot create hpides+autovec-db.sqsh
# enroot start -m ./autovec-db/:/autovec-db --rw hpides+autovec-db

BUILD_DIR=./build-clang-release
BENCH_ARGS="--benchmark_format=csv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"

mkdir -p ${BUILD_DIR}
CXX=clang++-15 cmake . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
cmake --build ${BUILD_DIR} -j

for BENCHMARK in hashing hash_bucket compressed_scan compare_to_bitmask; do
  ${BUILD_DIR}/${BENCHMARK} ${BENCH_ARGS} | tee ${BENCHMARK}.csv
done
