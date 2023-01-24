# built with
# docker build -t autovec-db
# with `autovec-db` being some random name

# For an interactive shell, use:
# docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash

BUILD_DIR="/autovec-db/build-clang-release"

docker run -it -v "$(pwd):/autovec-db" autovec-db mkdir -p ${BUILD_DIR}
docker run -it -v "$(pwd):/autovec-db" -e CXX=clang++-15 autovec-db cmake -S /autovec-db -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
docker run -it -v "$(pwd):/autovec-db" autovec-db cmake --build ${BUILD_DIR} -j

BENCH_ARGS="--benchmark_format=csv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"

docker run -it -v "$(pwd):/autovec-db" --workdir ${BUILD_DIR} autovec-db /bin/bash -c "./hashing ${BENCH_ARGS} | tee hashing.csv"
docker run -it -v "$(pwd):/autovec-db" --workdir ${BUILD_DIR} autovec-db /bin/bash -c "./hash_bucket ${BENCH_ARGS} | tee hash_bucket.csv"
docker run -it -v "$(pwd):/autovec-db" --workdir ${BUILD_DIR} autovec-db /bin/bash -c "./compressed_scan ${BENCH_ARGS} | tee compressed_scan.csv"
docker run -it -v "$(pwd):/autovec-db" --workdir ${BUILD_DIR} autovec-db /bin/bash -c "./compare_to_bitmask ${BENCH_ARGS} | tee compare_to_bitmask.csv"
