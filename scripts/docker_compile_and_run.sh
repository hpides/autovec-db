# built with
# docker build -t autovec-db
# with `autovec-db` being some random name

# For an interactive shell, use:
# docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash

docker run -it -v "$(pwd):/autovec-db" autovec-db mkdir -p /autovec-db/build-clang-release
docker run -it -v "$(pwd):/autovec-db" -e CXX=clang++-15 autovec-db cmake -S /autovec-db -B /autovec-db/build-clang-release -DCMAKE_BUILD_TYPE=Release
docker run -it -v "$(pwd):/autovec-db" autovec-db cmake --build /autovec-db/build-clang-release -j

BENCH_ARGS="--benchmark_format=csv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"

docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash -c "cd /autovec-db; ./build-clang-release/hashing ${BENCH_ARGS} | tee hashing.csv"
docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash -c "cd /autovec-db; ./build-clang-release/hash_bucket ${BENCH_ARGS} | tee hash_bucket.csv"
docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash -c "cd /autovec-db; ./build-clang-release/compressed_scan ${BENCH_ARGS} | tee compressed_scan.csv"
docker run -it -v "$(pwd):/autovec-db" autovec-db /bin/bash -c "cd /autovec-db; ./build-clang-release/compare_to_bitmask ${BENCH_ARGS} | tee compare_to_bitmask.csv"
