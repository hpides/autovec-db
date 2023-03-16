#! /usr/bin/env bash
set -e
set -x

# docker pull hpides/base_velox:v1
# docker run -it -v $(pwd):/velox/ --privileged hpides/base_velox:v1 /bin/bash
# -> apt install numactl
# -> git config --global --add safe.directory /velox
# -> ./run_benchmarks.sh

# or, enroot:
# enroot import docker://hpides/base_velox:v1
# enroot create hpides+base_velox+v1.sqsh

# numactl -N 0 enroot start -m $(pwd):/velox --rw hpides+base_velox+v1 /bin/bash
# -> git config --global --add safe.directory /velox
# -> ./run_benchmarks.sh 

mkdir -p build-runscript
cd build-runscript

for BRANCH in autovec compiler_vec novec xsimd xsimd-sse2
do
    for FLAGS in "" "-mtune=native" "-march=native -mtune=native"
    do
        # skip march=native for xsimd, as it fails due to missing avx512 implementations
        if [ "${BRANCH}" == "xsimd" ] || [ "${BRANCH}" == "xsimd-sse2" ]; then
            if [ "${FLAGS}" == "-march=native -mtune=native" ]; then
                continue
            fi
        fi

        echo "----"
        git switch ${BRANCH}
        sed -i "/^#NATIVE_FLAGS_HERE_MARKER$/{n;s/.*/set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} ${FLAGS}\")/}" ../velox/CMakeLists.txt
        sed -n 20,40p ../velox/CMakeLists.txt
        echo "----"

        C=clang-15 CXX=clang++-15 cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DVELOX_CODEGEN_SUPPORT=OFF -DVELOX_BUILD_BENCHMARKS=ON
        ninja velox_tpch_benchmark

        for run in {1..10}
        do
            numactl -N -0 ./velox/benchmarks/tpch/velox_tpch_benchmark --num_drivers=4 --minloglevel=5 --bm_min-iters=10 --data_path=../tpch-sf1 --bm_regex="q\d$|q1\d|q20|q22" --dynamic_cputhreadpoolexecutor=0 --dynamic_iothreadpoolexecutor=0 --cache_gb=10 | tee -a "10x_consistent_${BRANCH}_${FLAGS}.out"
        done
    done
done
