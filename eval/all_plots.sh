#!/usr/bin/env bash

if [[ $# -lt 2 || $# -gt 3 ]]
then
    echo "Usage: ./all_plots /path/to/results /path/to/output [x86_arch]"
    echo "  with x86_arch = (icelake|cascadelake). Default: icelake,cascadelake"
    exit 1
fi

set -e
RESULT_DIR=$1
PLOT_DIR=$2
X86_ARCH=${3:-"icelake,cascadelake"}
X86_ARCH_LIST=(${X86_ARCH//,/ })

export PYTHONPATH="$PWD/scripts"

for script in scripts/compare_to_bitmask.py scripts/compressed_scan.py scripts/dictionary_scan.py scripts/hashing.py scripts/hash_bucket.py
do
    echo "Running $script..."
    for x86_arch in ${X86_ARCH_LIST[@]}
    do
        python3 ${script} ${RESULT_DIR} ${PLOT_DIR} $x86_arch > /dev/null
    done
done

COMPILER_FLAGS=("none" "_mtune-native" "_march-native_mtune-native")

echo "Plotting Velox..."
    for x86_arch in ${X86_ARCH_LIST[@]}
    do
        python3 scripts/velox_tpch.py ${RESULT_DIR} ${PLOT_DIR} $x86_arch sf1 > /dev/null

        for flags in ${COMPILER_FLAGS[@]}
        do
            python3 scripts/velox_tpch.py ${RESULT_DIR} ${PLOT_DIR} $x86_arch sf1 $flags $flags > /dev/null
        done
    done

