#! /usr/bin/env python3

import sys
import os
from common import *
from dictionary_scan import filter_results as dict_filter, clean_up_names as dict_clean_names

BM_SUFFIX = {
    "compressed_scan": "scan",
    "dictionary_scan": "scan"
}

def get_table(systems, benchmarks):
    for benchmark in benchmarks:
        bm_results = pd.DataFrame()
        for system in systems:
            data = get_results(f"{result_path}/{system}", f"{benchmark}.csv")
            data = clean_up_results(data, BM_SUFFIX.get(benchmark, "NEVER_MATCHES"))

            if benchmark == "dictionary_scan":
                data = dict_filter(data)
                dict_clean_names(data)

            naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]
            print(f"Naive duration [{system}]: {naive_perf:.1f} us")

            if len(bm_results) == 0:
                bm_results['name'] = data['name']

            results = pd.DataFrame()
            results[system] = naive_perf / data['runtime']
            results['name'] = data['name']
            bm_results = pd.merge(bm_results, results, how='outer', on='name')

        print(f"\n===== RESULTS [{benchmark}] ====")
        print(bm_results.to_latex(index=False,
                               float_format="{:.1f}x".format,
                               na_rep="-"))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Need /path/to/results benchmark1,benchmark2")

    result_path = sys.argv[1]
    benchmarks = sys.argv[2].split(',')

    assert(len(benchmarks) > 0)

    # x86
    # x86_systems = ('icelake', 'cascadelake', 'skylake', 'rome')
    x86_systems = ('icelake', 'rome')
    # get_table(x86_systems, benchmarks)

    # NEON
    # m1_systems = ('m1', 'graviton2', 'graviton3', 'pi')
    m1_systems = ('m1', 'graviton3')
    # get_table(m1_systems, benchmarks)

    get_table((*x86_systems, *m1_systems), benchmarks)
