#! /usr/bin/env python3

import sys
import os
from common import *

def get_table(systems):
    results = pd.DataFrame()
    for system in systems:
        data = get_results(f"{result_path}/{system}", f"{benchmark}.csv")
        data = clean_up_results(data)
        naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]
        print(f"Naive duration [{system}]: {naive_perf:.1f} us")
        data['runtime'] = naive_perf / data['runtime']

        if benchmark == "compressed_scan":
            data['name'] = data['name'].str.replace(f"-scan", "")

        if len(results) == 0:
            results['name'] = data['name']
        else:
            # Check that the runs are in the same order.
            for idx, row in data.iterrows():
                assert(results['name'][idx] == row['name'])

        results[system] = data['runtime']

    print("\n===== RESULTS ====")
    print(results.to_latex(index=False,
                           float_format="{:.1f}x".format,
                           na_rep="-"))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Need /path/to/results benchmark")

    result_path = sys.argv[1]
    benchmark = sys.argv[2]

    # x86
    get_table(('icelake', 'cascadelake', 'skylake', 'rome'))

    # NEON
    get_table(('m1', 'graviton2', 'graviton3', 'pi'))
