#! /usr/bin/env python3

import sys
import os
from common import *

def plot_velox_tpch(ax, compiler_results, xsimd_results):
    results = pd.DataFrame()
    results['query'] = xsimd_results['query']
    results['vec'] = compiler_results['mean']
    results['xsimd'] = xsimd_results['mean']

    colors = {"xsimd": VARIANT_COLOR['x86'], "vec": VARIANT_COLOR['vec']}
    results.plot.bar(ax=ax, color=colors, edgecolor='black', lw=2, legend=False, width=0.7)

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results['query'].str.replace('q', ''), rotation=0)

    ax.set_ylabel("Runtime [ms]")
    ax.set_xlabel("TPC-H Query")

    Y_LIM = 300
    ax.set_ylim(0, Y_LIM)
    ax.set_yticks(range(0, Y_LIM + 1, 100))

    def add_slow_text(pos):
        text_args = {'rotation': 90, 'ha': 'center', 'va': 'top',
                     'bbox': {'facecolor': 'white', 'edgecolor': 'white', 'pad': -1}}
        if int(results.iloc[pos]['vec']) > Y_LIM:
            ax.text(pos - 0.70, Y_LIM, int(results.iloc[pos]['vec']), **text_args)
            ax.text(pos + 0.75, Y_LIM, int(results.iloc[pos]['xsimd']), **text_args)

    # Some queries are too slow for the plot y-axis limit. Show runtime explicitly.
    for query in range(len(results)):
        add_slow_text(query)



if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    compiler_flags = ""
    if len(sys.argv) == 5:
        compiler_flags = sys.argv[4]
        assert(compiler_flags in ['', '_mtune-native', '_march-skylake512_mtune-native', '_march-native_mtune-native'])

    # TODO: temporarily different columns until results are in same format
    x86_columns = ('query', 'duration')
    x86_xsimd_results = get_results(result_path, f"velox/{x86_arch}/velox_xsimd{compiler_flags}.csv", x86_columns)
    x86_compiler_results = get_results(result_path, f"velox/{x86_arch}/velox_compiler{compiler_flags}.csv", x86_columns)

    # TODO: temporary
    x86_xsimd_results = x86_xsimd_results.rename(columns={"duration": "mean"})
    x86_compiler_results = x86_compiler_results.rename(columns={"duration": "mean"})

    m1_columns = ('query', 'mean')
    m1_xsimd_results = get_results(result_path, "velox/m1/velox_xsimd.csv", m1_columns)
    m1_compiler_results = get_results(result_path, "velox/m1/velox_compiler.csv", m1_columns)

    assert(len(x86_xsimd_results) == len(x86_compiler_results))
    assert(len(m1_xsimd_results) == len(m1_compiler_results))
    assert(len(m1_xsimd_results) == len(x86_xsimd_results))

    fig, axes = plt.subplots(1, 2, figsize=(2*DOUBLE_FIG_WIDTH, 3))
    x86_ax, m1_ax = axes

    plot_velox_tpch(x86_ax, x86_compiler_results, x86_xsimd_results)
    plot_velox_tpch(m1_ax, m1_compiler_results, m1_xsimd_results)

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    handles, labels = x86_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2,
               frameon=False, columnspacing=1, handletextpad=0.3)
    fig.tight_layout()

    for ax in axes:
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"velox_tpch_{x86_arch}{compiler_flags}")
    SAVE_PLOT(plot_path)
