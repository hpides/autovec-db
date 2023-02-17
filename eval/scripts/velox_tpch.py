import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from common import *

def plot_velox_tpch(ax, compiler_results, xsimd_results):
    results = pd.DataFrame()
    results['query'] = xsimd_results['query']
    results['vec'] = compiler_results['duration']
    results['xsimd'] = xsimd_results['duration']

    colors = {"xsimd": VARIANT_COLOR['x86'], "vec": VARIANT_COLOR['vec']}
    results.plot.bar(ax=ax, color=colors, edgecolor='black', lw=2, legend=False, width=0.7)

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results['query'].str.replace('q', ''), rotation=0)

    ax.set_ylabel("Runtime in ms")
    ax.set_xlabel("TPCH Query")

    Y_LIM = 275
    ax.set_ylim(0, Y_LIM)
    ax.set_yticks(range(0, Y_LIM, 50))

    def add_slow_text(pos):
        text_args = {'rotation': 90, 'ha': 'center', 'va': 'top',
                     'bbox': {'facecolor': 'white', 'edgecolor': 'white', 'pad': 0}}
        if int(results.iloc[pos]['vec']) > Y_LIM:
            ax.text(pos - 0.6, Y_LIM, int(results.iloc[pos]['vec']), **text_args)
            ax.text(pos + 0.6, Y_LIM, int(results.iloc[pos]['xsimd']), **text_args)

    # Q13 and Q21 are too slow for plot. Show runtime explicitly.
    for q in range(len(results)):
        add_slow_text(q)
    # add_slow_text(4)  # Q13
    # add_slow_text(9)  # Q18
    # add_slow_text(12) # Q21


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    columns = ('query', 'duration')

    x86_xsimd_results = get_results(result_path, "velox_xsimd_tpch_sf1_x86.csv", columns)
    x86_compiler_results = get_results(result_path, "velox_compiler_simd_tpch_sf1_x86.csv", columns)

    m1_xsimd_results = get_results(result_path, "velox_xsimd_tpch_sf1_m1.csv", columns)
    m1_compiler_results = get_results(result_path, "velox_compiler_simd_tpch_sf1_m1.csv", columns)

    assert(len(x86_xsimd_results) == len(x86_compiler_results))
    assert(len(m1_xsimd_results) == len(m1_compiler_results))
    assert(len(m1_xsimd_results) == len(x86_xsimd_results))

    fig, axes = plt.subplots(1, 2, figsize=(2*DOUBLE_FIG_WIDTH, DOUBLE_FIG_HEIGHT))
    x86_ax, m1_ax = axes

    plot_velox_tpch(x86_ax, x86_compiler_results, x86_xsimd_results)
    plot_velox_tpch(m1_ax, m1_compiler_results, m1_xsimd_results)

    x86_ax.set_title("a) x86")
    m1_ax.set_title("b) M1")

    FIG_LEGEND(fig)

    for ax in axes:
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "velox_tpch")
    SAVE_PLOT(plot_path)
