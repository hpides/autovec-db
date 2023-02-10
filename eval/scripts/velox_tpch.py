import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from common import *


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    columns = ('query', 'duration')
    xsimd_results = get_results(result_path, "velox_xsimd_tpch_sf1_m1.csv", columns)
    compiler_results = get_results(result_path, "velox_compiler_simd_tpch_sf1_m1.csv", columns)

    assert(len(xsimd_results) == len(compiler_results))

    results = pd.DataFrame()
    results['query'] = xsimd_results['query']
    results['vec'] = compiler_results['duration']
    results['xsimd'] = xsimd_results['duration']

    fig, ax = plt.subplots(1, 1, figsize=DOUBLE_FIG_SIZE)

    colors = {"xsimd": VARIANT_COLOR['x86'], "vec": VARIANT_COLOR['vec']}
    results.plot.bar(ax=ax, color=colors, legend=False)

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results['query'].str.replace('q', ''), rotation=0)

    ax.set_ylabel("Runtime in ms")
    ax.set_xlabel("TPCH Query")

    Y_LIM = 210
    ax.set_ylim(0, Y_LIM)
    ax.set_yticks(range(0, Y_LIM, 50))

    def add_slow_text(pos):
        text_args = {'rotation': 90, 'ha': 'center', 'va': 'top',
                     'bbox': {'facecolor': 'white', 'edgecolor': 'white', 'pad': 0}}
        ax.text(pos + 0.5, Y_LIM, int(results.iloc[pos]['vec']), **text_args)
        ax.text(pos - 0.5, Y_LIM, int(results.iloc[pos]['xsimd']), **text_args)

    # Q13 and Q21 are too slow for plot. Show runtime explicitly.
    add_slow_text(4)  # Q13
    add_slow_text(12) # Q21

    FIG_LEGEND(fig)

    Y_GRID(ax)
    HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "velox_tpch")
    SAVE_PLOT(plot_path)
