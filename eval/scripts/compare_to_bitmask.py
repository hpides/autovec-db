import sys
import os
import re
sys.path.append(os.path.dirname(sys.path[0]))
from common import *


def plot_compare_to_bitmask(ax, data, name):
    scalar_perf = data[data['name'].str.contains('scalar')]['runtime'].values[0]

    for _, row in data.iterrows():
        variant = row['name']
        ax.bar(variant, scalar_perf / row['runtime'], **BAR(variant))

    ax.set_title(re.sub(r"\d+B-as-(\d+x\d+B)", r"\1", name))
    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticks(range(len(data['name'])))

    # Remove Input-64B-as-64x1B part and other small string stuff
    data['name'] = data['name'].str.replace(r"-Input-\d+B-as-\d+x\d+B", "", regex=True)
    data['name'] = data['name'].str.replace(r"::Benchmark", "")
    data['name'] = data['name'].str.replace(r"sized-(.+)?-vec", r"\1", regex=True)

    # ax.set_xticklabels(data['name'], rotation=60, rotation_mode='anchor', ha='right')
    ax.set_xticklabels(data['name'], rotation=DEFAULT_LABEL_ROTATION)
    ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    x86_results = get_results(result_path, "compare_to_bitmask_x86_cascadelake.csv")
    x86_results = clean_up_results(x86_results, "bitmask")

    m1_results = get_results(result_path, "compare_to_bitmask_m1.csv")
    m1_results = clean_up_results(m1_results, "bitmask")

    fig, (x86_axes, m1_axes) = plt.subplots(2, 4, figsize=(DOUBLE_FIG_WIDTH, 2*DOUBLE_FIG_HEIGHT))
    x86_16x1B_ax, x86_16x4B_ax, x86_64x1B_ax, x86_64x4B_ax = x86_axes
    m1_16x1B_ax, m1_16x4B_ax, m1_64x1B_ax, m1_64x4B_ax = m1_axes

    run_names = ['16B-as-16x1B', '16B-as-4x4B', '64B-as-64x1B', '64B-as-16x4B']
    for sub_axes, sub_results in [(x86_axes, x86_results), (m1_axes, m1_results)]:
        for name, ax in zip(run_names, sub_axes):
            # Gotta love this pandas syntax. Super obvious to add all the new symbol overloads...
            filter_name = sub_results.name.str.contains(name)
            filter_256 = ~sub_results.name.str.contains("256")
            filter_bitset = ~sub_results.name.str.contains("bitset")
            plot_compare_to_bitmask(ax, sub_results[filter_name & filter_256 & filter_bitset], name)

    fig.text(0, 0.5, "Speedup by factor x", rotation=90, va='center')
    fig.text(0.5, 1, "a) x86", ha='center')
    fig.text(0.5, 0.5, "b) M1", ha='center')

    fig.tight_layout(w_pad=-0.1)

    # x86_ax.set_ylim(0, 8.5)
    # x86_ax.set_yticks(range(0, 9, 2))

    # m1_ax.set_ylim(0, 16)
    # m1_ax.set_yticks(range(0, 16, 5))

    for ax in (*x86_axes, *m1_axes):
        Y_GRID(ax)
    #     HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "compare_to_bitmask")
    SAVE_PLOT(plot_path)
