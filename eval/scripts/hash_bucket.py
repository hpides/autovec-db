import sys
import os
from common import *


def plot_hash_bucket(ax, data):
    scalar_perf = data[data['name'].str.contains('scalar')]['runtime'].values[0]

    for _, row in data.iterrows():
        variant = row['name']
        ax.bar(variant, scalar_perf / row['runtime'], **BAR(variant))

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['name'], rotation=60, rotation_mode='anchor', ha='right')
    # ax.set_xticklabels(data['name'], rotation=75)
    # ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    x86_results = get_results(result_path, f"hash_bucket_x86_{x86_arch}.csv")
    x86_results = clean_up_results(x86_results, "find")

    m1_results = get_results(result_path, "hash_bucket_m1.csv")
    m1_results = clean_up_results(m1_results, "find")

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    plot_hash_bucket(x86_ax, x86_results)
    plot_hash_bucket(m1_ax, m1_results)

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup by factor x")

    # x86_ax.set_ylim(0, 2)
    # x86_ax.set_yticks(range(0, 55, 20))

    # m1_ax.set_ylim(0, 2)
    # m1_ax.set_yticks(range(0, 30, 10))

    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"hash_bucket_{x86_arch}")
    SAVE_PLOT(plot_path)