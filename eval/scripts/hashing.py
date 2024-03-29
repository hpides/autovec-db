#! /usr/bin/env python3

import sys
import os
from common import *


def plot_hashing(ax, data):
    naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]

    for _, row in data.iterrows():
        variant = row['name']
        ax.bar(variant, naive_perf / row['runtime'], **BAR(variant))

    scalar_perf = data[data['name'].str.contains('scalar')]['runtime'].values[0]
    y_pos = 1.05 if (scalar_perf / naive_perf) < 2 else 0.3

    if IS_PAPER_PLOT():
        ax.text(0, y_pos, f"\\ns{{{int(scalar_perf)}}}", size=10, **NAIVE_PERF_TEXT)
    else:
        ax.text(0, y_pos, f"{int(scalar_perf)}ns", **NAIVE_PERF_TEXT)

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['name'], rotation=60, rotation_mode='anchor', ha='right')
    ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    x86_results = get_results(result_path, f"{x86_arch}/hashing.csv")
    x86_results = clean_up_results(x86_results, "hash")

    m1_results = get_results(result_path, "m1/hashing.csv")
    m1_results = clean_up_results(m1_results, "hash")

    def filter_results(df):
        return df[~df["name"].str.contains("vec-64")]

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    plot_hashing(x86_ax, filter_results(x86_results))
    plot_hashing(m1_ax, filter_results(m1_results))

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup")

    x86_ax.set_ylim(0, 1.01)
    x86_ax.set_yticks([0, 1])

    m1_ax.set_ylim(0, 2)
    # m1_ax.set_yticks(range(0, 3, 0.5))

    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"hashing_{x86_arch}")
    SAVE_PLOT(plot_path)
