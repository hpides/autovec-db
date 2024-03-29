#! /usr/bin/env python3

import sys
import os
from common import *


def plot_compressed_scan(ax, data):
    naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]

    max_diff = 1
    for _, row in data.iterrows():
        variant = row['name']
        speedup = naive_perf / row['runtime']
        ax.bar(variant, speedup, **BAR(variant))
        max_diff = max(max_diff, speedup)

    y_pos = 1 + (max_diff / 20)
    if IS_PAPER_PLOT():
        ax.text(0, y_pos, f"\\us{{{int(naive_perf)}}}", size=10, **NAIVE_PERF_TEXT)
    else:
        ax.text(0, y_pos, f"{int(naive_perf)}us", **NAIVE_PERF_TEXT)

    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['name'], rotation=45, rotation_mode='anchor', ha='right')
    # ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    x86_results = get_results(result_path, f"{x86_arch}/compressed_scan.csv")
    x86_results = clean_up_results(x86_results, "scan")

    m1_results = get_results(result_path, "m1/compressed_scan.csv")
    m1_results = clean_up_results(m1_results, "scan")

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    def filter_results(df):
        no_avx512 = ~(df['name'] == "avx512vbmi")
        return df[no_avx512]

    plot_compressed_scan(x86_ax, filter_results(x86_results))
    plot_compressed_scan(m1_ax, m1_results)

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup")

    x86_ax.set_ylim(0, 17)
    x86_ax.set_yticks(range(0, 17, 5))

    m1_ax.set_ylim(0, 17)
    m1_ax.set_yticks(range(0, 17, 5))


    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"compressed_scan_{x86_arch}")
    SAVE_PLOT(plot_path)
