#! /usr/bin/env python3

import sys
import os
from common import *


def plot_hash_bucket(ax, data):
    naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]

    for _, row in data.iterrows():
        variant = row['name']
        speedup = naive_perf / row['runtime']
        bar_style = BAR(variant)

        # Only plot with min. 10% diff to avoid plotting noise.
        plotting_patched = 'patched' in row and (naive_perf / row['patched']) > (speedup * 1.1)
        if plotting_patched:
            patch_speedup = naive_perf / row['patched']
            PLOT_PATCHED_BAR(ax, variant, patch_speedup)
            bar_style['edgecolor'] = 'none'

        # Plot regular bar.
        ax.bar(variant, speedup, **bar_style)

    if IS_PAPER_PLOT():
        ax.text(0, 1.2, f"\\us{{{int(naive_perf) / 1000 :.1f}}}", size=10, **NAIVE_PERF_TEXT)
    else:
        ax.text(0, 1.2, f"{int(naive_perf) / 1000 :.1f}us", **NAIVE_PERF_TEXT)

    ax.tick_params(axis='x', which=u'both', length=0)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['name'], rotation=50, rotation_mode='anchor', ha='right')
    ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    x86_results = get_results(result_path, f"{x86_arch}/hash_bucket.csv")
    x86_results = clean_up_results(x86_results, "find")

    m1_results = get_results(result_path, "m1/hash_bucket.csv")
    m1_results = clean_up_results(m1_results, "find")

    m1_unpatched_results = get_results(result_path, "m1/hash_bucket_unpatched.csv")
    m1_unpatched_results = clean_up_results(m1_unpatched_results, "find")
    m1_results['patched'] = m1_results['runtime']
    m1_results['runtime'] = m1_unpatched_results['runtime']

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    plot_hash_bucket(x86_ax, x86_results)
    plot_hash_bucket(m1_ax, m1_results)

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup")

    x86_ax.set_ylim(0, 6.5)
    x86_ax.set_yticks(range(0, 7, 2))

    m1_ax.set_ylim(0, 4)
    m1_ax.set_yticks(range(0, 5, 1))

    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"hash_bucket_{x86_arch}")
    SAVE_PLOT(plot_path)
