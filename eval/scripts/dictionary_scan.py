#! /usr/bin/env python3

import sys
import os
from common import *

def filter_results(df):
    no_pred = ~df['name'].str.contains("predication")
    no_loop = ~df['name'].str.contains("loop")
    no_compress_plus_store = ~df['name'].str.contains("-PLUS-STORE")
    only_512_16bit_shuffle = ~df['name'].str.contains("8-BIT") & ~df['name'].str.contains("4-BIT")
    no_avx2 = ~df['name'].str.contains("avx2")
    idx = no_pred & no_loop & only_512_16bit_shuffle & no_avx2 & no_compress_plus_store
    return df[idx]


def clean_up_names(data):
    # Clean up names for labels
    data['name'] = data['name'].str.replace(r"(vec)-(\d+)-.*Strategy::(.*)", r"\1-\2-\3", regex=True)
    data['name'] = data['name'].str.replace(r"(avx512)-(\d+)-.*Strategy::(.*)", r"\1-\2-\3", regex=True)
    data['name'] = data['name'].str.replace(r"SHUFFLE-MASK-\d+-BIT", "shuffle", regex=True)

    data['name'] = data['name'].str.replace(r"(avx512)-(\d+)-COMPRESSSTORE", r"vpcompressd-\2", regex=True)
    # data['name'] = data['name'].str.replace(r"-COMPRESSSTORE", "-compress")


def plot_dictionary_scan(ax, data):
    naive_perf = data[data['name'].str.contains('naive')]['runtime'].values[0]

    max_diff = 1
    for _, row in data.iterrows():
        variant = row['name']
        speedup = naive_perf / row['runtime']
        bar_style = BAR(variant)
        max_diff = max(speedup, max_diff)

        # Only plot with min. 10% diff to avoid plotting noise.
        plotting_patched = 'patched' in row and (naive_perf / row['patched']) > (speedup * 1.1)
        if plotting_patched:
            patch_speedup = naive_perf / row['patched']
            PLOT_PATCHED_BAR(ax, variant, patch_speedup)
            bar_style['edgecolor'] = 'none'

        # Plot regular bar.
        ax.bar(variant, speedup, **bar_style)

    # Clean up names for labels
    clean_up_names(data)

    y_pos = 1 + (max_diff / 20)
    if IS_PAPER_PLOT():
        ax.text(0, y_pos, f"\\us{{{int(naive_perf)}}}", size=10, **NAIVE_PERF_TEXT)
    else:
        ax.text(0, y_pos, f"{int(naive_perf)}us", **NAIVE_PERF_TEXT)

    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticks(range(len(data['name'])))
    ax.set_xticklabels(data['name'], rotation=60, rotation_mode='anchor', ha='right')
    ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir, x86_arch = INIT(sys.argv)

    x86_results = get_results(result_path, f"{x86_arch}/dictionary_scan.csv")
    x86_results = clean_up_results(x86_results, "scan")

    m1_results = get_results(result_path, "m1/dictionary_scan.csv")
    m1_results = clean_up_results(m1_results, "scan")

    m1_unpatched_results = get_results(result_path, "m1/dictionary_scan_unpatched.csv")
    m1_unpatched_results = clean_up_results(m1_unpatched_results, "scan")
    m1_results['patched'] = m1_results['runtime']
    m1_results['runtime'] = m1_unpatched_results['runtime']

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE, gridspec_kw={'width_ratios': [1.5, 1]})

    plot_dictionary_scan(x86_ax, filter_results(x86_results))
    plot_dictionary_scan(m1_ax, filter_results(m1_results))

    x86_ax.set_title(f"a) x86 {x86_arch.capitalize()}")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup")

    x86_ax.set_ylim(0, 37)
    x86_ax.set_yticks(range(0, 37, 10))

    m1_ax.set_ylim(0, 16)
    m1_ax.set_yticks(range(0, 16, 5))


    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"dictionary_scan_{x86_arch}")
    SAVE_PLOT(plot_path)
