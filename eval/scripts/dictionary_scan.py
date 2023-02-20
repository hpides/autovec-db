import sys
import os
from common import *


def plot_dictionary_scan(ax, data):
    scalar_perf = data[data['name'].str.contains('scalar')]['runtime'].values[0]

    for _, row in data.iterrows():
        variant = row['name']
        ax.bar(variant, scalar_perf / row['runtime'], **BAR(variant))

    # Clean up names for labels
    data['name'] = data['name'].str.replace(r"((vec|x86)-\d+)-.*Strategy::(.*)", r"\1-\3", regex=True)
    data['name'] = data['name'].str.replace(r"SHUFFLE-MASK-\d+-BIT", "shuffle", regex=True)
    data['name'] = data['name'].str.replace(r"-COMPRESSSTORE", "")

    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticks(range(len(data['name'])))
    ax.set_xticklabels(data['name'], rotation=60, rotation_mode='anchor', ha='right')
    ALIGN_ROTATED_X_LABELS(ax)


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    x86_results = get_results(result_path, "dictionary_scan_x86.csv")
    x86_results = clean_up_results(x86_results, "scan")

    m1_results = get_results(result_path, "dictionary_scan_m1.csv")
    m1_results = clean_up_results(m1_results, "scan")

    def filter_results(df):
        no_pred = ~df['name'].str.contains("predication")
        no_loop = ~df['name'].str.contains("loop")
        no_compress_plus_store = ~df['name'].str.contains("-PLUS-STORE")
        only_512_8bit_shuffle = ~df['name'].str.contains("16-BIT") & ~df['name'].str.contains("4-BIT")
        no_avx2 = ~df['name'].str.contains("avx2")
        idx = no_pred & no_loop & only_512_8bit_shuffle & no_avx2 & no_compress_plus_store
        return df[idx]


    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE, gridspec_kw={'width_ratios': [1.5, 1]})

    plot_dictionary_scan(x86_ax, filter_results(x86_results))
    plot_dictionary_scan(m1_ax, filter_results(m1_results))

    x86_ax.set_title("a) x86")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Speedup by factor x")

    x86_ax.set_ylim(0, 21)
    x86_ax.set_yticks(range(0, 21, 5))

    m1_ax.set_ylim(0, 16)
    m1_ax.set_yticks(range(0, 16, 5))


    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "dictionary_scan")
    SAVE_PLOT(plot_path)
