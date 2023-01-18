import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from common import *


def plot_hashing(ax, data):
    for _, row in data.iterrows():
        variant = row['name']
        ax.bar(variant, row['runtime'], **BAR(variant))

    ax.set_xticks([x - 0.2 for x in range(len(data['name']))], data['name'], rotation=75, ha='center')
    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticklabels(data['name'], rotation=75)


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    x86_results = get_results(result_path, "hashing_x86.csv")
    x86_results = clean_up_results(x86_results)

    m1_results = get_results(result_path, "hashing_m1.csv")
    m1_results = clean_up_results(m1_results)

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    plot_hashing(x86_ax, x86_results)
    plot_hashing(m1_ax, m1_results)

    x86_ax.set_title("a) x86")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Runtime (ns)")

    x86_ax.set_ylim(0, 55)
    x86_ax.set_yticks(range(0, 55, 20))

    m1_ax.set_ylim(0, 30)
    m1_ax.set_yticks(range(0, 30, 10))


    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    # FIG_LEGEND(fig)

    plot_path = os.path.join(plot_dir, "hashing")
    SAVE_PLOT(plot_path)
