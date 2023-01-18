import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from common import *

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

INTEL_BLUE = '#0071c5'
APPLE_GREY = '#555555'


def plot_hashing(ax, data, color):
    ax.bar(data['name'], data['runtime'], width=0.6, color=color)
    ax.set_xticks([x - 0.2 for x in range(len(data['name']))], data['name'], rotation=75, ha='center')
    ax.tick_params(axis='x', which=u'both',length=0)
    ax.set_xticklabels(data['name'], rotation=75)


def get_results(result_dir, file_name):
    return pd.read_csv(f"{result_dir}/{file_name}")[['name', 'cpu_time']].rename(columns={"cpu_time": "runtime"})

def clean_up_results(results):
    results = results[results.name.str.contains("mean")]

    # Generic BM_... regex replace
    results.name = results.name.replace({r"BM_.*?<(.*)>/.*" : r'\1'}, regex=True)

    results.name = results.name.str.replace("_hash", "")
    results.name = results.name.str.replace("<", "_")
    results.name = results.name.str.replace(">", "")
    results.name = results.name.str.replace("naive_", "")
    results.name = results.name.str.replace("autovec_scalar", "autovec")
    return results


if __name__ == '__main__':
    result_path, plot_dir = INIT(sys.argv)

    x86_results = get_results(result_path, "hashing_x86.csv")
    x86_results = clean_up_results(x86_results)

    m1_results = get_results(result_path, "hashing_m1.csv")
    m1_results = clean_up_results(m1_results)

    fig, (x86_ax, m1_ax) = plt.subplots(1, 2, figsize=DOUBLE_FIG_SIZE)

    plot_hashing(x86_ax, x86_results, INTEL_BLUE)
    plot_hashing(m1_ax, m1_results, APPLE_GREY)

    x86_ax.set_title("a) x86")
    m1_ax.set_title("b) M1")

    x86_ax.set_ylabel("Runtime (ns)")

    # page_in_ax.set_ylim(0, 30)
    # page_in_ax.set_yticks(range(0, 31, 5))

    # page_out_ax.set_ylim(0, 30)
    # page_out_ax.set_yticks(range(0, 31, 5))


    HATCH_WIDTH()
    for ax in (x86_ax, m1_ax):
        Y_GRID(ax)
        HIDE_BORDERS(ax)

    # FIG_LEGEND(fig)

    plot_path = os.path.join(plot_dir, "hashing")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()
