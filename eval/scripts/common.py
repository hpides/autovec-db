import sys
import os
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


#######################################
# Plotting
#######################################

FS = 20
MILLION = 1_000_000
SINGLE_FIG_WIDTH = 5
SINGLE_FIG_HEIGHT = 3.5
SINGLE_FIG_SIZE = (SINGLE_FIG_WIDTH, SINGLE_FIG_HEIGHT)
DOUBLE_FIG_WIDTH = 10
DOUBLE_FIG_HEIGHT = 3.5
DOUBLE_FIG_SIZE = (DOUBLE_FIG_WIDTH, DOUBLE_FIG_HEIGHT)
IMG_TYPES = ['.png', '.svg']


INTEL_BLUE = '#0071c5'
APPLE_GREY = '#555555'


def INIT_PLOT():
    matplotlib.rcParams.update({
        'font.size': FS,
        'svg.fonttype': 'none',
    })


def INIT(args):
    if len(args) != 3:
        sys.exit("Need /path/to/results /path/to/plots")

    result_path = args[1]
    plot_dir = args[2]

    os.makedirs(plot_dir, exist_ok=True)
    INIT_PLOT()

    return result_path, plot_dir


def BAR_X_TICKS_POS(bar_width, num_bars, num_xticks):
    return [i - (bar_width / 2) + ((num_bars * bar_width) / 2) for i in range(num_xticks)]


def RESIZE_TICKS(ax, x=FS, y=FS):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(x)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(y)


def HATCH_WIDTH(width=4):
    matplotlib.rcParams['hatch.linewidth'] = width


def Y_GRID(ax):
    ax.grid(axis='y', which='major')
    ax.set_axisbelow(True)


def HIDE_BORDERS(ax, show_left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(show_left)


def FIG_LEGEND(fig):
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6,
               frameon=False, columnspacing=1, handletextpad=0.3
               #, borderpad=0.1, labelspacing=0.1, handlelength=1.8
              )
    fig.tight_layout()


def SAVE_PLOT(plot_path, img_types=None):
    if img_types is None:
        img_types = IMG_TYPES

    plot_paths = []
    for img_type in img_types:
        img_path = f"{plot_path}{img_type}"
        plot_paths.append(img_path)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)

    plt.figure()
    print(f"To view new plots, run:\n\topen {' '.join(plot_paths)}")


#######################################
# Benchmark Results
#######################################

def get_results(result_dir, file_name, columns=('name', 'cpu_time')):
    df = pd.read_csv(f"{result_dir}/{file_name}")
    df = df[[col for col in columns]]
    df = df.rename(columns={"cpu_time": "runtime"})
    return df


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