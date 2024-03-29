import sys
import os
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_colwidth', None)


#######################################
# Plotting
#######################################

FONT_SIZE = 20
MILLION = 1_000_000
SINGLE_FIG_WIDTH = 5
SINGLE_FIG_HEIGHT = 2.5
SINGLE_FIG_SIZE = (SINGLE_FIG_WIDTH, SINGLE_FIG_HEIGHT)
DOUBLE_FIG_WIDTH = 10
DOUBLE_FIG_HEIGHT = 1.5
DOUBLE_FIG_SIZE = (DOUBLE_FIG_WIDTH, DOUBLE_FIG_HEIGHT)
IMG_TYPES = ('.png', '.svg')
DEFAULT_LABEL_ROTATION = 60


INTEL_BLUE = '#0071c5'
APPLE_GREY = '#555555'

DEFAULT_X86_ARCH = 'icelake'

NAIVE_PERF_TEXT = {"rotation": 90, "ha": 'center', "va": 'bottom'}

VARIANT_COLOR_BLACK_WHITE = {
    "naive": '#f0f0f0',
    "scalar": '#f0f0f0',

    "autovec": '#bdbdbd',
    "bitset": '#bdbdbd',

    "vec": '#737373',

    "x86": '#252525',
    "neon": '#252525',
}

VARIANT_COLOR = {
    "naive": '#ffffb2',
    "scalar": '#ffffb2',

    "autovec": '#fecc5c',
    "bitset": '#fecc5c',

    "vec": '#fd8d3c',

    "x86": '#e31a1c',
    "sse4": '#e31a1c',
    "avx2": '#e31a1c',
    "avx512": '#e31a1c',
    "neon": '#e31a1c',
}

def GET_COLOR(variant_name, use_black_white=False):
    colors = VARIANT_COLOR_BLACK_WHITE if use_black_white else VARIANT_COLOR
    for name, color in colors.items():
        if name in variant_name:
            return color
    raise RuntimeError(f"No color found for variant: {variant_name}")


def ALIGN_ROTATED_X_LABELS(ax, offset=-10):
    # Taken from: https://stackoverflow.com/a/67459618/4505331
    from matplotlib.transforms import ScaledTranslation

    dx, dy = offset, 0
    fig = ax.get_figure()
    offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)

    # Apply offset to all xticklabels
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


def ASSERT_VARIANCE_IS_LOW(results, limit_percent=5):
    stddev = results[results.name.str.contains("_stddev")].copy().reset_index()
    mean = results[results.name.str.contains("_mean")].copy().reset_index()
    variance = (stddev['runtime'] / mean['runtime']) * 100
    if (variance > limit_percent).any():
        bad_runs = mean[variance > limit_percent].copy()
        bad_runs['var%'] = variance
        print(f"Variance too high in benchmark(s). Limit: {limit_percent}%:", file=sys.stderr)
        print(bad_runs, file=sys.stderr)


def INIT_PLOT():
    matplotlib.rcParams.update({
        'font.size': FONT_SIZE,
        'svg.fonttype': 'none',
    })


def INIT(args):
    if len(args) < 3:
        sys.exit("Need /path/to/results /path/to/plots [x86_arch]")

    result_path = args[1]
    plot_dir = args[2]

    os.makedirs(plot_dir, exist_ok=True)
    INIT_PLOT()

    x86_arch = DEFAULT_X86_ARCH
    if len(args) > 3:
        x86_arch = args[3]
        assert(x86_arch == 'icelake' or x86_arch == 'cascadelake')

    return result_path, plot_dir, x86_arch


def BAR(variant):
    return {
        "color": GET_COLOR(variant),
        "edgecolor": 'black',
        "width": 0.7,
        "lw": 2
    }

# Kinda hacky way to plot this. We are basically plotting three bars.
# 1) The regular bar
# 2) An empty bar that gives us edgecolor=black with the hight of patched
# 3) A patched bar with hatches with color of variant.
# We need 2 and 3, as hatch color == edge color.
def PLOT_PATCHED_BAR(ax, variant, patch_speedup):
    hatch_bar_style = BAR(variant)
    hatch_bar_style['edgecolor'] = hatch_bar_style['color']
    hatch_bar_style['color'] = 'white'
    hatch_bar_style['hatch'] = '//'
    ax.bar(variant, patch_speedup, **hatch_bar_style, zorder=0.9)

    base_bar_style = BAR(variant)
    base_bar_style['color'] = 'none'
    ax.bar(variant, patch_speedup, **base_bar_style, zorder=1.1)


def RESIZE_TICKS(ax, x=FONT_SIZE, y=FONT_SIZE):
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


def SAVE_PLOT(plot_path, img_types=IMG_TYPES):
    plot_paths = []
    for img_type in img_types:
        img_path = f"{plot_path}{img_type}"
        plot_paths.append(img_path)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)

    plt.figure()
    print(f"To view new plots, run:\n\topen {' '.join(plot_paths)}")


def IS_PAPER_PLOT():
    return os.getenv("AUTOVEC_DB_PLOTS") == "ON"


#######################################
# Benchmark Results
#######################################

def get_results(result_dir, file_name, columns=('name', 'cpu_time')):
    df = pd.read_csv(f"{result_dir}/{file_name}", skipinitialspace=True)
    df = df[[col for col in columns]]
    df = df.rename(columns={"cpu_time": "runtime"})
    return df


def clean_up_results(results, bm_suffix="NEVER_MATCHES"):
    ASSERT_VARIANCE_IS_LOW(results)

    results = results[results.name.str.contains("mean")]

    # Generic BM_... regex replace
    results.name = results.name.replace({r"BM_.*?<(.*)>.*mean" : r'\1'}, regex=True)

    results.name = results.name.str.replace(f"_{bm_suffix}", "")
    results.name = results.name.str.replace("<", "-")
    results.name = results.name.str.replace(">", "")
    results.name = results.name.str.replace("naive_scalar", "naive")
    results.name = results.name.str.replace("autovec_scalar", "autovec")
    results.name = results.name.str.replace("forced_scalar", "scalar")
    results.name = results.name.str.replace("_", "-")
    results.name = results.name.str.replace("vector", "vec")
    results.name = results.name.str.replace(" ", "")

    results.name = results.name.str.replace("x86-128", "sse4-128")
    results.name = results.name.str.replace("x86-256", "avx2-256")
    results.name = results.name.str.replace("x86-avx2", "avx2-256")
    results.name = results.name.str.replace("x86-512", "avx512")
    results.name = results.name.str.replace("x86-avx512", "avx512")

    results.name = results.name.str.replace("x86-", "sse4-")

    return results
