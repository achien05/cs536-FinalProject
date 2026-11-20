from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    1: "grey",
    2: "red",
    3: "blue",
    4: "green",
    5: "cyan",
}
LINESTYLES = {
    0: '-',
    1: '--',
    2: ':',
    3: '-.'
}


def read_log(filename):
    path = Path(filename).with_suffix(".txt")

    # Support UTF-8 and UTF-16
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")

    pattern = re.compile(
        r"([A-Za-z_]\w*)\s*:\s*"
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    )

    records = []
    for line in text.splitlines():
        values = {
            k: float(v) for k, v in pattern.findall(line)
        }
        if "iteration" in values:
            records.append(values)

    if not records:
        raise ValueError(f"No training data: {filename}")

    return pd.DataFrame(records)


def get_depth(filename):
    match = re.search(r"depth(\d+)", filename, re.I)
    if not match:
        raise ValueError(f"Missing depth: {filename}")
    return int(match.group(1))


def setup_plot(ax, figure_title, ylabel):
    ax.set_title(figure_title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(
        style="sci", axis="x", scilimits=(0, 0)
    )


def save_plot(fig, figure_title):
    fig.tight_layout()
    filename = re.sub(r"[^\w.-]+", "_", figure_title)
    fig.savefig(filename + ".png", dpi=220)
    plt.show()


# Plot a single metric
def my_plot(filename_list, display_col, figure_title):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for filename in filename_list:
        df = read_log(filename)
        depth = get_depth(filename)

        ax.plot(
            df["iteration"],
            df[display_col],
            color=COLORS.get(depth),
            label=str(depth),
            linewidth=1.5,
        )

    setup_plot(ax, figure_title, display_col)
    ax.legend(title="depth", loc="best")
    save_plot(fig, figure_title)

    return fig, ax


# Plot two metrics (solid / dashed)
def my_plot_pair(filename_list, display_col, figure_title):
    if not isinstance(display_col, tuple) or len(display_col) != 2:
        raise ValueError("display_col must be a tuple of 2 columns")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for filename in filename_list:
        df = read_log(filename)
        depth = get_depth(filename)
        color = COLORS.get(depth)

        ax.plot(
            df["iteration"],
            df[display_col[0]],
            color=color,
            linestyle=LINESTYLES[0],
            linewidth=1.5,
        )

        ax.plot(
            df["iteration"],
            df[display_col[1]],
            color=color,
            linestyle=LINESTYLES[1],
            linewidth=1.5,
        )

    setup_plot(ax, figure_title, "magnitude")

    # Legend 1: depth -> color
    depth_handles = [
        Line2D([0], [0], color=COLORS[d], lw=1.5, label=str(d))
        for d in sorted({get_depth(f) for f in filename_list})
    ]

    legend1 = ax.legend(
        handles=depth_handles,
        title="depth",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    ax.add_artist(legend1)

    # Legend 2: metric -> line style
    metric_handles = [
        Line2D([0], [0], color="black", lw=1.5,
               linestyle=LINESTYLES[0], label=display_col[0]),
        Line2D([0], [0], color="black", lw=1.5,
               linestyle=LINESTYLES[1], label=display_col[1]),
    ]

    ax.legend(
        handles=metric_handles,
        title="S.V.",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
    )

    save_plot(fig, figure_title)

    return fig, ax

# Plot n metrics (solid / dashed / weird dashed)
def my_plot_tuple(filename_list, display_col, figure_title):
    if not isinstance(display_col, tuple):
        raise ValueError("display_col must be a tuple")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for filename in filename_list:
        df = read_log(filename)
        depth = get_depth(filename)
        color = COLORS.get(depth)

        for i in range(len(display_col)):    
            if i >= len(LINESTYLES):
                i=len(LINESTYLES)-1
            ax.plot(
                df["iteration"],
                df[display_col[i]],
                color=color,
                linestyle=LINESTYLES[i],
                linewidth=1.5,
            )

    setup_plot(ax, figure_title, "magnitude")

    # Legend 1: depth -> color
    depth_handles = [
        Line2D([0], [0], color=COLORS[d], lw=1.5, label=str(d))
        for d in sorted({get_depth(f) for f in filename_list})
    ]

    legend1 = ax.legend(
        handles=depth_handles,
        title="depth",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    ax.add_artist(legend1)

    # Legend 2: metric -> line style
    metric_handles = [
        Line2D([0], [0], color="black", lw=1.5,
               linestyle=LINESTYLES[0], label=display_col[0]),
        Line2D([0], [0], color="black", lw=1.5,
               linestyle=LINESTYLES[1], label=display_col[1]),
    ]

    ax.legend(
        handles=metric_handles,
        title="S.V.",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
    )

    save_plot(fig, figure_title)

    return fig, ax