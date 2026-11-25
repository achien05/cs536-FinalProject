
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

COLORS = ["red", "blue", "green", "cyan",
          "magenta", "orange", "purple", "brown"]

LINESTYLES = ["-", "--", "-."]

# Read a training log and return its numerical records as a DataFrame.
# Parameters：
# (1) 'filename' (string): Path to the training log, The '.txt' extension is optional. Supports UTF-8 and UTF-16 encoded text files.
#
# Returns: pandas.DataFrame DataFrame containing iteration and all recorded metrics.
def read_log(filename):
    path = Path(filename)

    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

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
            key: float(value)
            for key, value in pattern.findall(line)
        }

        if "iteration" in values:
            records.append(values)

    if not records:
        raise ValueError(f"No training data found: {filename}")

    return pd.DataFrame(records)


# Configure the common appearance of a plot.
# Parameters:
# (1) 'ax' : matplotlib.axes.Axes
#     The axes to configure.
# (2) 'figure_title' (String): Title displayed above the plot.
# (3) 'ylabel' (string): Label displayed on the vertical axis.
# Returns: none
def setup_plot(ax, figure_title, ylabel):
  ax.set_title(figure_title)
  ax.set_xlabel("iteration")
  ax.set_ylabel(ylabel)
  ax.grid(True, alpha=0.25)

  ax.ticklabel_format(
      style="sci",
      axis="x",
      scilimits=(0, 0)
  )

# Save the plot as a PNG file and display it.
# Parameters:
# (1) 'fig' (matplotlib.figure.Figure): Figure to save.
# (2) 'figure_title' (string): Title used to generate the PNG filename. Unsupported filename characters are replaced by '_'.
#
# Returns: None
def save_plot(fig, figure_title):
    fig.tight_layout()

    filename = re.sub(
        r"[^\w.-]+", "_", figure_title
    )

    fig.savefig(filename + ".png", dpi=220)
    plt.show()


# Plot one metric from any number of training log files.
# Each file is represented by one solid line with a different color.
#
# Parameters
# (1) 'filename_list' (list[String]): List of training log filenames. Files are plotted in the given order.
#                                     The number of files is not fixed.
# (2) 'display_properties' (string): Name of the metric to plot. Example: "test_RMSE", "erank", "full_MSE".
# (3) 'figure_title' (String): Title of the figure and basis for the PNG filename.
# (4) 'legend_name' (String): Title of the legend.
#                             Example: "depth", "lambda", "optimizer".
# (5) 'legend_arms_name'(list(String)): Labels for the files, in the same order as filename_list.
#                                       Must contain one label per file. If None, the filenames are used as labels.
# (6) 'y_scale' (String): Scale type for the Y-axis.
#                          (a) "linear" : Default linear scale.
#                          (b) "log"    : Logarithmic scale (base 10).
# Returns: 'fig' : matplotlib.figure.Figure
def my_plot(
    filename_list,
    display_properties,
    figure_title,
    legend_name="Legend",
    legend_arms_name=None,
    y_scale="linear"
):

    if not filename_list:
        raise ValueError("filename_list cannot be empty")

    if y_scale not in ("linear", "log"):
        raise ValueError("y_scale must be 'linear' or 'log'")

    if legend_arms_name is None:
        legend_arms_name = [Path(f).stem for f in filename_list]

    if len(legend_arms_name) != len(filename_list):
        raise ValueError("legend_arms_name must have one label per file")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, filename in enumerate(filename_list):
        df = read_log(filename)

        if display_properties not in df.columns:
            raise ValueError(f"Metric '{display_properties}' not found in {filename}")

        color = COLORS[i % len(COLORS)]

        ax.plot(
            df["iteration"],
            df[display_properties],
            color=color,
            linestyle="-",
            linewidth=1.5,
            label=str(legend_arms_name[i])
        )

    setup_plot(
        ax,
        figure_title,
        display_properties
    )

    ax.set_yscale(y_scale)

    ax.legend(
        title=legend_name,
        loc="best"
    )

    save_plot(fig, figure_title)

    return fig, ax

# Plot 2 or 3 metrics from any number of training log files.
# Each file has a unique color. Metrics from the same file share
# that color but use different line styles.
# Line styles:
#     First metric  : solid (-)
#     Second metric : dashed (--)
#     Third metric  : dash-dot (-.)
# Two separate legends are displayed:
#     Legend 1: File identity -> color
#     Legend 2: Metric identity -> line style
# Parameters:
# (1) 'filename_list' (list (string) ) : List of training log filenames.
#                                        Files are plotted in the given order.
#                                        The number of files is not fixed.
# (2) 'display_properties' (tuple(string)) : Tuple containing exactly 2 or 3 metric names.
# (3) 'figure_title' (String): Title of the figure and basis for the PNG filename.
# (4) 'legend_name' (list(string)): Titles of the two legends. Must contain exactly two strings.
#                                   Example: ["depth", "S.V."]

# (5) 'legend_arms_name' (list (list(string))): Two-dimensional list of legend labels.
#                                               First inner list: Labels for files/colors, in filename_list order.
#                                                                 Must have the same length as filename_list.

#                                               Second inner list: Labels for metrics/line styles, in display_properties
#                                                                  order. Must have the same length as display_properties.
#                                               Example:[["2", "3", "4", "5"],["1st", "5th", "6th"]]
# Returns: 'fig' : matplotlib.figure.Figure
def my_plot_pair(
    filename_list,
    display_properties,
    figure_title,
    legend_name=None,
    legend_arms_name=None
):
    if not filename_list:
        raise ValueError("filename_list cannot be empty")

    if not isinstance(display_properties, tuple):
        raise ValueError(
            "display_properties must be a tuple"
        )

    if len(display_properties) not in (2, 3):
        raise ValueError(
            "display_properties must contain 2 or 3 metrics"
        )

    if legend_name is None:
        legend_name = ["Files", "Metrics"]

    if len(legend_name) != 2:
        raise ValueError(
            "legend_name must contain exactly 2 titles"
        )

    if legend_arms_name is None:
        legend_arms_name = [
            [Path(f).stem for f in filename_list],
            list(display_properties)
        ]

    if (
        len(legend_arms_name) != 2
        or len(legend_arms_name[0]) != len(filename_list)
        or len(legend_arms_name[1]) != len(display_properties)
    ):
        raise ValueError(
            "legend_arms_name must contain two lists: "
            "one label per file and one label per metric"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, filename in enumerate(filename_list):
        df = read_log(filename)
        color = COLORS[i % len(COLORS)]

        for j, metric in enumerate(display_properties):
            if metric not in df.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in {filename}"
                )

            ax.plot(
                df["iteration"],
                df[metric],
                color=color,
                linestyle=LINESTYLES[j],
                linewidth=1.5
            )

    setup_plot(ax, figure_title, "magnitude")

    # Legend 1: file identity -> color
    file_handles = [
        Line2D(
            [0], [0],
            color=COLORS[i % len(COLORS)],
            linestyle="-",
            lw=1.5,
            label=str(legend_arms_name[0][i])
        )
        for i in range(len(filename_list))
    ]

    legend1 = ax.legend(
        handles=file_handles,
        title=legend_name[0],
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    ax.add_artist(legend1)

    # Legend 2: metric identity -> line style
    metric_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=LINESTYLES[j],
            lw=1.5,
            label=str(legend_arms_name[1][j])
        )
        for j in range(len(display_properties))
    ]

    ax.legend(
        handles=metric_handles,
        title=legend_name[1],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62)
    )

    save_plot(fig, figure_title)

    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', choices=["fig1", "fig2", "fig3", "hypoth1"], default="fig1", help="choice of plot result")
    args = parser.parse_args()

    # figure 1 plot:
    fg12_legend_name = ["depth","sv"]
    fg12_legend_arms_name = [["2", "3", "4", "5"],["1st", "5th"]]

    # figure 1 GD
    if args.result == "fig1":
        fg1_GD_list = ["./fg1_result/fg1_GD_depth2","./fg1_result/fg1_GD_depth3","./fg1_result/fg1_GD_depth4","./fg1_result/fg1_GD_depth5",]
        fg1_Adam_list = ["./fg1_result/fg1_Adam_depth2","./fg1_result/fg1_Adam_depth3","./fg1_result/fg1_Adam_depth4","./fg1_result/fg1_Adam_depth5",]
        my_plot(
            filename_list = fg1_GD_list,
            display_properties = "test_MSE",
            figure_title ="figure 1 GD: Test MSE",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0],
            y_scale="log"
        )
        my_plot(
            filename_list = fg1_GD_list,
            display_properties = "erank",
            figure_title = "figure 1 GD: efficient rank",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0])

        my_plot_pair(
            filename_list=fg1_GD_list,
            display_properties=("sv1", "sv5"),
            figure_title="figure 1 GD: singular values",
            legend_name = fg12_legend_name,
            legend_arms_name=fg12_legend_arms_name)

        # figure 1 Adam:

        my_plot(
            filename_list = fg1_Adam_list,
            display_properties = "test_MSE",
            figure_title ="figure 1 Adam: Test MSE",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0],
            y_scale="log"
        )
        my_plot(
            filename_list = fg1_Adam_list,
            display_properties = "erank",
            figure_title = "figure 1 Adam: efficient rank",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0]
            )

        my_plot_pair(
            filename_list=fg1_Adam_list,
            display_properties=("sv1", "sv5"),
            figure_title="figure 1 Adam: singular values",
            legend_name = fg12_legend_name,
            legend_arms_name=fg12_legend_arms_name
            )
    if args.result == "fig2":
        # figure 2 plot
        fg2_GD_list = ["./fg2_result/fg2_GD_depth2","./fg2_result/fg2_GD_depth3","./fg2_result/fg2_GD_depth4","./fg2_result/fg2_GD_depth5",]
        fg2_Adam_list = ["./fg2_result/fg2_Adam_depth2","./fg2_result/fg2_Adam_depth3","./fg2_result/fg2_Adam_depth4","./fg2_result/fg2_Adam_depth5",]

        # figure 2 GD
        my_plot(
            filename_list = fg2_GD_list,
            display_properties = "test_MSE",
            figure_title ="figure 2 GD: Test MSE (mean MSE)",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0],
            y_scale="log"
        )
        my_plot(
            filename_list = fg2_GD_list,
            display_properties = "erank",
            figure_title = "figure 2 GD: efficient rank (mean MSE)",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0]
            )

        my_plot_pair(
            filename_list=fg2_GD_list,
            display_properties=("sv1", "sv5"),
            figure_title="figure 2 GD: singular values (mean MSE)",
            legend_name = fg12_legend_name,
            legend_arms_name=fg12_legend_arms_name
            )

        # figure 2 Adam
        my_plot(
            filename_list = fg2_Adam_list,
            display_properties = "test_MSE",
            figure_title ="figure 2 Adam: Test MSE (mean MSE)",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0],
            y_scale="log"
        )
        my_plot(
            filename_list = fg2_Adam_list,
            display_properties = "erank",
            figure_title = "figure 2 Adam: efficient rank (mean MSE)",
            legend_name = fg12_legend_name[0],
            legend_arms_name = fg12_legend_arms_name[0]
            )

        my_plot_pair(
            filename_list=fg2_Adam_list,
            display_properties=("sv1", "sv5"),
            figure_title="figure 2 Adam: singular values (mean MSE)",
            legend_name = fg12_legend_name,
            legend_arms_name=fg12_legend_arms_name
            )


    if args.result == "fig3":
        # figure 3 plot

        fg3_GD_list = ["./fg3_result/fg3_GD_lam0.txt","./fg3_result/fg3_GD_lam001.txt",]
        fg3_Adam_list = ["./fg3_result/fg3_Adam_lam0.txt","./fg3_result/fg3_Adam_lam001.txt",]

        fg3_legend_name = ["lambda","sv"]
        fg3_legend_arms_name = [["0", "0.1",],["1st", "5th","6th"]]

        # figure 3 GD
        my_plot(
            filename_list = fg3_GD_list,
            display_properties = "test_MSE",
            figure_title ="figure 3 GD: Test MSE",
            legend_name = fg3_legend_name[0],
            legend_arms_name = fg3_legend_arms_name[0],
            y_scale="log"
            )
        my_plot(
            filename_list = fg3_GD_list,
            display_properties = "erank",
            figure_title = "figure 3 GD: efficient rank",
            legend_name = fg3_legend_name[0],
            legend_arms_name = fg3_legend_arms_name[0]
            )

        my_plot_pair(
            filename_list=fg3_GD_list,
            display_properties=("sv1", "sv5", "sv6"),
            figure_title="figure 3 GD: singular values",
            legend_name = fg3_legend_name,
            legend_arms_name=fg3_legend_arms_name
            )

        # figure 3 Adam
        my_plot(
            filename_list = fg3_Adam_list,
            display_properties = "test_MSE",
            figure_title ="figure 3 Adam: Test MSE",
            legend_name = fg3_legend_name[0],
            legend_arms_name = fg3_legend_arms_name[0],
            y_scale="log"
            )
        my_plot(
            filename_list = fg3_Adam_list,
            display_properties = "erank",
            figure_title = "figure 3 Adam: efficient rank",
            legend_name = fg3_legend_name[0],
            legend_arms_name = fg3_legend_arms_name[0]
            )

        my_plot_pair(
            filename_list=fg3_Adam_list,
            display_properties=("sv1", "sv5", "sv6"),
            figure_title="figure 3 Adam: singular values",
            legend_name = fg3_legend_name,
            legend_arms_name=fg3_legend_arms_name
            )

# hypothesis1: Add Gaussian Noise to see how the model with penalty performance
    if args.result == "hypoth1":
        folder = "./hypothesis"
        my_plot(
            filename_list = [f"{folder}/adam_noise0_depth3_lam0.txt",f"{folder}/adam_noise0_depth3_lam001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis1: Test MSE noise=0, penalty =0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            y_scale="log"
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise005_depth3_lam0.txt",f"{folder}/adam_noise005_depth3_lam001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis1: Test MSE noise = 0.05, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            y_scale="log"
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise01_depth3_lam0.txt",f"{folder}/adam_noise01_depth3_lam001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis1: Test MSE noise = 0.1, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            y_scale="log"
            )
        my_plot(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam0001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis1: Test MSE noise = 0.15, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            y_scale="log"
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis1: Test MSE noise = 0.2, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            y_scale="log"
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam0001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis: Test MSE noise = 0.15, penalty = 0.0001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.0001],
            y_scale="log"
            )
        my_plot(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam0001.txt",],
            display_properties = "test_MSE",
            figure_title ="Hypothesis: Test MSE noise = 0.2, penalty = 0.0001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.0001],
            y_scale="log"
            )

        # efficient rank
        my_plot(
            filename_list = [f"{folder}/adam_noise0_depth3_lam0.txt",f"{folder}/adam_noise0_depth3_lam001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise=0, penalty =0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise005_depth3_lam0.txt",f"{folder}/adam_noise005_depth3_lam001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.05, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise01_depth3_lam0.txt",f"{folder}/adam_noise01_depth3_lam001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.1, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            )
        my_plot(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.15, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.2, penalty = 0.001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.001],
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam0001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.15, penalty = 0.0001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.0001],
            )

        my_plot(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam0001.txt",],
            display_properties = "erank",
            figure_title ="Hypothesis1: erank noise = 0.2, penalty = 0.0001",
            legend_name = "lambda",
            legend_arms_name = [0.0, 0.0001],
            )

        # singular values
        my_plot_pair(
            filename_list = [f"{folder}/adam_noise0_depth3_lam0.txt",f"{folder}/adam_noise0_depth3_lam001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.0, penalty = 0.001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise005_depth3_lam0.txt",f"{folder}/adam_noise005_depth3_lam001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.05, penalty = 0.001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise01_depth3_lam0.txt",f"{folder}/adam_noise01_depth3_lam001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.1, penalty = 0.001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.15, penalty = 0.001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.2, penalty = 0.001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise015_depth3_lam0.txt",f"{folder}/adam_noise015_depth3_lam0001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.15, penalty = 0.0001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.0001], ["1st", "5th","6th"]],
        )

        my_plot_pair(
            filename_list = [f"{folder}/adam_noise02_depth3_lam0.txt",f"{folder}/adam_noise02_depth3_lam0001.txt",],
            display_properties = ("sv1", "sv5", "sv6"),
            figure_title ="Hypothesis1: SV noise = 0.2, penalty = 0.0001",
            legend_name = ["lambda", "sv"],
            legend_arms_name = [[0.0, 0.0001], ["1st", "5th","6th"]],
        )

if __name__ == "__main__":
    main()