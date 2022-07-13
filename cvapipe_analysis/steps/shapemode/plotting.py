import re
import warnings
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats as scistats
from typing import List, Optional

def dataset_summary_table(
    df: pd.DataFrame,
    levels: List,
    factor: str,
    rank_factor_by: Optional[str]=None,
    save: Optional[Path] = None
):

    """
    Generates a summary table from a dataframe.
    
    Parameters
    --------------------
    df: pandas df
        Input dataframe to be summarized.
    levels: list
        List of column names. These names will be used to index
        rows in the summary dataframe.
    factor: str
        Column name to stratify the data by. Each value of this
        factor will be represented by one column in the summary
        dataframe.
    rank_factor_by: str
        Column name to be used to sort the columns of the summary
        dataframe by. If none is provided, then columns are
        sorted in alphabetical order.
    save: Path
        Path to save the results.

    Returns
    -------
        df_summary: pandas df
            Summary dataframe
    """
        
    # Check if all variable are available
    for col in levels+[factor]+[rank_factor_by]:
        if (col is not None) and (col not in df.columns):
            raise ValueError(f"Column {col} not found in the input dataframe.")
    
    # Fill missing data with NA
    for level in levels:
        df[level].fillna("NA", inplace=True)

    # Count number of cells
    df_summary = df.groupby(levels+[factor]).size()
    df_summary = df_summary.unstack(level=-1)

    if rank_factor_by is not None:
        # Rank columns if a ranking variable is provided
        order = (
            df.groupby([factor])[[rank_factor_by]]
            .min()
            .sort_values(by=rank_factor_by)
            .index
        )
    else:
        # Uses alphabetical order
        order = df[factor].unique()
        order = sorted(order)
        
    # Rank dataframe
    df_summary = df_summary[order]

    # Create a column for total number of cells
    df_summary = pd.concat(
        [
            df_summary,
            pd.DataFrame(
                [
                    pd.Series(
                        dict(
                            zip(
                                df_summary.columns,
                                [df_summary[c].sum() for c in df_summary.columns],
                            )
                        ),
                        name=("", "", "Total"),
                    )
                ]
            ),
        ],
        axis=0,
    )

    # Create a row for order number
    df_order = pd.DataFrame([
            pd.Series(
                dict(zip(df_summary.columns, np.arange(1, 2 + len(order)))),
                name=("", "", "Order"),
            )
        ])
    df_summary = pd.concat([df_summary, df_order], axis=0)

    # Set pandas display properties
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        styler = (
        df_summary.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-family", "Helvetica"),
                    ("font-size", "10pt"),
                    ("border", "1px solid gray"),
                ],
            }
        ]
        )
        .set_properties(**{"text-align": "left", "border": "1px solid gray"})
        #.bar(color="#D7BDE2")
        .format(lambda x: f"{x if x>0 else ''}")
    )

    # Save view of the table as jpeg as well as csv
    if save:
        try:
            import imgkit
            from xvfbwrapper import Xvfb
            vdisplay = Xvfb()
            vdisplay.start()
            imgkit.from_string(styler.render(), f"{save}.jpg")
            vdisplay.stop()
        except:
            warnings.warn('Not abel to convert pandas dataframe to an image. Please check your imgkit and xvfbwrapper installations.')
            pass
        df_summary.to_csv(f"{save}_summary.csv")
    
    return df_summary

def paired_correlation(
    df: pd.DataFrame,
    features: List,
    save: Path,
    units: Optional[List] = None,
    off: Optional[float] = 0
):

    """
    Create pairwise correlation between columns of a dataframe.
    
    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains the features.
    features: list
        List of column names. Every feature in this list will
        be plotted agains each other.
    save: Path
        Path to save the result
    units: List
        List of same length of features with a multiplication
        factor for each feature.
    save: Path
        Path to save the results.
    off: float
        The plot axes will span off% to 100-off% of the data.
    """
    
    # Check if all variable are available
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the input dataframe.")
    
    npts = df.shape[0]

    cmap = plt.cm.get_cmap("tab10")

    # Drop rows for with one or more feature are NA
    df = df.dropna(subset=features)

    if units is None:
        units = np.ones(len(features))

    # Check if all variable are available
    if len(units) != len(features):
        raise ValueError(f"Features and units should have same length.")

    # Clip the limits of the plot
    prange = []
    for f, un in zip(features, units):
        prange.append(np.percentile(un * df[f].values, [off, 100 - off]))

    # Create a grid of nfxnf
    nf = len(features)
    fig, axs = plt.subplots(
        nf,
        nf,
        figsize=(2 * nf, 2 * nf),
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    # Make the plots
    for f1id, (f1, un1) in enumerate(zip(features, units)):

        yrange = []
        for f2id, (f2, un2) in enumerate(zip(features, units)):

            ax = axs[f1id, f2id]

            y = un1 * df[f1].values
            x = un2 * df[f2].values

            valids = np.where(
                (
                    (y > prange[f1id][0])
                    & (y < prange[f1id][1])
                    & (x > prange[f2id][0])
                    & (x < prange[f2id][1])
                )
            )
            
            # Add plots on lower triangle
            if f2id < f1id:
                xmin = x[valids].min()
                xmax = x[valids].max()
                ymin = y[valids].min()
                ymax = y[valids].max()
                yrange.append([ymin, ymax])
                ax.plot(
                    x[valids], y[valids], ".", markersize=1, color="black", alpha=0.1
                )
                ax.plot([xmin, xmax], [xmin, xmax], "--")
                if f2id:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="y", which="both", length=0.0)
                if f1id < nf - 1:
                    ax.tick_params(axis="x", which="both", length=0.0)

            # Add annotations on upper triangle
            elif f2id > f1id:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="x", which="both", length=0.0)
                ax.tick_params(axis="y", which="both", length=0.0)
                pearson, p_pvalue = scistats.pearsonr(x, y)
                spearman, s_pvalue = scistats.spearmanr(x, y)
                ax.text(
                    0.05,
                    0.8,
                    f"Pearson: {pearson:.2f}",
                    size=10,
                    ha="left",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.05,
                    0.6,
                    f"P-value: {p_pvalue:.1E}",
                    size=10,
                    ha="left",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.05,
                    0.4,
                    f"Spearman: {spearman:.2f}",
                    size=10,
                    ha="left",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.05,
                    0.2,
                    f"P-value: {s_pvalue:.1E}",
                    size=10,
                    ha="left",
                    transform=ax.transAxes,
                )
                
            # Single variable distribution at diagonal
            else:
                ax.set_frame_on(False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="y", which="both", length=0.0)
                ax.hist(
                    x[valids],
                    bins=16,
                    density=True,
                    histtype="stepfilled",
                    color="white",
                    edgecolor="black",
                    label="Complete",
                )
                ax.hist(
                    x[valids],
                    bins=16,
                    density=True,
                    histtype="stepfilled",
                    color=cmap(0),
                    alpha=0.2,
                    label="Incomplete",
                )

            if f1id == nf - 1:
                ax.set_xlabel(f2, fontsize=7)
            if not f2id and f1id:
                ax.set_ylabel(f1, fontsize=7)

        if yrange:
            ymin = np.min([ymin for (ymin, ymax) in yrange])
            ymax = np.max([ymax for (ymin, ymax) in yrange])
            for f2id, f2 in enumerate(features):
                ax = axs[f1id, f2id]
                if f2id < f1id:
                    ax.set_ylim(ymin, ymax)

    # Global annotation
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.title(f"Total number of points: {npts}", fontsize=24)

    # Save
    plt.savefig(f"{save}.png", dpi=300)
    plt.close("all")


# -------------------------------
# NOT YET DOCUMENTED
# -------------------------------
    
#  plot function
def splot(
    selected_metrics,
    selected_metrics_abb,
    selected_structures,
    structure_metric,
    cells,
    save_flag,
    pic_root,
    name,
    markersize,
    remove_cells,
):

    # Rows and columns
    nrows = len(selected_metrics)
    ncols = len(selected_structures)

    # Plotting parameters
    fac = 1000
    ms = markersize
    fs2 = np.round(np.interp(nrows * ncols, [6, 21, 50], [25, 15, 10]))
    fs = np.round(fs2 * 2 / 3)
    # lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    # Plotting flags
    # W = 500

    # Time for a flexible scatterplot
    w1 = 0.001
    w2 = 0.01
    w3 = 0.001
    h1 = 0.001
    h2 = 0.01
    h3 = 0.001
    xp = 0.1
    yp = 0.1
    xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
    xw = xx * xp
    xx = xx * (1 - xp)
    yw = yy * yp
    yy = yy * (1 - yp)

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            pos = np.argwhere((cells["structure_name"] == struct).to_numpy())
            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            # selcel = (cells['structure_name'] == struct).to_numpy()
            # struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

            # metricX = metric
            # metricY = struct
            abbX = selected_metrics_abb[yi]
            abbY = selected_structures[xi]

            # select subplot
            i = i + 1
            row = nrows - np.ceil(i / ncols) + 1
            row = row.astype(np.int64)
            col = i % ncols
            if col == 0:
                col = ncols
            print(f"{i}_{row}_{col}")

            # Main scatterplot
            ax = fig.add_axes(
                [
                    w1 + ((col - 1) * (xw + xx + w2)) + xw,
                    h1 + ((row - 1) * (yw + yy + h2)) + yw,
                    xx,
                    yy,
                ]
            )
            ax.plot(x, y, "b.", markersize=ms)
            if len(remove_cells) > 0:
                cr = remove_cells[f"{metric} vs {structure_metric}"].astype(np.int)
                _, i_cr, _ = np.intersect1d(pos, cr, return_indices=True)
                if len(i_cr) > 0:
                    ax.plot(x[i_cr], y[i_cr], "r.", markersize=2 * ms)

            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid()

            ax.text(
                xlim[1],
                ylim[1],
                f"n= {len(x)}",
                fontsize=fs,
                verticalalignment="top",
                horizontalalignment="right",
            )
            if len(remove_cells) > 0:
                ax.text(
                    xlim[0],
                    ylim[1],
                    f"n= {len(i_cr)}",
                    fontsize=fs,
                    verticalalignment="top",
                    horizontalalignment="left",
                    color=[1, 0, 0, 1],
                )

            # Bottom histogram
            ax = fig.add_axes(
                [
                    w1 + ((col - 1) * (xw + xx + w2)) + xw,
                    h1 + ((row - 1) * (yw + yy + h2)),
                    xx,
                    yw,
                ]
            )
            ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
            ylimBH = ax.get_ylim()
            ax.set_xticks(xticks)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xlim(left=xlim[0], right=xlim[1])
            ax.grid()
            ax.invert_yaxis()
            for n, val in enumerate(xticks):
                if val >= xlim[0] and val <= xlim[1]:
                    if int(val) == val:
                        val = int(val)
                    else:
                        val = np.round(val, 2)
                    ax.text(
                        val,
                        ylimBH[0],
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )

            ax.text(
                np.mean(xlim),
                ylimBH[1],
                f"{abbX}",
                fontsize=fs2,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            ax.axis("off")

            # Side histogram
            ax = fig.add_axes(
                [
                    w1 + ((col - 1) * (xw + xx + w2)),
                    h1 + ((row - 1) * (yw + yy + h2)) + yw,
                    xw,
                    yy,
                ]
            )
            ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 0.5], orientation="horizontal")
            xlimSH = ax.get_xlim()
            ax.set_yticks(yticks)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.grid()
            ax.invert_xaxis()
            for n, val in enumerate(yticks):
                if val >= ylim[0] and val <= ylim[1]:
                    if int(val) == val:
                        val = int(val)
                    else:
                        val = np.round(val, 2)
                    ax.text(
                        xlimSH[0],
                        val,
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )

            ax.text(
                xlimSH[1],
                np.mean(ylim),
                f"{abbY}",
                fontsize=fs2,
                horizontalalignment="left",
                verticalalignment="center",
                rotation=90,
            )
            ax.axis("off")

    if save_flag:
        plot_save_path = pic_root / f"{name}.png"
        plt.savefig(plot_save_path, format="png", dpi=150)
        plt.close("all")
    else:
        plt.show()


# function defintion
def oplot(
    cellnuc_metrics,
    cellnuc_abbs,
    pairs,
    cells,
    save_flag,
    pic_root,
    name,
    markersize,
    remove_cells,
):

    # Selecting number of pairs
    no_of_pairs, _ = pairs.shape
    nrows = np.floor(np.sqrt(2 / 3 * no_of_pairs))
    if nrows == 0:
        nrows = 1
    ncols = np.floor(nrows * 3 / 2)
    while nrows * ncols < no_of_pairs:
        ncols += 1

    # Plotting parameters
    fac = 1000
    ms = markersize
    fs2 = np.round(np.interp(nrows * ncols, [6, 21, 50], [25, 12, 8]))
    fs = np.round(fs2 * 2 / 3)
    # lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    # Plotting flags
    # W = 500

    # Time for a flexible scatterplot
    w1 = 0.001
    w2 = 0.01
    w3 = 0.001
    h1 = 0.001
    h2 = 0.01
    h3 = 0.001
    xp = 0.1
    yp = 0.1
    xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
    xw = xx * xp
    xx = xx * (1 - xp)
    yw = yy * yp
    yy = yy * (1 - yp)

    fig = plt.figure(figsize=(16, 9))

    for i, xy_pair in enumerate(pairs):

        print(i)

        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        abbX = cellnuc_abbs[xy_pair[0]]
        abbY = cellnuc_abbs[xy_pair[1]]

        # data
        x = cells[metricX].to_numpy() / fac
        y = cells[metricY].to_numpy() / fac

        # select subplot
        row = nrows - np.ceil((i + 1) / ncols) + 1
        row = row.astype(np.int64)
        col = (i + 1) % ncols
        if col == 0:
            col = ncols
        col = col.astype(np.int64)
        print(f"{i}_{row}_{col}")

        # Main scatterplot
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xx,
                yy,
            ]
        )
        ax.plot(x, y, "b.", markersize=ms)

        if len(remove_cells) > 0:
            try:
                cr = remove_cells[f"{metricX} vs {metricY}"].astype(np.int)
            except:
                cr = remove_cells[f"{metricY} vs {metricX}"].astype(np.int)
            ax.plot(x[cr], y[cr], "r.", markersize=2 * ms)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()

        ax.text(
            xlim[1],
            ylim[1],
            f"n= {len(x)}",
            fontsize=fs,
            verticalalignment="top",
            horizontalalignment="right",
            color=[0.75, 0.75, 0.75, 0.75],
        )
        if len(remove_cells) > 0:
            ax.text(
                xlim[0],
                ylim[1],
                f"n= {len(cr)}",
                fontsize=fs,
                verticalalignment="top",
                horizontalalignment="left",
                color=[1, 0, 0, 1],
            )

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
        ylimBH = ax.get_ylim()
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.invert_yaxis()
        for n, val in enumerate(xticks):
            if val >= xlim[0] and val <= xlim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                ax.text(
                    val,
                    ylimBH[0],
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color=[0.75, 0.75, 0.75, 0.75],
                )

        ax.text(
            np.mean(xlim),
            ylimBH[1],
            f"{abbX}",
            fontsize=fs2,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.axis("off")

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 0.5], orientation="horizontal")
        xlimSH = ax.get_xlim()
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.invert_xaxis()
        for n, val in enumerate(yticks):
            if val >= ylim[0] and val <= ylim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                ax.text(
                    xlimSH[0],
                    val,
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=[0.75, 0.75, 0.75, 0.75],
                )

        ax.text(
            xlimSH[1],
            np.mean(ylim),
            f"{abbY}",
            fontsize=fs2,
            horizontalalignment="left",
            verticalalignment="center",
            rotation=90,
        )
        ax.axis("off")

    if save_flag:
        plot_save_path = pic_root / f"{name}.png"
        plt.savefig(plot_save_path, format="png", dpi=150)
        plt.close("all")
    else:
        plt.show()

def vertical_distributions(
    df_input,
    yvar,
    ylabel,
    units,
    factor,
    sortby,
    function,
    clip=None,
    force_float=False,
    colorby=None,
    color_fix=None,
    references=None,
    highlight=None,
    top_text=False,
    use_std=False,
    save=None,
):

    # Preparation

    variables = np.unique([factor, yvar, sortby[0], colorby])

    df = df_input[variables].copy()

    df = df.dropna(subset=variables)

    # Apply function
    if function is not None:
        df[yvar] = function(df[yvar])
    df[yvar] = units * df[yvar]

    if clip is not None:
        pcts = np.percentile(df[yvar], clip)

    if force_float:
        df = df.astype({sortby[0]: "float"})

    df["jitter"] = np.random.normal(size=df.shape[0])

    xmin = df.jitter.min()
    xmax = df.jitter.max()

    nlevels = len(df[factor].unique())

    fig, axs = plt.subplots(
        1, nlevels, figsize=(0.7 * nlevels, 6), gridspec_kw={"wspace": 0.0}
    )

    sorter = (
        df[[factor, sortby[0]]]
        .groupby(factor)
        .agg(sortby[1])
        .sort_values(by=sortby[0], ascending=True)
        .index.tolist()
    )

    if use_std:
        operation = np.mean
    else:
        operation = np.median

    # Reference
    if references is not None:
        reference_value = operation(df.loc[df[factor].isin(references), yvar].values)

    # Colors
    if colorby is not None:
        color_values = list(df[colorby].unique())
        color_values_parse = [re.sub("[^\d\.]", "", cv) for cv in color_values]
        color_values = [color_values[i] for i in np.argsort(color_values_parse)]
        ncolors = len(color_values)
        print(f"Number of colors: {ncolors}")
        if ncolors < 10:
            color_mode = "discrete"
            cmap = plt.cm.get_cmap("Dark2")
        else:
            color_mode = "continuous"
            cmap = plt.cm.get_cmap("jet")

    for ax, fac in enumerate(sorter):
        df_fac = df.loc[df[factor] == fac]
        # axs[ax].axis('off')
        axs[ax].spines["top"].set_visible(False)
        axs[ax].spines["right"].set_visible(False)
        if ax:
            axs[ax].spines["left"].set_visible(False)
            axs[ax].set_yticks([])
        axs[ax].set_xticks([])
        if colorby is not None:
            if color_mode == "discrete":
                for c, cvalue in enumerate(color_values):
                    df_color = df_fac.loc[df_fac[colorby] == cvalue]
                    nsamples = df_color.shape[0]
                    if nsamples > 0:
                        color = cmap(c)
                        if color_fix is not None:
                            if cvalue in color_fix:
                                color = color_fix[cvalue]
                        color = [color] * nsamples
                        axs[ax].scatter(
                            df_color.jitter, df_color[yvar], s=1, alpha=0.2, c=color
                        )
            if color_mode == "continuous":
                for c, cvalue in enumerate(color_values):
                    df_color = df_fac.loc[df_fac[colorby] == cvalue]
                    nsamples = df_color.shape[0]
                    if nsamples > 0:
                        color = cmap(c / (ncolors - 1))
                        if color_fix is not None:
                            if cvalue in color_fix:
                                color = color_fix[cvalue]
                        color = [color] * nsamples
                        axs[ax].scatter(
                            df_color.jitter, df_color[yvar], s=1, alpha=0.2, c=color
                        )
        else:
            axs[ax].scatter(df_fac.jitter, df_fac[yvar], s=1, alpha=0.2, c="#6495ED")
        ymid = operation(df_fac[yvar].values)
        y_qi, y_q1, y_q3, y_qs = np.percentile(df_fac[yvar].values, [10, 25, 75, 90])

        if use_std:
            y_qi = ymid - 2 * df_fac[yvar].values.std()
            y_q1 = ymid - df_fac[yvar].values.std()
            y_q3 = ymid + df_fac[yvar].values.std()
            y_qs = ymid + 2 * df_fac[yvar].values.std()

        if fac in references:
            print(f"Data clips: {pcts[0]:.2f} - {pcts[1]:.2f}")
            print(
                f"[{fac}] - Qi: {y_qi:.2f}, Q1: {y_q1:0.2f}, Q3: {y_q3:0.2f}, Qs: {y_qs:.2f}"
            )

        axs[ax].scatter(0, ymid, s=30, c="k")
        deltax = 0.25 * (xmax - xmin)
        axs[ax].plot([-deltax, deltax], [y_qi, y_qi], color="k", alpha=0.5, linewidth=3)
        axs[ax].plot([-deltax, deltax], [y_q1, y_q1], color="k", alpha=0.8, linewidth=6)
        axs[ax].plot([-deltax, deltax], [y_q3, y_q3], color="k", alpha=0.8, linewidth=6)
        axs[ax].plot([-deltax, deltax], [y_qs, y_qs], color="k", alpha=0.5, linewidth=3)

        color = "black"
        if highlight is not None:
            if (y_q3 < reference_value) | (y_q1 > reference_value):
                color = "red"
        axs[ax].set_xlabel(
            fac,
            rotation=90,
            fontdict={"size": 18, "color": color},
            labelpad=5,
            ha="center",
        )

        if top_text:
            toptext = df_fac[sortby[0]].mean()
            axs[ax].text(
                0.5,
                1.00,
                f"{toptext:.2f}",
                size=10,
                ha="center",
                transform=axs[ax].transAxes,
            )

        axs[ax].set_xlim(xmin, xmax)
        if clip:
            axs[ax].set_ylim(pcts)

    if color_mode == "discrete":
        legend_elements = []
        for c, cvalue in enumerate(color_values):
            legend_elements.append(
                matplotlib.patches.Patch(
                    facecolor=cmap(c) if cvalue not in color_fix else color_fix[cvalue],
                    edgecolor="k",
                    label=cvalue,
                )
            )
        if references is not None:
            legend_elements.append(
                matplotlib.lines.Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="k",
                    lw=4,
                    label=f'Reference:\n{",".join(references)}',
                )
            )
        axs[-1].legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=colorby,
        )

    if references is not None:
        axs[-1].add_artist(
            matplotlib.patches.ConnectionPatch(
                xyA=(xmin, reference_value),
                xyB=(xmax, reference_value),
                coordsA=axs[0].transData,
                coordsB=axs[-1].transData,
                linestyle="--",
                linewidth=2,
                color="k",
            )
        )

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel(ylabel, fontsize=24, labelpad=10)

    if save is not None:
        fig.savefig(save, bbox_inches="tight", format="png", dpi=300)
    plt.show()

    try:
        df_agg = df[[factor, sortby[0]]].groupby(factor).agg(["mean", "min"])
        df_agg = df_agg.sort_values(by=[(sortby[0], "mean")])
        display(df_agg)
    except:
        pass
