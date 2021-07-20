import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.stats import gaussian_kde

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



def initial_parsing(df):
    """
    TBD
    """

    # Load dataset
    cells = df.copy().reset_index()

    # %% Check out columns, keep a couple
    keepcolumns = [
        "CellId",
        "structure_name",
        "MEM_roundness_surface_area_lcc",
        "MEM_shape_volume_lcc",
        "NUC_roundness_surface_area_lcc",
        "NUC_shape_volume_lcc",
        "STR_connectivity_cc",
        "STR_shape_volume",
        "MEM_position_depth_lcc",
        "NUC_position_depth_lcc"
    ]
    
    cells = cells[keepcolumns]

    # %% Rename columns
    cells = cells.rename(
        columns={
            "MEM_roundness_surface_area_lcc": "Cell surface area",
            "MEM_shape_volume_lcc": "Cell volume",
            "NUC_roundness_surface_area_lcc": "Nuclear surface area",
            "NUC_shape_volume_lcc": "Nuclear volume",
            "STR_connectivity_cc": "Number of pieces",
            "STR_shape_volume": "Structure volume",
            "STR_shape_volume_lcc": "Structure volume alt",
            "MEM_position_depth_lcc": "Cell height",
            "NUC_position_depth_lcc": "Nucleus height"
        }
    )
    
    # %% Add a column
    cells["Cytoplasmic volume"] = cells["Cell volume"] - cells["Nuclear volume"]

    return cells

def outliers_removal(df, output_dir, log, detect_based_on_structure_features=True):
    """
    TBD
    """

    # Load dataset
    cells = initial_parsing(df=df)

    # %% Threshold for determing outliers
    cell_dens_th_CN = 1e-20  # for cell-nucleus metrics across all cells
    cell_dens_th_S = 1e-10  # for structure volume metrics

    # Remove outliers

    # %% Remove cells that lack a Structure Volume value
    cells_ao = cells[["CellId", "structure_name"]].copy()
    cells_ao["Outlier"] = "No"
    CellIds_remove = cells.loc[cells["Structure volume"].isnull(), "CellId"].values
    cells_ao.loc[cells_ao["CellId"].isin(CellIds_remove), "Outlier"] = "yes_missing_structure_volume"
    cells = cells.drop(cells[cells["CellId"].isin(CellIds_remove)].index)
    cells.reset_index(drop=True)
    log.info(
        f"Removing {len(CellIds_remove)} cells that lack a Structure Volume measurement value"
    )
    log.info(
        f"Shape of remaining dataframe: {cells.shape}"
    )

    # %% Feature set for cell and nuclear features
    cellnuc_metrics = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]
    cellnuc_abbs = [
        "Cell area",
        "Cell vol",
        "Cell height",
        "Nuc area",
        "Nuc vol",
        "Nuc height",
        "Cyto vol",
    ]

    # %% All metrics including height
    L = len(cellnuc_metrics)
    pairs = np.zeros((int(L * (L - 1) / 2), 2)).astype(np.int)
    i = 0
    for f1 in np.arange(L):
        for f2 in np.arange(L):
            if f2 > f1:
                pairs[i, :] = [f1, f2]
                i += 1

    # %% The typical six scatter plots
    xvec = [1, 1, 6, 1, 4, 6]
    yvec = [4, 6, 4, 0, 3, 3]
    pairs2 = np.stack((xvec, yvec)).T

    # %% Just one
    xvec = [1]
    yvec = [4]

    # %% Parameters
    nbins = 100
    N = 10000
    fac = 1000
    Rounds = 5

    # %% For all pairs compute densities
    remove_cells = cells["CellId"].to_frame().copy()
    for i, xy_pair in enumerate(pairs):

        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        log.info(f"{metricX} vs {metricY}")

        # data
        x = cells[metricX].to_numpy() / fac
        y = cells[metricY].to_numpy() / fac

        # density estimate, repeat because of probabilistic nature of density estimate
        # used here
        for r in np.arange(Rounds):
            remove_cells[f"{metricX} vs {metricY}_{r}"] = np.nan
            log.info(f"Round {r + 1} of {Rounds}")
            rs = int(r)
            xS, yS = resample(
                x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
            )
            k = gaussian_kde(np.vstack([xS, yS]))
            cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
            cell_dens = cell_dens / np.sum(cell_dens)
            remove_cells.loc[
                remove_cells.index[np.arange(len(cell_dens))],
                f"{metricX} vs {metricY}_{r}",
            ] = cell_dens

    # %% Summarize across repeats
    remove_cells_summary = cells["CellId"].to_frame().copy()
    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        log.info(f"{metricX} vs {metricY}")
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        filter_col = [
            col for col in remove_cells if col.startswith(f"{metricX} vs {metricY}")
        ]
        x = remove_cells[filter_col].to_numpy()
        pos = np.argwhere(np.any(x < cell_dens_th_CN, axis=1))
        y = x[pos, :].squeeze()

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        xr = np.log(x.flatten())
        xr = np.delete(xr, np.argwhere(np.isinf(xr)))
        axs[0].hist(xr, bins=100)
        axs[0].set_title("Histogram of cell probabilities (log scale)")
        axs[0].set_yscale("log")
        im = axs[1].imshow(np.log(y), aspect="auto")
        plt.colorbar(im)
        axs[1].set_title("Heatmap with low probability cells (log scale)")

        plot_save_path = f"{output_dir}/{metricX}_vs_{metricY}_cellswithlowprobs.png"
        plt.savefig(plot_save_path, format="png", dpi=150)
        plt.close("all")

        remove_cells_summary[f"{metricX} vs {metricY}"] = np.median(x, axis=1)

    # %% Identify cells to be removed
    CellIds_remove_dict = {}
    CellIds_remove = np.empty(0, dtype=int)
    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        CellIds_remove_dict[f"{metricX} vs {metricY}"] = np.argwhere(
            remove_cells_summary[f"{metricX} vs {metricY}"].to_numpy() < cell_dens_th_CN
        )
        CellIds_remove = np.union1d(
            CellIds_remove, CellIds_remove_dict[f"{metricX} vs {metricY}"]
        )
        log.info(len(CellIds_remove))

    # %% Plot and remove outliers
    plotname = "CellNucleus"
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        f"{plotname}_6_org_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        f"{plotname}_6_org_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        f"{plotname}_6_outliers",
        2,
        CellIds_remove_dict,
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        output_dir,
        f"{plotname}_21_org_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        output_dir,
        f"{plotname}_21_org_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        output_dir,
        f"{plotname}_21_outliers",
        2,
        CellIds_remove_dict,
    )
    log.info(cells.shape)
    CellIds_remove = (
        cells.loc[cells.index[CellIds_remove], "CellId"].squeeze().to_numpy()
    )
    cells_ao.loc[
        cells_ao["CellId"].isin(CellIds_remove), "Outlier"
    ] = "yes_abnormal_cell_or_nuclear_metric"
    cells = cells.drop(cells.index[cells["CellId"].isin(CellIds_remove)])
    log.info(
        f"Removing {len(CellIds_remove)} cells due to abnormal cell or nuclear metric"
    )
    log.info(cells.shape)
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        f"{plotname}_6_clean_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        f"{plotname}_6_clean_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        output_dir,
        f"{plotname}_21_clean_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        output_dir,
        f"{plotname}_21_clean_fine",
        0.5,
        [],
    )

    # %% Feature sets for structures
    selected_metrics = [
        "Cell volume",
        "Cell surface area",
        "Nuclear volume",
        "Nuclear surface area",
    ]
    selected_metrics_abb = ["Cell Vol", "Cell Area", "Nuc Vol", "Nuc Area"]
    selected_structures = [
        "LMNB1",
        "ST6GAL1",
        "TOMM20",
        "SEC61B",
        "ATP2A2",
        "LAMP1",
        "RAB5A",
        "SLC25A17",
        "TUBA1B",
        "TJP1",
        "NUP153",
        "FBL",
        "NPM1",
        "SON",
    ]
    structure_metric = "Structure volume"

    # %% Parameters
    N = 1000
    fac = 1000
    Rounds = 5
    
    if detect_based_on_structure_features:
    
        # We may want to skip this part when running the test dataset
        # or any small dataset that does not have enough cells per
        # structure.
    
        # %% For all pairs compute densities
        remove_cells = cells["CellId"].to_frame().copy()
        for xm, metric in enumerate(selected_metrics):
            for ys, struct in enumerate(selected_structures):

                # data
                x = (
                    cells.loc[cells["structure_name"] == struct, [metric]]
                    .squeeze()
                    .to_numpy()
                    / fac
                )
                y = (
                    cells.loc[cells["structure_name"] == struct, [structure_metric]]
                    .squeeze()
                    .to_numpy()
                    / fac
                )

                # density estimate, repeat because of probabilistic nature of density
                # estimate used here
                for r in np.arange(Rounds):
                    if ys == 0:
                        remove_cells[f"{metric} vs {structure_metric}_{r}"] = np.nan
                    rs = int(r)
                    xS, yS = resample(
                        x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
                    )
                    k = gaussian_kde(np.vstack([xS, yS]))
                    cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
                    cell_dens = cell_dens / np.sum(cell_dens)
                    remove_cells.loc[
                        cells["structure_name"] == struct,
                        f"{metric} vs {structure_metric}_{r}",
                    ] = cell_dens

    # remove_cells = pd.read_csv(data_root_extra / 'structures.csv')

    # %% Summarize across repeats
    remove_cells_summary = cells["CellId"].to_frame().copy()
    for xm, metric in enumerate(selected_metrics):
        log.info(metric)

        filter_col = [
            col
            for col in remove_cells
            if col.startswith(f"{metric} vs {structure_metric}")
        ]
        x = remove_cells[filter_col].to_numpy()
        pos = np.argwhere(np.any(x < cell_dens_th_S, axis=1))
        y = x[pos, :].squeeze()

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        xr = np.log(x.flatten())
        xr = np.delete(xr, np.argwhere(np.isinf(xr)))
        axs[0].hist(xr, bins=100)
        axs[0].set_title("Histogram of cell probabilities (log scale)")
        axs[0].set_yscale("log")
        im = axs[1].imshow(np.log(y), aspect="auto")
        plt.colorbar(im)
        axs[1].set_title("Heatmap with low probability cells (log scale)")

        plot_save_path = (
            f"{output_dir}/{metric}_vs_{structure_metric}_cellswithlowprobs.png"
        )
        plt.savefig(plot_save_path, format="png", dpi=150)

        remove_cells_summary[f"{metric} vs {structure_metric}"] = np.median(x, axis=1)

    # %% Identify cells to be removed
    CellIds_remove_dict = {}
    CellIds_remove = np.empty(0, dtype=int)
    for xm, metric in enumerate(selected_metrics):
        log.info(metric)
        CellIds_remove_dict[f"{metric} vs {structure_metric}"] = np.argwhere(
            remove_cells_summary[f"{metric} vs {structure_metric}"].to_numpy()
            < cell_dens_th_S
        )
        CellIds_remove = np.union1d(
            CellIds_remove, CellIds_remove_dict[f"{metric} vs {structure_metric}"]
        )
        log.info(len(CellIds_remove))

    # %% Plot and remove outliers
    plotname = "Structures"
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_1_org_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_2_org_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_1_org_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_2_org_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_1_outliers",
        2,
        CellIds_remove_dict,
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_2_outliers",
        2,
        CellIds_remove_dict,
    )
    log.info(cells.shape)
    CellIds_remove = (
        cells.loc[cells.index[CellIds_remove], "CellId"].squeeze().to_numpy()
    )
    cells_ao.loc[
        cells_ao["CellId"].isin(CellIds_remove), "Outlier"
    ] = "yes_abnormal_structure_volume_metrics"
    cells = cells.drop(cells.index[cells["CellId"].isin(CellIds_remove)])
    log.info(f"Removing {len(CellIds_remove)} cells due to structure volume metrics")
    log.info(cells.shape)
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_1_clean_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_2_clean_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_1_clean_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        f"{plotname}_2_clean_thick",
        2,
        [],
    )

    # %% Final diagnostic plot
    cells = initial_parsing(df=df)
    CellIds_remove_dict = {}

    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        CellIds_remove_dict[f"{metricX} vs {metricY}"] = np.argwhere(
            (cells_ao["Outlier"] == "yes_abnormal_cell_or_nuclear_metric").to_numpy()
        )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        output_dir,
        "Check_cellnucleus",
        2,
        CellIds_remove_dict,
    )

    CellIds_remove_dict = {}
    for xm, metric in enumerate(selected_metrics):
        CellIds_remove_dict[f"{metric} vs {structure_metric}"] = np.argwhere(
            (
                (cells_ao["Outlier"] == "yes_abnormal_structure_volume_metrics")
                | (cells_ao["Outlier"] == "yes_abnormal_cell_or_nuclear_metric")
            ).to_numpy()
        )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        output_dir,
        "Check_structures_1",
        2,
        CellIds_remove_dict,
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        output_dir,
        "Check_structures_2",
        2,
        CellIds_remove_dict,
    )

    cells_ao = cells_ao.set_index("CellId", drop=True)

    return cells_ao
