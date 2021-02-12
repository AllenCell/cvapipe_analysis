import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.stats import gaussian_kde

from .plotting import splot, oplot


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
        "mem_roundness_surface_area_lcc",
        "mem_shape_volume_lcc",
        "dna_roundness_surface_area_lcc",
        "dna_shape_volume_lcc",
        "str_connectivity_cc",
        "str_shape_volume",
        "mem_position_depth_lcc",
        "dna_position_depth_lcc"
    ]
    
    cells = cells[keepcolumns]

    # %% Rename columns
    cells = cells.rename(
        columns={
            "mem_roundness_surface_area_lcc": "Cell surface area",
            "mem_shape_volume_lcc": "Cell volume",
            "dna_roundness_surface_area_lcc": "Nuclear surface area",
            "dna_shape_volume_lcc": "Nuclear volume",
            "str_connectivity_cc": "Number of pieces",
            "str_shape_volume": "Structure volume",
            "str_shape_volume_lcc": "Structure volume alt",
            "mem_position_depth_lcc": "Cell height",
            "dna_position_depth_lcc": "Nucleus height"
        }
    )
    
    # %% Add a column
    cells["Cytoplasmic volume"] = cells["Cell volume"] - cells["Nuclear volume"]

    return cells

def outliers_removal(df, output_dir, log, detect_based_on_structure_features):
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
