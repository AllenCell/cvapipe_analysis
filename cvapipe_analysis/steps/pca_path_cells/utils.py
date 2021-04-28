import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def find_closest_cells(
    df,
    dist_cols=[
        "DNA_MEM_PC1",
        "DNA_MEM_PC2",
        "DNA_MEM_PC3",
        "DNA_MEM_PC4",
        "DNA_MEM_PC5",
        "DNA_MEM_PC6",
        "DNA_MEM_PC7",
        "DNA_MEM_PC8",
    ],
    metric="euclidean",
    id_col="CellId",
    N_cells=10,
    location=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
):
    """
    looks through df and finds N closest cells (rows) to location
    using all dist_cols as equally weighted embedding dimensions
    metric can be any string that works with scipy.spatial.distance.cdist
    Returns a len(N_cells) df of cell ids and columns matching dist_cols_pattern
    cells are sorted by distance and also have an additional columns of overall distance
    """

    loc_2d = np.expand_dims(location, 0)
    dists = np.squeeze(cdist(df[dist_cols], loc_2d, metric))
    dist_col = f"{metric} distance to location"
    loc_str = ", ".join([f"{loc:.3f}" for loc in location])
    df_dists = pd.DataFrame({dist_col: dists, "location": [loc_str] * len(df)})

    df_ids_and_dims = df[[id_col, *dist_cols]]

    df_out = pd.concat([df_ids_and_dims, df_dists], axis="columns")

    return (
        df_out.sort_values(by=[dist_col])
        .head(N_cells)
        .reindex([id_col, "location", dist_col, *dist_cols], axis="columns")
    )


def scan_pc_for_cells(
    df,
    pc=1,
    path=np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    dist_cols=[
        "DNA_MEM_PC1",
        "DNA_MEM_PC2",
        "DNA_MEM_PC3",
        "DNA_MEM_PC4",
        "DNA_MEM_PC5",
        "DNA_MEM_PC6",
        "DNA_MEM_PC7",
        "DNA_MEM_PC8",
    ],
    metric="euclidean",
    id_col="CellId",
    N_cells=10,
):
    """
    scans pc along path for N_cells closest to each point on path
    """

    df_out = pd.DataFrame()

    for p in path:
        point = np.zeros(len(dist_cols))
        point[pc - 1] = p

        df_point = find_closest_cells(
            df,
            dist_cols=dist_cols,
            metric=metric,
            id_col=id_col,
            N_cells=N_cells,
            location=point,
        )

        df_out = df_out.append(df_point)

    return df_out
