import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_analysis(df, feature_names, prefix, npcs_to_calc=8, npcs_to_show=5, save=None):

    # Calculate PCs
    df_pca = df[feature_names]
    matrix_of_features = df_pca.values.copy()
    matrix_of_features_ids = df_pca.index
    
    pca = PCA(n_components=npcs_to_calc)
    pca = pca.fit(matrix_of_features)
    matrix_of_features_transform = pca.transform(matrix_of_features)

    pc_names = [f"{prefix}_PC{c}" for c in range(1, 1 + npcs_to_calc)]

    df_trans = pd.DataFrame(data=matrix_of_features_transform, columns=pc_names)
    df_trans.index = matrix_of_features_ids

    # Add PCs to the input dataframe
    df = df.merge(df_trans[pc_names], how="outer", left_index=True, right_index=True)
    
    # Analysis of explained variance
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    df_dimred = {}
    for comp, pc_name in enumerate(pc_names):
        load = loading[:, comp]
        pc = [v for v in load]
        apc = [v for v in np.abs(load)]
        total = np.sum(apc)
        cpc = [100 * v / total for v in apc]
        df_dimred[pc_name] = pc
        df_dimred[pc_name.replace("_PC", "_aPC")] = apc
        df_dimred[pc_name.replace("_PC", "_cPC")] = cpc
        if comp > npcs_to_show - 1:
            break

    df_dimred["features"] = df_pca.columns
    df_dimred = pd.DataFrame(df_dimred)
    df_dimred = df_dimred.set_index("features", drop=True)

    # Make plot of explained variance
    plt.plot(100 * pca.explained_variance_ratio_[:npcs_to_calc], "-o")
    title = "Cum. variance: (1+2) = {0}%, Total = {1}%".format(
        int(100 * pca.explained_variance_ratio_[:2].sum()),
        int(100 * pca.explained_variance_ratio_[:].sum()),
    )
    plt.xlabel("Component", fontsize=18)
    plt.ylabel("Explained variance (%)", fontsize=18)

    plt.xticks(
        ticks=np.arange(npcs_to_calc),
        labels=np.arange(1, 1 + npcs_to_calc),
        fontsize=14,
    )
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save}.jpg")
    plt.close("all")

    with open(f"{save}.txt", "w") as flog:

        # Save table of feature importance
        for comp in range(npcs_to_show):

            print(
                f"\nExamplined variance by PC{comp+1} = {100*pca.explained_variance_ratio_[comp]:.1f}%",
                file=flog,
            )

            pc_name = pc_names[comp]
            df_sorted = df_dimred.sort_values(
                by=[pc_name.replace("_PC", "_aPC")], ascending=False
            )
            pca_cum_contrib = np.cumsum(
                df_sorted[pc_name.replace("_PC", "_aPC")].values
                / df_sorted[pc_name.replace("_PC", "_aPC")].sum()
            )
            pca_cum_thresh = np.abs(pca_cum_contrib - 0.80).argmin()
            df_sorted = df_sorted.head(n=pca_cum_thresh + 1)

            print(
                df_sorted[
                    [
                        pc_name,
                        pc_name.replace("_PC", "_aPC"),
                        pc_name.replace("_PC", "_cPC"),
                    ]
                ].head(),
                file=flog,
            )
            
    # Adjust the sign of PCs so that larger cells are represent by positive values
    for pcid, pc_name in enumerate(pc_names):
        pearson = np.corrcoef(df.mem_shape_volume_lcc.values, df[pc_name].values)
        if pearson[0, 1] < 0:
            df[pc_name] *= -1
            pca.components_[pcid] *= -1
            
    return df, pc_names, pca


def umap_analysis(df, feature_names, prefix, npcs_to_calc=2):

    # Compute components
    df_umap = df[feature_names]
    matrix_of_features = df_umap.values.copy()
    matrix_of_features_ids = df_umap.index

    reducer = umap.UMAP(n_components=npcs_to_calc)
    embedding = reducer.fit_transform(matrix_of_features)

    umap_names = [f"{prefix}_UMAP{c}" for c in range(1, 1 + embedding.shape[1])]

    df_trans = pd.DataFrame(data=embedding, columns=umap_names)
    df_trans.index = matrix_of_features_ids

    # Add to dataframe
    df = df.merge(df_trans[umap_names], how="outer", left_index=True, right_index=True)

    return df
