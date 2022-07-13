import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Optional

from cvapipe_analysis.tools import general

class ShapeModesCalculator(general.DataProducer):
    """
    Class for calculating shape modes.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """
    
    subfolder = 'shapemode/avgshape'
    
    def __init__(self, config):
        super().__init__(config)

    def save(self):
        save_as = self.get_rel_output_file_path_as_str(self.row)
        return save_as
    
    def workflow(self):
        return
    
    @staticmethod
    def get_output_file_name():
        return None

        
class pPCA:
    '''
    Simple class for store PCA objects with persistent feature names.
    '''
    def __init__(
        self,
        pca: PCA,
        features: List
    ):
        self.pca = pca
        self.features = features
    def get_pca(self):
        return self.pca
    def get_feature_names(self):
        return self.features

def pca_analysis(
    df: pd.DataFrame,
    feature_names: List,
    prefix: str,
    npcs_to_calc: int,
    save: Optional[Path] = None
):
    
    """
    Performs principal component analysis on specific columns of
    a pandas dataframe.

    Parameters
    --------------------
    df: pandas df
        Dataframe that contains the columns that should be used
        as input for PCA.
    features_names: list
        List of column names. All columns must be present in df.
    prefix: str
        String to be appended to the column names that represent
        the calculated principal components.
    npcs_to_calc: int
        dimenion of the dimensionality reduced dataset.
    save: Path
        Path to save the results.

    Returns
    -------
        df: pandas df
            Input dataset with new columns for the PCs calcukated
            in by this function
        pc_names: list
            PCs column names 
        pca: pPCA
            PCA objected fitted to the input data. This can be used
            to perform inverse transfrom or transform new data.
    """
        
    # Get feature matrix
    df_pca = df[feature_names]
    matrix_of_features = df_pca.values.copy()
    matrix_of_features_ids = df_pca.index
    
    # Fit and transform the data
    pca = PCA(n_components=npcs_to_calc)
    pca = pca.fit(matrix_of_features)
    matrix_of_features_transform = pca.transform(matrix_of_features)

    pc_names = [f"{prefix}_PC{c}" for c in range(1, 1 + npcs_to_calc)]

    # Dataframe of transformed variable
    df_trans = pd.DataFrame(data=matrix_of_features_transform, columns=pc_names)
    df_trans.index = matrix_of_features_ids

    # Add PCs to the input dataframe
    df = df.merge(df_trans[pc_names], how="outer", left_index=True, right_index=True)
    
    # Analysis of explained variance
    df_dimred = {}
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    for comp, pc_name in enumerate(pc_names):
        load = loading[:, comp]
        pc = [v for v in load]
        apc = [v for v in np.abs(load)]
        total = np.sum(apc)
        cpc = [100 * v / total for v in apc]
        df_dimred[pc_name] = pc
        df_dimred[pc_name.replace("_PC", "_aPC")] = apc
        df_dimred[pc_name.replace("_PC", "_cPC")] = cpc

    # Store results as a dataframe
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

    # Log feature importance along each PC
    with open(f"{save}.txt", "w") as flog:

        for comp in range(npcs_to_calc):

            print(
                f"\nExamplined variance by PC{comp+1} = {100*pca.explained_variance_ratio_[comp]:.1f}%",
                file=flog,
            )

            # Feature importance is reported in 3 ways:
            # _PC - raw loading
            # _aPC - absolute loading
            # _cPC - normalized cummulative loading
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

    # Check wether all features are available
    f = 'mem_shape_volume_lcc'
    if f not in df.columns:
        raise ValueError(f"Column {f} not found in the input dataframe. This\
        column is required to adjust the sign of PCs so that larger cells are\
        represent by positive values")
            
    # Adjust the sign of PCs so that larger cells are represent by positive values
    for pcid, pc_name in enumerate(pc_names):
        pearson = np.corrcoef(df.mem_shape_volume_lcc.values, df[pc_name].values)
        if pearson[0, 1] < 0:
            df[pc_name] *= -1
            pca.components_[pcid] *= -1

    # Creates persistent PCA object
    pca = pPCA(pca=pca, features=feature_names)

    return df, pc_names, pca
