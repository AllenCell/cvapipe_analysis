import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Union

from cvapipe_analysis.tools import general, cluster, shapespace, plotting, viz

class ShapeModeCalculator(general.DataProducer):
    """
    Class for calculating shape modes.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """
    
    subfolder = 'shapemode/pca'
    
    def __init__(self, config):
        super().__init__(config)
        self.plot_maker = plotting.ShapeModePlotMaker(config)
        
    def save(self):
        save_as = self.get_rel_output_file_path_as_str(self.row)
        return save_as

    def set_dataframe(self, df):
        self.df = df
        self.aliases_with_coeffs_available = []
        for obj, value in self.config['data']['segmentation'].items():
            if isinstance(value, dict):
                self.aliases_with_coeffs_available.append(value['alias'])
        # Aliases used for PCA is a subset of aliases_with_coeffs_available
        self.aliases = self.config['pca']['aliases']
        self.prefix = "_".join(self.aliases)
        self.features = [
            f for f in self.df.columns if any(
                w in f for w in [f"{alias}_shcoeffs_L" for alias in self.aliases]
            )
        ]
    
    def calculate_pca(self):
        df_pca = self.df[self.features]
        matrix_of_features = df_pca.values.copy()
        self.npcs = self.config['pca']['number_of_pcs']
        pca = PCA(self.npcs)
        pca = pca.fit(matrix_of_features)
        matrix_of_features_transform = pca.transform(matrix_of_features)
        pc_names = [f"{self.prefix}_PC{c}" for c in range(1, 1+self.npcs)]
        self.df_trans = pd.DataFrame(data=matrix_of_features_transform, columns=pc_names)
        self.df_trans.index = df_pca.index
        self.pca = pca
        return

    def calculate_feature_importance(self):
        df_dimred = {}
        loading = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        for comp, pc_name in enumerate(self.df_trans.columns):
            load = loading[:, comp]
            pc = [v for v in load]
            apc = [v for v in np.abs(load)]
            total = np.sum(apc)
            cpc = [100 * v / total for v in apc]
            df_dimred[pc_name] = pc
            df_dimred[pc_name.replace("_PC", "_aPC")] = apc
            df_dimred[pc_name.replace("_PC", "_cPC")] = cpc
        # Store results as a dataframe
        df_dimred["features"] = self.features
        df_dimred = pd.DataFrame(df_dimred)
        df_dimred = df_dimred.set_index("features", drop=True)
        self.df_dimred = df_dimred
        return

    def save_feature_importance(self):
        abs_path_txt_file = self.abs_path_local_staging/f"{self.subfolder}/feature_importance.txt"
        with open(abs_path_txt_file, "w") as flog:
            for pc in range(self.npcs):
                print(f"\nExplained variance PC{pc+1}={100*self.pca.explained_variance_ratio_[pc]:.1f}%", file=flog)
                # _PC - raw loading
                # _aPC - absolute loading
                # _cPC - normalized cummulative loading
                pc_name = self.df_trans.columns[pc]
                df_sorted = self.df_dimred.sort_values(
                    by=[pc_name.replace("_PC", "_aPC")], ascending=False
                )
                pca_cum_contrib = np.cumsum(
                    df_sorted[pc_name.replace("_PC", "_aPC")].values/
                    df_sorted[pc_name.replace("_PC", "_aPC")].sum()
                )
                pca_cum_thresh = np.abs(pca_cum_contrib-0.80).argmin()
                df_sorted = df_sorted.head(n=pca_cum_thresh+1)
                print(df_sorted[[
                    pc_name,
                    pc_name.replace("_PC", "_aPC"),
                    pc_name.replace("_PC", "_cPC"),]].head(), file=flog
                )
        return

    def sort_shape_modes(self):
        ranker = f"{self.config['pca']['sorter']}_shape_volume"
        ranker = [f for f in self.df.columns if ranker in f][0]
        for pcid, pc in enumerate(self.df_trans.columns):
            pearson = np.corrcoef(self.df[ranker].values, self.df_trans[pc].values)
            if pearson[0, 1] < 0:
                self.df_trans[pc] *= -1
                self.pca.components_[pcid] *= -1

    def get_coordinates_matrix(self, coords, comp):
        '''Coords has shape (N,). Creates a matrix of shape
        (N,M), where M is the reduced dimension. comp is an
        integer from 1 to npcs.'''
        npts = len(coords)
        matrix = np.zeros((npts, self.pca.n_components), dtype=np.float32)
        matrix[:, comp] = coords
        return matrix 

    def get_shcoeffs_for_map_point_shapes(self, shapemode):
        self.space.set_active_axis(shapemode, digitize=True)
        map_points = self.config['pca']['map_points']
        coords = [m*self.space.get_active_scale() for m in map_points]
        matrix = self.get_coordinates_matrix(coords, int(shapemode[-1])-1)
        # Uses inver PCA here
        df_inv = pd.DataFrame(self.pca.inverse_transform(matrix))
        df_inv.columns = self.features
        df_inv['shapemode'] = shapemode
        df_inv['bin'] = np.arange(1, 1+len(map_points))
        return df_inv

    def get_shcoeffs_for_all_map_point_shapes(self):
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            df_coeffs = pd.concat(executor.map(
                self.get_shcoeffs_for_map_point_shapes, [s for s in self.space.iter_shapemodes(self.config)]
            ))
        return df_coeffs

    def recontruct_meshes(self):
        self.meshes = {}
        nrec = len(self.df_coeffs)
        lrec = 2*self.config['features']['SHE']['lmax']
        for alias in tqdm(self.aliases_with_coeffs_available):
                self.meshes[alias] = []
                for index, row in tqdm(self.df_coeffs.iterrows(), total=nrec):
                    mesh = self.get_mesh_from_series(row, alias, lrec)
                    self.meshes[alias].append(mesh)
        return
        
    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def <<<<<<<< reconstruct the meshes here and correct nuclear location.
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            df, bin_indexes = fix_nuclear_position

            def process_this_index(index_row):

                #Change the coordinate system of nuclear centroid
                #from nuclear to the aligned cell.

                index, row = index_row

                dxc, dyc, dzc = transform_coords_to_mem_space(
                    xo = row["dna_position_x_centroid_lcc"],
                    yo = row["dna_position_y_centroid_lcc"],
                    zo = row["dna_position_z_centroid_lcc"],
                    # Cell alignment angle
                    angle = row["mem_shcoeffs_transform_angle_lcc"],
                    # Cell centroid
                    cm = [row[f"mem_position_{k}_centroid_lcc"] for k in ["x", "y", "z"]],
                )

                return (dxc, dyc, dzc)

            # Change the reference system of the vector that
            # defines the nuclear location relative to the cell
            # of all cells that fall into the same bin.
            for (b, indexes) in bin_indexes:
                # Subset with cells from the same bin.
                df_tmp = df.loc[df.index.isin(indexes)]            
                # Change reference system for all cells in parallel.
                nuclei_cm_fix = []
                with DistributedHandler(distributed_executor_address) as handler:
                    future = handler.batched_map(
                        process_this_index,
                        [index_row for index_row in df_tmp.iterrows()],
                    )
                    nuclei_cm_fix.append(future)
                # Average changed nuclear centroid over all cells
                mean_nuclei_cm_fix = np.array(nuclei_cm_fix[0]).mean(axis=0)
                # Store
                df_agg.loc[b, "dna_dxc"] = mean_nuclei_cm_fix[0]
                df_agg.loc[b, "dna_dyc"] = mean_nuclei_cm_fix[1]
                df_agg.loc[b, "dna_dzc"] = mean_nuclei_cm_fix[2]

    
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    '''
    def workflow(self, df):
        
        self.set_dataframe(df)
        self.calculate_pca()
        self.sort_shape_modes()
        self.calculate_feature_importance()
        self.save_feature_importance()
        
        self.plot_maker.plot_explained_variance(self.pca)
        self.plot_maker.execute(display=False)
        
        self.space = shapespace.ShapeSpace(self.config)
        self.space.set_shape_space_axes(self.df_trans, self.df)
        self.df_coeffs = self.get_shcoeffs_for_all_map_point_shapes()

        self.recontruct_meshes()
        '''
        animator = viz.Animator(self.config)
        for shapemode in self.space.iter_shapemodes(self.config):        
            df_paths = animator.animate_shape_modes_and_save_meshes(self)        
            import pdb; pdb.set_trace()
        return
        '''

    
    @staticmethod
    def get_output_file_name():
        return None

    @staticmethod
    def get_mesh_from_series(row, alias, lmax):
        coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
        for l in range(lmax):
            for m in range(l + 1):
                try:
                    # Cosine SHE coefficients
                    coeffs[0, l, m] = row[[f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]]
                    # Sine SHE coefficients
                    coeffs[1, l, m] = row[[f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]]
                # If a given (l,m) pair is not found, it is assumed to be zero
                except:
                    pass
        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
        return mesh
