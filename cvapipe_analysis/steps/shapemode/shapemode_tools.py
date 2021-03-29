import vtk
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Union
from vtk.util.numpy_support import numpy_to_vtk as np2vtk
from vtk.util.numpy_support import vtk_to_numpy as vtk2np

from cvapipe_analysis.tools import general, cluster, shapespace, plotting

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
            if value['alias'] in self.config['features']['SHE']['aliases']:
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

    def compute_shcoeffs_for_all_map_point_shapes(self):
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            df_coeffs = pd.concat(executor.map(
                self.get_shcoeffs_for_map_point_shapes, [s for s in self.space.iter_shapemodes(self.config)]
            ), ignore_index=True)
        self.df_coeffs = df_coeffs
        return

    def get_reference_and_moving_aliases(self):
        ref_obj = self.config['features']['align']['reference']
        ref_alias = self.config['data']['segmentation'][ref_obj]['alias']
        mov_aliases = [a for a in self.aliases_with_coeffs_available if a!= ref_alias]
        return ref_alias, mov_aliases
    
    def compute_displacement_vector_relative_to_reference(self):
        '''Objects with SHE coefficients available can have their position
        adjusted relative to the object specified as reference for alignment.'''
        ref_alias, mov_aliases = self.get_reference_and_moving_aliases()
        for mov_alias in mov_aliases:
            for shapemode, df_shapemode in self.df_coeffs.groupby('shapemode'):
                disp_vector = []
                self.space.set_active_axis(shapemode, digitize=True)
                for b, df_bin in df_shapemode.groupby('bin'):
                    self.space.set_active_bin(b)
                    for CellId in self.space.iter_active_cellids():
                        suffixes = [f'position_{u}_centroid_lcc' for u in ['x', 'y', 'z']]
                        ro = [self.df.at[CellId, f'{mov_alias}_{s}'] for s in suffixes]
                        cm = [self.df.at[CellId, f'{ref_alias}_{s}'] for s in suffixes]
                        angle = self.df.at[CellId, f'{ref_alias}_shcoeffs_transform_angle_lcc']
                        if np.isnan(angle):
                            '''Angle should be nan if no alignment was applied by cvapipe_analysis.
                            In that case, both ro and cm were calculated in an previously aligned
                            frame of reference. Therefore, not rotation is required.'''
                            disp_vector.append(np.array(ro)-np.array(cm))
                        else:
                            disp_vector.append(self.rotate_vector_relative_to_point(ro, cm, angle))

                    dr_mean = np.array(disp_vector).mean(axis=0).tolist()
                    for du, suffix in zip(dr_mean, ['dx', 'dy', 'dz']):
                        self.df_coeffs.loc[df_bin.index, f'{mov_alias}_{suffix}'] = du
        return

    def recontruct_meshes(self, save_meshes=True):
        self.meshes = {}
        # Reconstruct mesh with twice more detail than original parameterization
        lrec = 2*self.config['features']['SHE']['lmax']
        abs_path_avgshape = self.abs_path_local_staging/f"shapemode/avgshape"
        for shapemode, df_sm in self.df_coeffs.groupby('shapemode'):
            self.meshes[shapemode] = {}
            for alias in self.aliases:
                self.meshes[shapemode][alias] = []
                for _, row in df_sm.iterrows():
                    mesh = self.get_mesh_from_series(row, alias, lrec)
                    if f'{alias}_dx' in self.df_coeffs.columns:
                        dr_mean = row[[f'{alias}_d{u}' for u in ['x', 'y', 'z']]]
                        mesh = self.translate_mesh_points(mesh, dr_mean.values)
                    if save_meshes:
                        fname = abs_path_avgshape/f"{alias}_{shapemode}_{row.bin}.vtk"
                        shtools.save_polydata(mesh, str(fname))
                    self.meshes[shapemode][alias].append(mesh)
        return
    
    def generate_and_save_animated_2d_contours(self):
        swapxy_on_zproj = self.config['pca']['plot']['swapxy_on_zproj']
        for shapemode, meshes in tqdm(self.meshes.items(), total=len(self.meshes)):
            projections = self.plot_maker.get_2d_contours(meshes, swapxy_on_zproj)
            for proj, contours in projections.items():
                self.plot_maker.animate_contours(contours, f"{shapemode}_{proj}")

    def combine_animated_gifs(self):
        shapemodes = [mode for mode in self.space.iter_shapemodes(self.config)]
        self.plot_maker.combine_and_save_animated_gifs(shapemodes)

    def create_shape_space(self, df):
        self.set_dataframe(df)
        self.calculate_pca()
        self.sort_shape_modes()        
        self.space = shapespace.ShapeSpace(self.config)
        self.space.set_shape_space_axes(self.df_trans, self.df)
        return
    
    def workflow(self, df):
        self.create_shape_space(df)
        self.calculate_feature_importance()
        self.save_feature_importance()
        self.plot_maker.plot_explained_variance(self.pca)
        self.plot_maker.execute(display=False)

        self.compute_shcoeffs_for_all_map_point_shapes()
        self.compute_displacement_vector_relative_to_reference()
        print("Generating 3D meshes. This might take some time...")
        self.recontruct_meshes()
        self.generate_and_save_animated_2d_contours()
        self.combine_animated_gifs()
        return
    
    @staticmethod
    def get_output_file_name():
        return None

    @staticmethod
    def translate_mesh_points(mesh, r):
        coords = vtk2np(mesh.GetPoints().GetData())
        coords += np.array(r, dtype=np.float32).reshape(1,3)
        mesh.GetPoints().SetData(np2vtk(coords))
        return mesh
    
    @staticmethod
    def get_mesh_from_series(row, alias, lmax):
        coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
        for l in range(lmax):
            for m in range(l + 1):
                try:
                    # Cosine SHE coefficients
                    coeffs[0, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                    ]
                    # Sine SHE coefficients
                    coeffs[1, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                    ]
                # If a given (l,m) pair is not found, it is assumed to be zero
                except: pass
        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
        return mesh
    
    @staticmethod
    def rotate_vector_relative_to_point(vector, point, angle):
        angle = np.pi*angle/180.0
        rot_mx = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]])
        u = np.array(vector)-np.array(point)
        return np.matmul(rot_mx, u)
