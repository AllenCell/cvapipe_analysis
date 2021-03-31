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

from cvapipe_analysis.tools import io, cluster, shapespace, plotting

class ShapeModeCalculator(io.DataProducer):
    """
    Class for calculating shape modes.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """

    def __init__(self, control):
        super().__init__(control)
        self.stepfolder = "shapemode/avgshape"
        self.plot_maker = plotting.ShapeModePlotMaker(control)

    def execute(self):
        '''Implements its own execution method bc this step can't
        be framed as a per row calculation.'''
        computed = False
        path_to_output_file = self.get_output_file_name()
        if not path_to_output_file.is_file() or self.control.overwrite():
            try:
                self.workflow()
                computed = True
                path_to_output_file = self.save()
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                path_to_output_file = None
        self.status(None, path_to_output_file, computed)
        return path_to_output_file
        
    def workflow(self):
        self.create_shape_space()
        self.calculate_feature_importance()
        self.save_feature_importance()
        self.plot_maker.plot_explained_variance(self.pca)
        self.plot_maker.plot_paired_correlations(self.space.axes)
        self.plot_maker.execute(display=False)

        self.compute_shcoeffs_for_all_shape_modes()
        self.compute_displacement_vector_relative_to_reference()
        print("Generating 3D meshes. This might take some time...")
        self.recontruct_meshes()
        self.generate_and_save_animated_2d_contours()
        self.plot_maker.combine_and_save_animated_gifs()
        return

    def get_output_file_name(self):
        rel_path = f"{self.stepfolder}/combined.tif"
        return self.control.get_staging()/rel_path
        
    def save(self):
        return self.get_output_file_name()
        
    def set_dataframe(self, df):
        self.df = df
        self.features = [
            f for f in self.df.columns if any(
                w in f for w in [
                    f"{alias}_shcoeffs_L"
                    for alias in self.control.get_aliases_for_pca()
                ]
            )
        ]
        
    def calculate_pca(self):
        df_pca = self.df[self.features]
        matrix_of_features = df_pca.values.copy()
        pca = PCA(self.control.get_number_of_shape_modes())
        pca = pca.fit(matrix_of_features)
        matrix_of_features_transform = pca.transform(matrix_of_features)
        self.df_trans = pd.DataFrame(
            data=matrix_of_features_transform,
            columns=[f for f in self.control.iter_shape_modes()])
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
        df_dimred["features"] = self.features
        df_dimred = pd.DataFrame(df_dimred)
        df_dimred = df_dimred.set_index("features", drop=True)
        self.df_dimred = df_dimred
        return

    def save_feature_importance(self):
        path = f"{self.stepfolder}/feature_importance.txt"
        abs_path_txt_file = self.control.get_staging()/path
        print(abs_path_txt_file)
        with open(abs_path_txt_file, "w") as flog:
            for col, sm in enumerate(self.control.iter_shape_modes()):
                exp_var = 100*self.pca.explained_variance_ratio_[col]
                print(f"\nExplained variance {sm}={exp_var:.1f}%", file=flog)
                '''_PC: raw loading, _aPC: absolute loading and
                _cPC: normalized cummulative loading'''
                pc_name = self.df_trans.columns[col]
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
        ranker = self.control.get_alias_for_sorting_shape_modes()
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

    def get_shcoeffs_for_all_map_points(self, shape_mode):
        self.space.set_active_axis(shape_mode, digitize=True)
        mps = self.control.get_map_points()
        coords = [m*self.space.get_active_scale() for m in mps]
        matrix = self.get_coordinates_matrix(coords, int(shape_mode[-1])-1)
        # Inverse PCA here: PCA coords -> shcoeffs
        df_inv = pd.DataFrame(self.pca.inverse_transform(matrix))
        df_inv.columns = self.features
        df_inv['shape_mode'] = shape_mode
        df_inv['mpId'] = np.arange(1, 1+len(mps))
        return df_inv

    def compute_shcoeffs_for_all_shape_modes(self):
        df_coeffs = []
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            df_coeffs = pd.concat(executor.map(
                self.get_shcoeffs_for_all_map_points, self.control.get_shape_modes()
            ), ignore_index=True)
        self.df_coeffs = df_coeffs
        return

    def compute_displacement_vector_relative_to_reference(self):
        '''Aliases with SHE coefficients available can have their position
        adjusted relative to the alias specified as reference for alignment.
        In the case of variance paper we have ref=mem and mov=['dna','str'].'''
        ref_alias = self.control.get_alignment_reference_alias()
        mov_aliases = self.control.get_alignment_moving_aliases()
        for mov_alias in mov_aliases:
            for sm, df_sm in self.df_coeffs.groupby('shape_mode'):
                disp_vector = []
                self.space.set_active_axis(sm, digitize=True)
                for mpId, df_mp in df_sm.groupby('mpId'):
                    self.space.set_active_map_point_index(mpId)
                    CellIds = self.space.get_active_cellids()
                    if not len(CellIds):
                        raise ValueError(f"No cells found at map point index {mpId}.")
                    for CellId in CellIds:
                        suffixes = [f'position_{u}_centroid_lcc' for u in ['x', 'y', 'z']]
                        ro = [self.df.at[CellId, f'{mov_alias}_{s}'] for s in suffixes]
                        cm = [self.df.at[CellId, f'{ref_alias}_{s}'] for s in suffixes]
                        angle = self.df.at[CellId, f'{ref_alias}_shcoeffs_transform_angle_lcc']
                        if np.isnan(angle):
                            '''Angle should be nan if no alignment was applied by
                            cvapipe_analysis. In that case, both ro and cm were
                            calculated in an previously aligned frame of reference
                            (outside cvapipe_analysis). Therefore, not rotation is
                            required.'''
                            disp_vector.append(np.array(ro)-np.array(cm))
                        else:
                            disp_vector.append(self.rotate_vector_relative_to_point(ro, cm, angle))

                    dr_mean = np.array(disp_vector).mean(axis=0).tolist()
                    for du, suffix in zip(dr_mean, ['dx', 'dy', 'dz']):
                        self.df_coeffs.loc[df_mp.index, f'{mov_alias}_{suffix}'] = du
        return

    def recontruct_meshes(self, save_meshes=True):
        self.meshes = {}
        # Reconstruct mesh with twice more detail than original parameterization
        lrec = 2*self.control.get_lmax()
        abs_path_avgshape = self.control.get_staging()/f"shapemode/avgshape"
        for sm, df_sm in self.df_coeffs.groupby("shape_mode"):
            self.meshes[sm] = {}
            for alias in self.control.get_aliases_for_pca():
                self.meshes[sm][alias] = []
                for _, row in df_sm.iterrows():
                    mesh = self.get_mesh_from_series(row, alias, lrec)
                    if f'{alias}_dx' in self.df_coeffs.columns:
                        dr_mean = row[[f'{alias}_d{u}' for u in ['x', 'y', 'z']]]
                        mesh = self.translate_mesh_points(mesh, dr_mean.values)
                    if save_meshes:
                        fname = abs_path_avgshape/f"{alias}_{sm}_{row.mpId}.vtk"
                        shtools.save_polydata(mesh, str(fname))
                    self.meshes[sm][alias].append(mesh)
        return

    def generate_and_save_animated_2d_contours(self):
        swap = self.control.swapxy_on_zproj()
        for sm, meshes in tqdm(self.meshes.items(), total=len(self.meshes)):
            projs = self.plot_maker.get_2d_contours(meshes, swap)
            for proj, contours in projs.items():
                self.plot_maker.animate_contours(contours, f"{sm}_{proj}")

    def create_shape_space(self):
        self.calculate_pca()
        self.sort_shape_modes()
        self.space = shapespace.ShapeSpace(self.control)
        self.space.set_shape_space_axes(self.df_trans, self.df)
        return
    
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
