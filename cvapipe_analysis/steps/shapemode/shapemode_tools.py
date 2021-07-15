import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from aicsshparam import shtools
from vtk.util.numpy_support import numpy_to_vtk as np2vtk
from vtk.util.numpy_support import vtk_to_numpy as vtk2np

from cvapipe_analysis.tools import io, shapespace, plotting, viz

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
        self.space = shapespace.ShapeSpace(control)
        self.plot_maker_sm = plotting.ShapeModePlotMaker(control)
        self.plot_maker_sp = plotting.ShapeSpacePlotMaker(control)

    def set_data(self, df):
        self.df = df

    def execute(self):
        '''Implements its own execution method bc this step can't
        be framed as a per row calculation.'''
        computed = False
        path_to_output_file = self.get_output_file_name()
        if not path_to_output_file.is_file() or self.control.overwrite():
            try:
                self.workflow()
                computed = True
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                path_to_output_file = None
        self.status(None, path_to_output_file, computed)
        return path_to_output_file
        
    def workflow(self):
        self.space.execute(self.df)
        self.space.save_summary("shapemode/summary.html")
        self.plot_maker_sp.save_feature_importance(self.space)
        self.plot_maker_sp.plot_explained_variance(self.space)
        self.plot_maker_sp.plot_pairwise_correlations(self.space)
        self.plot_maker_sp.execute(display=False)

        self.compute_shcoeffs_for_all_shape_modes()
        self.compute_displacement_vector_relative_to_reference()
        print("Generating 3D meshes. This might take some time...")
        self.recontruct_meshes()
        self.generate_and_save_animated_2d_contours()
        self.plot_maker_sm.combine_and_save_animated_gifs()
        return

    def get_output_file_name(self):
        rel_path = "shapemode/avgshape/combined.tif"
        return self.control.get_staging()/rel_path
        
    def save(self):
        # For consistency.
        return

    def get_coordinates_matrix(self, coords, comp):
        '''Coords has shape (N,). Creates a matrix of shape
        (N,M), where M is the reduced dimension. comp is an
        integer from 1 to npcs.'''
        npts = len(coords)
        matrix = np.zeros((npts, self.space.pca.n_components), dtype=np.float32)
        matrix[:, comp] = coords
        return matrix

    def get_shcoeffs_for_all_map_points(self, shape_mode):
        self.space.set_active_shape_mode(shape_mode, digitize=True)
        mps = self.control.get_map_points()
        coords = [m*self.space.get_active_scale() for m in mps]
        matrix = self.get_coordinates_matrix(coords, int(shape_mode[-1])-1)
        df_inv = self.space.invert(matrix)
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
                self.space.set_active_shape_mode(sm, digitize=True)
                for mpId, df_mp in df_sm.groupby('mpId'):
                    self.space.set_active_map_point_index(mpId)
                    CellIds = self.space.get_active_cellids()
                    if not len(CellIds):
                        raise ValueError(f"No cells found at map point index {mpId}.")
                    for CellId in CellIds:
                        suffixes = [f'position_{u}_centroid_lcc' for u in ['x', 'y', 'z']]
                        ro = [self.df.at[CellId, f'{mov_alias}_{s}'] for s in suffixes]
                        cm = [self.df.at[CellId, f'{ref_alias}_{s}'] for s in suffixes]
                        angle = self.df.at[CellId, f'{ref_alias}_transform_angle_lcc']
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
                    mesh = viz.MeshToolKit.get_mesh_from_series(row, alias, lrec)
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
        abs_path_avgshape = self.control.get_staging()/f"shapemode/avgshape"
        for sm, meshes in tqdm(self.meshes.items(), total=len(self.meshes)):
            projs = viz.MeshToolKit.get_2d_contours(meshes, swap)
            for proj, contours in projs.items():
                fname = f"{abs_path_avgshape}/{sm}_{proj}.gif"
                viz.MeshToolKit.animate_contours(self.control, contours, save=fname)

    @staticmethod
    def translate_mesh_points(mesh, r):
        coords = vtk2np(mesh.GetPoints().GetData())
        coords += np.array(r, dtype=np.float32).reshape(1,3)
        mesh.GetPoints().SetData(np2vtk(coords))
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
