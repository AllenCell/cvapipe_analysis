import vtk
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

class ShapeSpaceBasic():
    """
    Basic functionalities of shape space that does
    not require loading the manifest from mshapemode
    step.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    bins = None
    active_axis = None
    
    def __init__(self, config):
        self.config = config
        self.set_path_to_local_staging_folder(config['project']['local_staging'])
        #self.load_shapemode_manifest()

    def set_path_to_local_staging_folder(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self.local_staging = path

    def get_path_to_local_staging_folder(self):
        return self.local_staging
        
    def set_active_axis(self, axis_name):
        self.active_axis = axis_name

    def load_shapemode_manifest(self):
        path_to_shape_space_manifest = self.local_staging/"shapemode/shapemode.csv"
        self.df_results = pd.read_csv(path_to_shape_space_manifest, index_col=0)

    @staticmethod
    def get_number_of_map_points(config):
        return len(config['pca']['map_points'])
        
    @staticmethod
    def iter_shapemodes(config):
        npcs=config['pca']['number_of_pcs']
        prefix = "_".join(config['pca']['aliases'])
        for pc_name in [f"{prefix}_PC{pc}" for pc in range(1, npcs+1)]:
            yield pc_name

    @staticmethod
    def iter_intensities(config):
        for intensity in config['parameterization']['intensities'].keys():
            yield intensity

    @staticmethod
    def iter_aggtype(config):
        for agg in config['aggregation']['type']:
            yield agg

    @staticmethod
    def iter_map_points(config):
        for map_point in config['pca']['map_points']:
            yield map_point

    @staticmethod
    def iter_bins(config):
        for b in range(1,1+len(config['pca']['map_points'])):
            yield b
            
    @staticmethod
    def iter_structures(config):
        for s in config['structures']['desc'].keys():
            yield s

    @staticmethod
    def iter_param_values_as_dataframe(config, iters, no_equals=None):
        df = []
        combinations = []
        for _, it in iters:
            # accepts either a list or an iterator function
            iterator = it if isinstance(it, list) else it(config)
            combinations.append([v for v in iterator])
        for combination in itertools.product(*combinations):
            row = {}
            for (itname, _), value in zip(iters, combination):
                row[itname] = value
            df.append(row)
        df = pd.DataFrame(df)
        if no_equals is not None:
            for itname1, itname2 in no_equals:
                df = df.loc[df[itname1]!=df[itname2]]
        return df.reset_index(drop=True)
        
class ShapeSpace(ShapeSpaceBasic):
    """
    Implements functionalities to navigate the shape
    space.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    active_bin = None
    active_scale = None
    active_structure = None
    
    def __init__(self, config):
        super().__init__(config)
        self.removal_pct = self.config['pca']['removal_pct']
        
    def __repr__(self):
        return f"<{self.__class__.__name__}>original:{self.axes_original.shape}-filtered:{self.axes.shape}"
        
    def set_shape_space_axes(self, df, df_meta):
        self.axes_original = df.copy()
        self.axes = self.remove_extreme_points(self.axes_original, self.removal_pct)
        self.meta = df_meta.loc[self.axes.index, ['structure_name']].copy()

    def load_shape_space_axes(self):
        cols = ['structure_name', 'crop_seg', 'crop_raw']
        path_to_shapemode_manifest = self.local_staging/"shapemode/manifest.csv"
        df_tmp = pd.read_csv(path_to_shapemode_manifest, index_col=0, low_memory=False)
        self.axes_original = df_tmp[[pc for pc in self.iter_shapemodes(self.config)]].copy()
        self.axes = self.remove_extreme_points(self.axes_original, self.removal_pct)
        self.meta = df_tmp.loc[self.axes.index, cols].copy()
            
    def set_active_axis(self, axis_name, digitize):
        if axis_name not in self.axes.columns:
            raise ValueError(f"Axis {axis_name} not found.")
        self.active_axis = axis_name
        if digitize:
            self.digitize_active_axis()
        return

    def set_active_bin(self, b):
        self.active_bin = b

    def deactive_bin(self):
        self.active_bin = None
        
    def set_active_structure(self, structure):
        if isinstance(structure, str):
            structure = [structure]
        self.active_structure=structure

    def deactive_structure(self):
        self.active_structure = None

    def get_active_cellids(self):
        df_tmp = self.meta
        if self.active_bin is not None:
            df_tmp = df_tmp.loc[df_tmp.bin==self.active_bin]
        if self.active_structure is not None:
            df_tmp = df_tmp.loc[df_tmp.structure_name.isin(self.active_structure)]
        return df_tmp.index.values.tolist()
    
    def iter_active_cellids(self):
        for CellId in self.get_active_cellids():
            yield CellId

    def get_active_scale(self):
        return self.active_scale
            
    def digitize_active_axis(self):
        if self.active_axis is None:
            raise ValueError("No active axis.")
            
        values = self.axes[self.active_axis].values.astype(np.float32)
        values -= values.mean()
        self.active_scale = values.std()
        values /= self.active_scale

        LINF = self.config['pca']['map_points'][0]
        LSUP = self.config['pca']['map_points'][-1]
        nbins = len(self.config['pca']['map_points'])
        binw = (LSUP-LINF)/(2*(nbins-1))

        bin_centers = np.linspace(LINF, LSUP, nbins)
        bin_edges = np.unique([(b-binw, b+binw) for b in bin_centers])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.meta['bin'] = np.digitize(values, bin_edges)
        return
    
    def get_index_of_bin(self, b):
        if self.active_axis is None:
            raise ValueError("No active axis.")

        index = self.df_shapemode.loc[
            (self.df_shapemode.shapemode==self.active_axis)&(self.df_shapemode.bin==b)
        ].index
    
        if len(index) == 0:
            raise ValueError(f"No row found for {self.active_axis} and bin {b}.")
    
        if len(index) > 1:
            warnings.warn(f"More than one index found for pc {self.pc_name} and\
            bin {self.map_point}. Something seems wrong with the dataframe of\
            VTK paths generated in the step shapemode. Continuing with\
            first index.")
        return index[0]

    def get_dna_mesh_of_bin(self, b):
        index = self.get_index_of_bin(b)
        return self.read_mesh(self.df_shapemode.at[index, 'dnaMeshPath'])
    
    def get_mem_mesh_of_bin(self, b):
        index = self.get_index_of_bin(b)
        return self.read_mesh(self.df_shapemode.at[index, 'memMeshPath'])

    @staticmethod
    def read_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    @staticmethod
    def remove_extreme_points(axes, pct):
        df_tmp = axes.copy()
        df_tmp["extreme"] = False
        for ax in axes.columns:
            finf, fsup = np.percentile(axes[ax].values, [pct, 100 - pct])
            df_tmp.loc[(df_tmp[ax] < finf), "extreme"] = True
            df_tmp.loc[(df_tmp[ax] > fsup), "extreme"] = True

        df_tmp = df_tmp.loc[df_tmp.extreme == False]
        df_tmp = df_tmp.drop(columns=["extreme"])
        return df_tmp
