import vtk
import numpy as np
import pandas as pd
from pathlib import Path

class ShapeSpace:

    bins = None
    active_axis=None
    map_points=[-2,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]
    
    def __init__(self, axes, config, removal_pct=1):

        if axes is not None:
            self.axes_original = axes.copy()
            self.axes = self.remove_extreme_points(axes, removal_pct)
        self.set_path_to_local_staging_folder(config['project']['local_staging'])
        self.load_shapemode_manifest()

    def __repr__(self):
        return f"<{self.__class__.__name__}>original:{self.axes_original.shape}{self.axes.shape}"

    def set_path_to_local_staging_folder(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self.local_staging = path
        
    def get_path_to_local_staging_folder(self):
        return self.local_staging
    
    def set_active_axis(self, axis_name):
        if axis_name not in self.axes.columns:
            raise ValueError(f"Axis {axis_name} not found.")
        self.active_axis = axis_name

    def get_active_axis(self):
        return self.active_axis
    
    def set_map_points(map_points):
        self.map_points = map_points
    
    def get_number_of_map_points():
        return len(self.map_points)
    
    def load_shapemode_manifest(self):
        path_to_shape_space_manifest = self.local_staging/"shapemode/shapemode.csv"
        if path_to_shape_space_manifest.is_file():
            self.df_results = pd.read_csv(path_to_shape_space_manifest, index_col=0)
        else:
            raise ValueError("File shapemode.csv not found.")
        print(f"Dataframe loaded: {self.df_results.shape}")
        
    def iter_map_points(self):
        for b, map_point in enumerate(self.map_points):
            yield b+1, map_point
    
    def get_indexes_in_bin(self, b):
        if self.bins is None:
            raise ValueError("No digitized axis found.")
        return self.bins.loc[self.bins.bin==b].index

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

    def digitize_active_axis(self):

        if self.active_axis is None:
            raise ValueError("No active axis.")
        
        values = self.axes[self.get_active_axis()].values.astype(np.float32)
        values -= values.mean()
        pc_std = values.std()
        values /= pc_std

        LINF = -2.0 # inferior limit = -2 std
        LSUP = 2.0 # superior limit = 2 std
        nbins = len(self.map_points)
        binw = (LSUP-LINF)/(2*(nbins-1))

        bin_centers = np.linspace(LINF, LSUP, nbins)
        bin_edges = np.unique([(b-binw, b+binw) for b in bin_centers])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Aplly digitization
        bins = self.axes.copy()
        bins['bin'] = np.digitize(values, bin_edges)
        self.bins = bins[['bin']]
        
        return

    @staticmethod
    def read_mesh(path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    def get_index_of_bin(self, b):
        if self.active_axis is None:
            raise ValueError("No active axis.")

        index = self.df_results.loc[
            (self.df_results.shapemode==self.active_axis)&(self.df_results.bin==b)
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
        return self.read_mesh(self.df_results.at[index, 'dnaMeshPath'])
    
    def get_mem_mesh_of_bin(self, b):
        index = self.get_index_of_bin(b)
        return self.read_mesh(self.df_results.at[index, 'memMeshPath'])

class ShapeSpaceBasic(ShapeSpace):
    def __init__(self, config):
        super().__init__(None, config)
    def set_active_axis(self, axis_name):
        self.active_axis = axis_name
