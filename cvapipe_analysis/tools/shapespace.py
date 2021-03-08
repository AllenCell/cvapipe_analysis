import numpy as np
import pandas as pd

class ShapeSpace:

    active_axis=None
    map_points=[-2,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]

    def __init__(self, axes, removal_pct=1):

        self.axes_original = axes.copy()
        self.axes = self.remove_extreme_points(axes, removal_pct)

    def __repr__(self):
        return f"<{self.__class__.__name__}>{self.axes.shape}"

    def remove_extreme_points(self, axes, removal_pct):
        return axes

    def set_activate_axis(self, axis_name):
        if axis_name not in self.axes.columns:
            raise ValueError(f"Axis {axis_name} not found.")
        self.active_axis = axis_name

    def get_activate_axis(self):
        return self.active_axis
    
    def set_map_points(map_points):
        self.map_points = map_points
    
    def get_number_of_map_points():
        return len(self.map_points)
    
    def link_results_folder(self, path):
        path_to_shape_space_manifest = path / 'shapemode.csv'
        if path_to_shape_space_manifest.is_file():
            self.df_results = pd.read_csv(path_to_shape_space_manifest, index_col=0)
        else:
            raise ValueError(f"File shapemode.csv not found in the folder.")


