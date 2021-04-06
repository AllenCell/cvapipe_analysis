import vtk
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA

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
    map_point_index = None
    active_shape_mode = None

    def __init__(self, control):
        self.control = control
        
    def set_active_shape_mode(self, sm):
        self.active_shapeMode = sm

    @staticmethod
    def expand(dc):
        '''Works like the function expand from snakemake. Given
        a dict like this:
        {'var1': ['x1', 'x2'], 'var2': ['y1', 'y2', 'y3']},
        it return elements of the permutation of all possible
        combinations of values:
        [{'var1': 'x1', 'var2': 'y1'},
         {'var1': 'x1', 'var2': 'y2'},
         {'var1': 'x1', 'var2': 'y3'},
         {'var1': 'x2', 'var2': 'y1'},
         {'var1': 'x2', 'var2': 'y2'},
         {'var1': 'x2', 'var2': 'y3'}]'''
        keys, values = zip(*dc.items())
        combs = [
            dict(zip(keys, v)) for v in itertools.product(*values)
        ]
        return pd.DataFrame(combs)

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
        
    '''
    #TODO: move these to controller
    @staticmethod
    def iter_intensities(config):
        for intensity in config['parameterization']['intensities'].keys():
            yield intensity

    @staticmethod
    def iter_aggtype(config):
        for agg in config['aggregation']['type']:
            yield agg

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
    '''

class ShapeSpace(ShapeSpaceBasic):
    """
    Implements functionalities to navigate the shape
    space. Process for shape space creation:
    features -> axes -> filter extremes -> shape modes

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    active_scale = None
    active_structure = None

    def __init__(self, control):
        super().__init__(control)

    def execute(self, df):
        self.df = df
        self.features = self.control.get_features_for_pca(df)
        self.workflow()

    def workflow(self):
        self.calculate_pca()
        self.calculate_feature_importance()
        pct = self.control.get_removal_pct()
        self.shape_modes = self.remove_extreme_points(self.axes, pct)
        self.calculate_feature_importance()

    ############
    # AXES
    ############
        
    def calculate_pca(self):
        self.df_pca = self.df[self.features]
        matrix_of_features = self.df_pca.values.copy()
        pca = PCA(self.control.get_number_of_shape_modes())
        pca = pca.fit(matrix_of_features)
        matrix_of_features_transform = pca.transform(matrix_of_features)
        self.axes = pd.DataFrame(
            data=matrix_of_features_transform,
            columns=[f for f in self.control.iter_shape_modes()])
        self.axes.index = self.df_pca.index
        self.pca = pca
        self.sort_pca_axes()
        return

    def sort_pca_axes(self):
        ranker = self.control.get_alias_for_sorting_pca_axes()
        ranker = f"{ranker}_shape_volume"
        for pcid, pc in enumerate(self.axes.columns):
            pearson = np.corrcoef(self.df[ranker].values, self.axes[pc].values)
            if pearson[0, 1] < 0:
                self.axes[pc] *= -1
                self.pca.components_[pcid] *= -1

    def calculate_feature_importance(self):
        df_feats = {}
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        for comp, pc_name in enumerate(self.axes.columns):
            load = loadings[:, comp]
            pc = [v for v in load]
            apc = [v for v in np.abs(load)]
            total = np.sum(apc)
            cpc = [100 * v / total for v in apc]
            df_feats[pc_name] = pc
            df_feats[pc_name.replace("_PC", "_aPC")] = apc
            df_feats[pc_name.replace("_PC", "_cPC")] = cpc
        df_feats["features"] = self.features
        df_feats = pd.DataFrame(df_feats)
        df_feats = df_feats.set_index("features", drop=True)
        self.df_feats = df_feats
        return

    ############
    # SHAPE MODES
    ############
    
    def set_active_shape_mode(self, shape_mode, digitize):
        if shape_mode not in self.shape_modes.columns:
            raise ValueError(f"Shape mode {shape_mode} not found.")
        self.active_shape_mode = shape_mode
        if digitize:
            self.digitize_active_shape_mode()
        return

    def set_active_map_point_index(self, mp):
        nmps = self.control.get_number_of_map_points()
        if (mp<1) or (mp>nmps):
            raise ValueError(f"Map point index must be in the range [1,{nmps}]")
        self.active_map_point_index = mp

    def deactivate_map_point_index(self):
        self.active_map_point_index = None

    def get_active_scale(self):
        return self.active_scale

    def digitize_active_shape_mode(self):
        if self.active_shape_mode is None:
            raise ValueError("No active axis.")
        values = self.shape_modes[self.active_shape_mode].values.astype(np.float32)
        values -= values.mean()
        self.active_scale = values.std()
        values /= self.active_scale
        bin_centers = self.control.get_map_points()
        binw = 0.5*np.diff(bin_centers).mean()
        bin_edges = np.unique([(b-binw, b+binw) for b in bin_centers])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.meta = self.shape_modes.copy()
        self.meta['mpId'] = np.digitize(values, bin_edges)
        return
    
    def get_active_cellids(self):
        df_tmp = self.meta
        if self.active_map_point_index is not None:
            df_tmp = df_tmp.loc[df_tmp.mpId==self.active_map_point_index]
        if self.active_structure is not None:
            df_tmp = df_tmp.loc[df_tmp.structure_name.isin(self.active_structure)]
        return df_tmp.index.values.tolist()

    def browse(self, constraints):
        '''Limits the shape space in a specific region and
        returns the cells ids in that regions (shape mode and
        map point index). Can also constrain by structure
        name and then only cell ids of the same structure will
        be returned.'''
        if not "shape_mode" in constraints:
            raise ValueError("Shape mode not in constraints.")
        val = constraints['shape_mode']
        if val != self.active_shape_mode:
            self.set_active_shape_mode(val, True)
        df = self.meta
        if "mpId" in constraints:
            val = constraints['mpId']
            df = df.loc[df.mpId==val]
            if not len(df):
                print(f"No cells found at mpId: {val}")
                return []
        if "structure" in constraints:
            val = constraints["structure"]
            sid = self.df.loc[self.df.structure_name==val].index
            df = df.loc[df.index.isin(sid)]
            if not len(df):
                print(f"No {val} cells found at mpId.")
                return []
        return df.index.values.tolist()

    def get_aggregated_df(self, variables, include_cellIds=True):
        df = self.expand(variables)
        if include_cellIds:
            df["CellIds"] = 0
            df.CellIds = df.CellIds.astype(object)
            for index, row in df.iterrows():
                df.at[index, "CellIds"] = self.browse(row)
        df.mpId = df.mpId.astype(np.int64)
        return df

    def save_summary(self, df, path):
        filters = dict((k, df[k].unique()[0]) for k in ["aggtype", "alias"])
        for k, v in filters.items():
            df = df.loc[df[k]==v]     
        for index, row in df.iterrows():
            df.at[index, "ncells"] = len(row.CellIds)
        df.ncells = df.ncells.astype(int)
        df.mpId -= self.control.get_center_map_point_index()
        df = df.drop(columns=[k for k in filters.keys()]+["CellIds"])
        df = df.set_index(["shape_mode", "structure", "mpId"])
        df = df.unstack(level=-1)
        df.to_html(self.control.get_staging()/path)
        return df
                