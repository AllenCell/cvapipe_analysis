import os
import vtk
import math
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import cluster as spcluster
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
from vtk.util.numpy_support import numpy_to_vtk as np2vtk

from cvapipe_analysis.tools import general

class PlotMaker(general.LocalStagingWriter):
    """
    Support class. Should not be instantiated directly.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    figs = []
    df = None
    dpi = 72
    
    def __init__(self, config):
        super().__init__(config)
        
    def set_dataframe(self, df, dropna=[]):
        self.df_original = df.copy()
        if dropna:
            self.df_original = self.df_original.dropna(subset=dropna)
        self.df = self.df_original
        
    def check_dataframe_exists(self):
        if self.df is None:
            raise ValueError("Please set a dataframe first.")
        return True

    def filter_dataframe(self, filters):
        self.check_dataframe_exists()
        self.df = self.get_filtered_dataframe(self.df_original, filters)

    def execute(self, display=True, **kwargs):
        if 'dpi' in kwargs: self.dpi = kwargs['dpi']
        prefix = None if 'prefix' not in kwargs else kwargs['prefix']
        self.workflow()
        self.save(display, prefix)
        self.figs=[]
            
    def save(self, display=True, prefix=None):
        for (fig, signature) in self.figs:
            if display:
                fig.show()
            else:
                fname = signature
                if prefix is not None:
                    fname = f"{prefix}_{signature}"
                print(fname)
                fig.savefig(self.abs_path_local_staging/f"{self.subfolder}/{fname}.png")
                plt.close(fig)

    @staticmethod
    def get_filtered_dataframe(df, filters):
        for k, v in filters.items():
            values = v if isinstance(v, list) else [v]
            df = df.loc[df[k].isin(values)]
        return df
                                
    @staticmethod
    def get_dataframe_desc(df):
        desc = "-".join([str(sorted(df[f].unique())) for f in ['intensity','shapemode','bin']])
        desc = desc.replace("[","").replace("]","").replace("'","").replace(", ",":")
        return desc
                
class StereotypyPlotMaker(PlotMaker):
    """
    Class for ranking and plotting structures according
    to their stereotypy value.
    
    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    subfolder = 'stereotypy/plots'
    max_number_of_pairs = 0

    def __init__(self, config):
        super().__init__(config)
        self.structures = config['structures']['desc']

    def set_max_number_of_pairs(self, n):
        self.max_number_of_pairs = n if n > 0 else 0
    
    def make_boxplot(self):
        self.check_dataframe_exists()
        labels = []
        fig, ax = plt.subplots(1,1, figsize=(7,8), dpi=self.dpi)
        for sid, sname in enumerate(reversed(self.structures.keys())):
            df_s = self.df.loc[self.df.structure_name==sname]
            if self.max_number_of_pairs > 0:
                df_s = df_s.sample(n=np.min([len(df_s), self.max_number_of_pairs]))
            npairs = len(df_s)
            y = np.random.normal(size=npairs, loc=sid, scale=0.1)
            ax.scatter(df_s.Pearson, y, s=1, c='k', alpha=0.1)
            box = ax.boxplot(
                df_s.Pearson,
                positions=[sid],
                showmeans=True,
                widths=0.75,
                sym='',
                vert=False,
                patch_artist=True,
                meanprops={
                    "marker": "s",
                    "markerfacecolor": "black",
                    "markeredgecolor": "white",
                    "markersize": 5
                }
            )
            label = f"{self.structures[sname][0]} (N={npairs:04d})"
            labels.append(label)
            box['boxes'][0].set(facecolor=self.structures[sname][1])
            box['medians'][0].set(color='black')
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.2,1.0)
        ax.set_xlabel("Pearson correlation coefficient", fontsize=14)
        ax.set_title(self.get_dataframe_desc(self.df))
        ax.grid(True)
        plt.tight_layout()
        self.figs.append((fig, self.get_dataframe_desc(self.df)))
        return

    def workflow(self):
        self.make_boxplot()
    
class ConcordancePlotMaker(PlotMaker):
    """
    Class for creating concordance heatmaps.
    
    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    subfolder = 'concordance/plots'

    def __init__(self, config):
        super().__init__(config)
        self.structures = config['structures']['desc']
    
    def get_dataframe_of_center_bin(self, intensity, shapemode):
        center_bin = int(0.5*(len(self.config['pca']['map_points'])+1))
        return self.get_filtered_dataframe(
            self.df_original,
            {'intensity':intensity, 'shapemode':shapemode, 'bin': [center_bin]}
        )
    
    @staticmethod
    def get_correlation_matrix(df, rank):
        df_corr = df.groupby(['structure_name1', 'structure_name2']).agg(['mean']).reset_index()
        df_corr = df_corr.pivot(index='structure_name1', columns='structure_name2', values=('Pearson', 'mean'))
        df_corr = df_corr[rank]
        df_corr = df_corr.loc[rank]
        matrix = df_corr.values
        np.fill_diagonal(matrix, 1.0)
        return matrix
        
    def build_correlation_matrix(self):
        bins = sorted(self.df.bin.unique())
        if len(bins) > 2:
            raise ValueError(f"More than 2 bin values found: {bins}")
        matrices = []
        for b in bins:
            df_bin = self.df.loc[self.df.bin==b]
            matrices.append(self.get_correlation_matrix(df_bin, self.structures))
        self.matrix = matrices[0]
        if len(bins) > 1:
            matrix_inf = np.tril(matrices[0], -1)
            matrix_sup = np.triu(matrices[1],  1)
            self.matrix = matrix_inf + matrix_sup
        return

    def make_relative_heatmap(self):
        df = self.get_dataframe_of_center_bin(self.intensity, self.shapemode)
        corr = self.get_correlation_matrix(df, self.structures)
        self.matrix = corr - self.matrix
        np.fill_diagonal(self.matrix, 0)
        self.make_heatmap(diagonal=True, cmap='PRGn', prefix='heatmap_relative')
        return
    
    def make_heatmap(self, diagonal=False, cmap='RdBu', **kwargs):
        ns = self.matrix.shape[0]
        fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=self.dpi)
        ax.imshow(-self.matrix, cmap=cmap, vmin=-1.0,vmax=1.0)
        ax.set_xticks(np.arange(self.matrix.shape[0]))
        ax.set_yticks(np.arange(self.matrix.shape[0]))
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([v[0] for k,v in self.structures.items()])
        for edge, spine in ax.spines.items():
                spine.set_visible(False)
        ax.set_xticks(np.arange(ns+1)-.5, minor=True)
        ax.set_yticks(np.arange(ns+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False)
        if diagonal:
            ax.plot([-0.5, ns-0.5], [-0.5, ns-0.5], 'k-', linewidth=2)
        ax.set_title(self.get_dataframe_desc(self.df))
        prefix = 'heatmap' if 'prefix' not in kwargs else kwargs['prefix']
        self.figs.append((fig, f'{prefix}_'+self.get_dataframe_desc(self.df)))
        return
    
    def make_dendrogram(self):
        Z = spcluster.hierarchy.linkage(self.matrix, 'average')
        Z = spcluster.hierarchy.optimal_leaf_ordering(Z, self.matrix)
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        dn = spcluster.hierarchy.dendrogram(
            Z,
            labels = [v[0] for k, v in self.structures.items()],
            leaf_rotation = 90
        )
        ax.set_title(self.get_dataframe_desc(self.df))
        self.figs.append((fig, 'dendrogram_'+self.get_dataframe_desc(self.df)))
        return

    def check_and_store_parameters(self):
        bins = self.df.bin.unique()
        if len(bins) > 2:
            raise ValueError(f"More than 2 bins found in the dataframe: {bins}")
        intensities = self.df.intensity.unique()
        if len(intensities) > 1:
            raise ValueError(f"Multiples intensities found in the dataframe: {intensities}")
        shapemodes = self.df.shapemode.unique()
        if len(shapemodes) > 1:
            raise ValueError(f"Multiples shapemodes found in the dataframe: {shapemodes}")
        self.intensity = intensities[0]
        self.shapemode = shapemodes[0]
        self.bins = bins
        self.multiple_bins=len(bins)>1
        return
    
    def workflow(self):
        self.check_and_store_parameters()
        self.build_correlation_matrix()
        self.make_heatmap(diagonal=self.multiple_bins)
        if self.multiple_bins:
            self.make_relative_heatmap()
        else:
            self.make_dendrogram()
        return
    
class ShapeModePlotMaker(PlotMaker):
    """
    Class for creating plots for shape mode step.
    
    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    subfolder = 'shapemode/avgshape'

    def __init__(self, config):
        super().__init__(config)
        
    def plot_explained_variance(self, pca):
        npcs_to_calc = self.config['pca']['number_of_pcs']
        fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=self.dpi)
        ax.plot(100 * pca.explained_variance_ratio_[:npcs_to_calc], "-o")
        title = "Cum. variance: (1+2) = {0}%, Total = {1}%".format(
            int(100 * pca.explained_variance_ratio_[:2].sum()),
            int(100 * pca.explained_variance_ratio_[:].sum()),
        )
        ax.set_xlabel("Component", fontsize=18)
        ax.set_ylabel("Explained variance (%)", fontsize=18)
        ax.set_xticks(np.arange(npcs_to_calc))
        ax.set_xticklabels(np.arange(1, 1 + npcs_to_calc))
        ax.set_title(title, fontsize=18)
        plt.tight_layout()
        self.figs.append((fig, 'explained_variance'))
        return
        
    def animate_contours(self, contours, prefix):
        nbins = len(self.config['pca']['map_points'])
        hmin, hmax, vmin, vmax = self.config['pca']['plot']['limits']
        offset = 0.05*(hmax-hmin)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.tight_layout()
        plt.close()
        ax.set_xlim(hmin-offset, hmax+offset)
        ax.set_ylim(vmin-offset, vmax+offset)
        ax.set_aspect("equal")

        lines = []
        for alias, _ in contours.items():
            for obj, value in self.config["data"]["segmentation"].items():
                if value['alias'] == alias:
                    break
            (line,) = ax.plot([], [], lw=2, color=value['color'])
            lines.append(line)

        def animate(i):
            for alias, line in zip(contours.keys(), lines):
                ct = contours[alias][i]
                mx = ct[:, 0]
                my = ct[:, 1]
                line.set_data(mx, my)
            return lines

        anim = animation.FuncAnimation(
            fig, animate, frames=nbins, interval=100, blit=True
        )
        fname = self.abs_path_local_staging/f"{self.subfolder}/{prefix}.gif"
        anim.save(fname, writer="imagemagick", fps=nbins)
        plt.close("all")
        return

    @staticmethod
    def find_plane_mesh_intersection(mesh, proj):

        # Find axis orthogonal to the projection of interest
        axis = [a for a in [0, 1, 2] if a not in proj][0]
        
        # Get all mesh points
        points = vtk2np(mesh.GetPoints().GetData())

        if not np.abs(points[:, axis]).sum():
            raise Exception("Only zeros found in the plane axis.")

        mid = np.mean(points[:, axis])

        # Set the plane a little off center to avoid undefined intersections
        # Without this the code hangs when the mesh has any edge aligned with the
        # projection plane
        mid += 0.75
        offset = 0.1 * np.ptp(points, axis=0).max()

        # Create a vtkPlaneSource
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(4)
        plane.SetYResolution(4)
        if axis == 0:
            plane.SetOrigin(mid, points[:, 1].min() - offset, points[:, 2].min() - offset)
            plane.SetPoint1(mid, points[:, 1].min() - offset, points[:, 2].max() + offset)
            plane.SetPoint2(mid, points[:, 1].max() + offset, points[:, 2].min() - offset)
        if axis == 1:
            plane.SetOrigin(points[:, 0].min() - offset, mid, points[:, 2].min() - offset)
            plane.SetPoint1(points[:, 0].min() - offset, mid, points[:, 2].max() + offset)
            plane.SetPoint2(points[:, 0].max() + offset, mid, points[:, 2].min() - offset)
        if axis == 2:
            plane.SetOrigin(points[:, 0].min() - offset, points[:, 1].min() - offset, mid)
            plane.SetPoint1(points[:, 0].min() - offset, points[:, 1].max() + offset, mid)
            plane.SetPoint2(points[:, 0].max() + offset, points[:, 1].min() - offset, mid)
        plane.Update()
        plane = plane.GetOutput()

        # Trangulate the plane
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(plane)
        triangulate.Update()
        plane = triangulate.GetOutput()

        # Calculate intersection
        intersection = vtk.vtkIntersectionPolyDataFilter()
        intersection.SetInputData(0, mesh)
        intersection.SetInputData(1, plane)
        intersection.Update()
        intersection = intersection.GetOutput()

        # Get coordinates of intersecting points
        points = vtk2np(intersection.GetPoints().GetData())

        # Sorting points clockwise
        # This has been discussed here:
        # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
        # but seems not to be very efficient. Better version is proposed here:
        # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
        coords = points[:, proj]
        center = tuple(
            map(
                operator.truediv,
                reduce(lambda x, y: map(operator.add, x, y), coords),
                [len(coords)] * 2,
            )
        )
        coords = sorted(
            coords,
            key=lambda coord: (
                -135
                -math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))
            )
            % 360,
        )

        # Store sorted coordinates
        #points[:, proj] = coords
        return np.array(coords)

    @staticmethod
    def get_2d_contours(named_meshes, swapxy_on_zproj=False):
        contours = {}
        projs = [[0, 1], [0, 2], [1, 2]]
        if swapxy_on_zproj:
            projs = [[0, 1], [1, 2], [0, 2]]
        for dim, proj in zip(['z', 'y', 'x'], projs):
            contours[dim] = {}
            for alias, meshes in named_meshes.items():
                contours[dim][alias] = []
                for mesh in meshes:
                    coords = ShapeModePlotMaker.find_plane_mesh_intersection(mesh, proj)
                    if swapxy_on_zproj and dim=='z':
                        coords = coords[:, ::-1]
                    contours[dim][alias].append(coords)
        return contours
    
    def workflow(self):
        return
