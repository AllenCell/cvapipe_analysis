import os
import json
import dask
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import cluster as spcluster

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