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
        for k in filters.keys():
            self.df = self.df.loc[self.df[k]==filters[k]]

    def execute(self, save_as=None, **kwargs):
        if 'dpi' in kwargs: self.dpi = kwargs['dpi']
        self.workflow()
        self.save(save_as)
            
    def save(self, save_as):
        for (fig, signature) in self.figs:
            if save_as is None:
                fig.show()
            else:
                fig.savefig(self.abs_path_local_staging/f"{self.subfolder}/{save_as}_{signature}.png")
                plt.close(fig)
            
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
        ax.set_title("-".join([str(self.df[f].unique())
                               for f in ['intensity','shapemode','bin']])
                    )
        ax.grid(True)
        plt.tight_layout()
        self.figs.append((fig, 'boxplot'))
        return

    def workflow(self):
        self.make_boxplot()
    
class ConcordancePlotMaker(PlotMaker):
    """
    Class for creating concordance heatmap.
    
    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    subfolder = 'concordance/plots'

    def __init__(self, config):
        super().__init__(config)
        self.structures = config['structures']['desc']

    def build_correlation_matrix(self):
        self.df_corr = self.df.groupby(['structure_name1', 'structure_name2']).agg(['mean']).reset_index()
        self.df_corr = self.df_corr.pivot(index='structure_name1', columns='structure_name2', values=('Pearson', 'mean'))
        self.df_corr = self.df_corr[self.structures]
        self.df_corr = self.df_corr.loc[self.structures]
        self.matrix = self.df_corr.values
        np.fill_diagonal(self.matrix, 1.0)
        return

    def make_heatmap(self):
        fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=self.dpi)
        cmap = 'RdBu'
        ax.imshow(-self.matrix, cmap=cmap, vmin=-1.0,vmax=1.0)
        for edge, spine in ax.spines.items():
                spine.set_visible(False)
        ax.set_xticks(np.arange(self.matrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.matrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        self.figs.append((fig, 'heatmap'))
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
        self.figs.append((fig, 'dendrogram'))
    
    def workflow(self):
        self.build_correlation_matrix()
        self.make_dendrogram()
        self.make_heatmap()
        return