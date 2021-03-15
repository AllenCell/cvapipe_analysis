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

from cvapipe_analysis.tools import general

class StereotypyPlotMaker(general.PlotMaker):
    """
    DESC
    
    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    dpi = 300
    subfolder = 'stereotypy/plots'
    max_number_of_pairs = 0

    def __init__(self, config):
        super().__init__(config)
        self.structures = config['structures']['desc']

    def set_max_number_of_pairs(self, n):
        self.max_number_of_pairs = n if n > 0 else 0

    def make_plot(self, df):
        labels = []
        fig, ax = plt.subplots(1,1, figsize=(7,8), dpi=self.dpi)
        for sid, sname in enumerate(reversed(self.structures.keys())):
            df_s = df.loc[df.structure_name==sname]
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
        ax.set_title("-".join([str(df[f].unique()[0])
                               for f in ['intensity','shapemode','bin']])
                    )
        ax.grid(True)
        plt.tight_layout()
        return fig
            
    def execute(self, df, save_as=None, **kwargs):
        if 'dpi' in kwargs: self.dpi = kwargs['dpi']
        fig = self.make_plot(df)
        self.save(fig, save_as)