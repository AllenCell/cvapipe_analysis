import os
import vtk
import json
import psutil
import pickle
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
from aicscytoparam import cytoparam
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import concurrent

from cvapipe_analysis.tools import general, cluster

class ConcordanceCalculator(general.DataProducer):
    """
    Provides the functionalities necessary for
    calculating the concordance of cells using
    their parameterized intensity representation.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    subfolder = 'concordance/values'
    
    def __init__(self, config):
        super().__init__(config)
        
    def save(self):
        save_as = self.get_rel_output_file_path_as_str(self.row)
        pd.DataFrame([self.row]).to_csv(save_as, index=False)
        return save_as
    
    def workflow(self, row):
        self.set_row(row)
        agg_rep1=self.read_agg_parameterized_intensity(
            row.rename({'structure_name1': 'structure_name'}))
        agg_rep2=self.read_agg_parameterized_intensity(
            row.rename({'structure_name2': 'structure_name'}))
        self.row['Pearson']=self.correlate_representations(agg_rep1, agg_rep2)
        return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Batch concordance calculation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())
    
    df = pd.read_csv(args['csv'], index_col=0)

    config = general.load_config_file()
        
    calculator = ConcordanceCalculator(config)            
    with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
        executor.map(calculator.execute, [row for _,row in df.iterrows()])


        
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import cluster
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from aicsimageio import AICSImage, writers

def make_labels(bin_num):
    # Labels based on structure name and number of cell available per structure per bin
    labels = []
    nstrucs = len(structures_list)-1
    for i in range(nstrucs):
        gene = structures_list[rank[i]]
        ncells = df.loc[(df.index.isin(bin_indexes[bin_num][1]))&(df.structure_name==gene)].shape[0]
        label = f'{gene_to_struct[gene]} (n={ncells})'
        labels.append(label)
    return labels

def get_correlation_matrix(bin_num):
    # Create correlation matrix for a given bin
    nstrucs = len(structures_list)-1
    pcorr = np.zeros((nstrucs,nstrucs), dtype=np.float)
    for sid1, sname1 in tqdm(enumerate(structures_list[:-1]), total=nstrucs, leave=False):
        for sid2, sname2 in enumerate(structures_list[:-1]):
            v1 = avg_img[sid1,bin_num].flatten()
            v2 = avg_img[sid2,bin_num].flatten()
            valids = avg_img[-1,bin_num].flatten()>0
            pcorr[sid1,sid2] = np.corrcoef(v1[valids],v2[valids])[0,1]
    pcorr_ranked = pcorr[rank,:]
    pcorr_ranked = pcorr_ranked[:,rank]
    return pcorr_ranked

def get_correlation_between_reps(rep1, rep2):
    pcorr = np.corrcoef(rep1, rep2)[0,1]
    return pcorr

<<<<<<<<<<<<<<<<< continue here:
    
    to read pickle: pkl.load( open ("myDicts.p", "rb") )

def get_correlation_matrix_from_reps()

def make_heatmap(mode, bin_num1, bin_num2=None, pcorr_ref=None, dendrogram=False):
    # Make heatmap or difference heatmaps. Also creates dendrogram
    pcorr_inf = get_correlation_matrix(bin_num=bin_num1)
    if bin_num2 is not None:
        pcorr_inf = np.tril(pcorr_inf, -1)
        pcorr_sup = get_correlation_matrix(bin_num=bin_num2)
        pcorr_sup = np.triu(pcorr_sup, 1)

        pcorr = pcorr_inf + pcorr_sup
        if pcorr_ref is not None:
            pcorr = pcorr_ref-pcorr
            np.fill_diagonal(pcorr, 0)
    else:
        pcorr = pcorr_inf
    
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    cmap = 'RdBu' if pcorr_ref is None else 'PRGn'
    ax.imshow(-pcorr, cmap=cmap, vmin=-1.0,vmax=1.0)
    for edge, spine in ax.spines.items():
            spine.set_visible(False)
    ax.set_xticks(np.arange(pcorr.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(pcorr.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    if bin_num2 is None:
        plt.savefig(f'{out}/heatmap_PC{mode}_B{bin_num1-4}.svg', format='svg')
    else:
        if pcorr_ref is None:
            plt.savefig(f'{out}/heatmap_triangle_PC{mode}_B{bin_num1-4}-B{bin_num2-4}.svg', format='svg', bbox_inches="tight")
        else:
            plt.savefig(f'{out}/heatmap_diff-triangle_PC{mode}_B{bin_num1-4}-B{bin_num2-4}.svg', format='svg', bbox_inches="tight")
    plt.close('all')
    
    if dendrogram:
        Z = cluster.hierarchy.linkage(pcorr, 'average')
        Z = cluster.hierarchy.optimal_leaf_ordering(Z, pcorr)
        fig, ax = plt.subplots(1,1, figsize=(6,3))
        dn = cluster.hierarchy.dendrogram(
            Z,
            labels = make_labels(bin_num1),
            leaf_rotation = 90
        )
        plt.savefig(f'{out}/dendrogram_PC{mode}_B{bin_num1-4}.svg',
                    format='svg', bbox_inches="tight")
        plt.close('all')
        
    return pcorr
'''