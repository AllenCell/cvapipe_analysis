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
from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode

class StereotypyCalculator(general.DataProducer):
    """
    Provides the functionalities necessary for
    calculating the stereotypy of cells using their
    parameterized intensity representation.
    >> Plotting: TBD
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    subfolder = 'stereotypy/values'
    
    def __init__(self, config):
        super().__init__(config)
        
    @staticmethod
    def get_output_file_name(row):
        return f"{row.intensity}-{row.structure_name}-{row.shapemode}-B{row.bin}.csv"
    
    def save(self):
        df = pd.DataFrame(
            list(zip(self.CellIds,self.CellIdsTarget,self.pcorrs)),
            columns=['CellId1', 'CellId2', 'Pearson']
        )
        save_as = self.get_rel_output_file_path_as_str(self.row)
        df.to_csv(save_as)
        return save_as

    def shuffle_target_cellids(self):
        if len(self.CellIds) < 2: raise
        self.CellIdsTarget = self.CellIds.copy()
        while True:
            random.shuffle(self.CellIdsTarget)
            for id1, id2 in zip(self.CellIds,self.CellIdsTarget):
                if id1 == id2:
                    break
            else: return
    
    def iter_over_pairs(self):
        for id1, id2 in zip(self.CellIds, self.CellIdsTarget):
            yield (id1, id2)
    
    def correlate(self, indexes):
        names = self.get_available_parameterized_intensities()
        rep1 = self.read_parameterized_intensity(indexes[0])
        rep2 = self.read_parameterized_intensity(indexes[1])
        rep1 = rep1[names.index(self.row.intensity)]
        rep2 = rep2[names.index(self.row.intensity)]
        pcor = np.corrcoef(rep1.flatten(), rep2.flatten())
        # Returns Nan if rep1 or rep2 is empty.
        return pcor[0,1]        
    
    def calculate(self, row):
        self.digest_row_with_cellids(row)
        self.shuffle_target_cellids()
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            self.pcorrs = list(
                executor.map(self.correlate, self.iter_over_pairs()))
        return
    
    def workflow(self, row):
        rel_path_to_output_file = self.check_output_exist(row)
        if (rel_path_to_output_file is None) or self.config['project']['overwrite']:
            try:
                self.calculate(row)
                rel_path_to_output_file = self.save()
            except:
                rel_path_to_output_file = None
        self.status(row.name, rel_path_to_output_file)
        return rel_path_to_output_file

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Batch stereotypy calculation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())
    
    df = pd.read_csv(args['csv'], index_col=0)

    config = general.load_config_file()
    
    calculator = StereotypyCalculator(config)
    for _, row in df.iterrows():
        '''Concurrent processes inside. Do not use concurrent here.'''
        calculator.workflow(row)
