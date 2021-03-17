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

