import os
import vtk
import json
import psutil
import pickle
import random
import argparse
import warnings
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
from aicscytoparam import cytoparam
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler

from cvapipe_analysis.tools import io, general, controller


class CorrelationCalculator(io.DataProducer):
    """
    Provides the functionalities necessary for
    calculating the correlation of cells using their
    parameterized intensity representation.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, control):
        super().__init__(control)
        self.ncores = control.get_ncores()
        self.subfolder = 'correlation/values'

    def set_indexes(self, indexes_i, indexes_j):
        self.indexes_i = indexes_i
        self.indexes_j = indexes_j
        _, self.rep_aliases = self.read_parameterized_intensity(indexes_i[0], True)
        return

    def workflow(self):
        self.reps_i = self.load_representations(self.indexes_i)
        self.reps_j = self.load_representations(self.indexes_j)
        with concurrent.futures.ProcessPoolExecutor(self.ncores) as executor:
                executor.map(self.get_ij_pair_correlation_and_save, self.iter_over_ij_pairs())
        return

    def get_output_file_name(self):
        pass

    def load_representations(self, indexes):
        # Representations of all cells in the manifest should be present.
        # Given the size of the single cell dataset, this function
        # currently only supports the segmented images so the data
        # can be load as type bool. Otherwise we run into lack of
        # memory issues to load the representations.
        with concurrent.futures.ProcessPoolExecutor(self.ncores) as executor:
            reps = list(tqdm(
                executor.map(self.read_parameterized_intensity_as_boolean, indexes),
            total=len(indexes)))
        return np.array(reps)

    def save_ij_pair_correlation(self, idxi, idxj, corrs):
        index_i = self.indexes_i[idxi]
        index_j = self.indexes_j[idxj]
        for index_folder, index_file in zip([index_i, index_j], [index_j, index_i]):
            path = self.control.get_staging() / f"{self.subfolder}/{index_folder}"
            path.mkdir(parents=True, exist_ok=True)
            for alias, corr in corrs.items():
                with open(path / f"{index_file}.{alias}", "w") as ftxt:
                    ftxt.write(f"{corr:.5f}")
        return

    def iter_over_ij_pairs(self):
        for idxi, index_i in enumerate(self.indexes_i):
            for idxj, index_j in enumerate(self.indexes_j):
                    yield (idxi, idxj)

    def get_npairs(self):
        n1 = len(self.indexes_i)
        n2 = len(self.indexes_j)
        return n1*n2

    def get_ij_pair_correlation(self, idxi, idxj):
        corrs = {}
        rep1, rep2 = self.reps_i[idxi], self.reps_j[idxj]
        index_i, index_j = self.indexes_i[idxi], self.indexes_j[idxj]
        for alias, r1, r2 in zip(self.rep_aliases, rep1, rep2):
            corrs[alias] = self.correlate_representations(r1, r2)
        return corrs

    def get_ij_pair_correlation_and_save(self, pair):
        idxi, idxj = pair
        corrs = self.get_ij_pair_correlation(idxi, idxj)
        self.save_ij_pair_correlation(idxi, idxj, corrs)
        return

if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description='Batch correlation calculation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col=0)

    calculator = CorrelationCalculator(control)
    indexes_i = df.loc[df.Group==0].index
    indexes_j = df.loc[df.Group==1].index
    calculator.set_indexes(indexes_i, indexes_j)
    calculator.workflow()
