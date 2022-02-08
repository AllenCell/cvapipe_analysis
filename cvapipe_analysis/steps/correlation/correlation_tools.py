import os
import sys
import argparse
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import io as skio
from joblib import Parallel, delayed

from cvapipe_analysis.tools import io, general, controller, bincorr

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

    WARNING: This class is optimized to compute
    correlations on binary representations only.
    """

    rep_length = 532610 # Need some work to make general

    def __init__(self, control):
        super().__init__(control)
        self.ncores = control.get_ncores()
        self.subfolder = 'correlation/values'

    def read_representation_as_boolean(self, eindex):
        i, index = eindex
        rep = self.read_parameterized_intensity(index).astype(bool).flatten()
        if not rep.sum():
            self.usable[i] = False
        self.reps[i] = rep
        return

    def load_representations(self):
        self.ncells = len(self.row.CellIds)
        self.usable = np.ones(self.ncells, dtype=bool)
        self.reps = np.zeros((self.ncells, self.rep_length), dtype=bool)
        repsize = int(sys.getsizeof(self.reps)) / float(1 << 20)
        print(f"Representations shape: {self.reps.shape} ({self.reps.dtype}, {repsize:.1f}Mb)")

        self.corrs = np.zeros((self.ncells, self.ncells), dtype=np.float32)
        corrssize = int(sys.getsizeof(self.corrs)) / float(1 << 20)
        print(f"Correlations shape: {self.corrs.shape} ({self.corrs.dtype}, {corrssize:.1f}Mb)")

        print(f"Loading representations using {self.ncores} cores...")

        _ = Parallel(n_jobs=self.ncores, backend="threading")(
            delayed(self.read_representation_as_boolean)(eindex)
            for eindex in tqdm(enumerate(self.row.CellIds), total=self.ncells)
        )

    def get_next_pair(self):
        for i in range(self.ncells):
            for j in range(i+1, self.ncells):
                yield (i, j)

    def correlate_ij(self, ij):
        i, j = ij
        corr = np.nan
        if self.usable[i] and self.usable[j]:
            corr = bincorr.calculate(self.reps[i], self.reps[j], self.rep_length)
        self.corrs[i, j] = self.corrs[j, i] = corr
        return

    def workflow(self):
        self.load_representations()
        npairs = int(self.ncells*(self.ncells-1)/2)

        _ = Parallel(n_jobs=self.ncores, backend="threading")(
            delayed(self.correlate_ij)(ij)
            for ij in tqdm(self.get_next_pair(), total=npairs, miniters=self.ncells)
        )
        return

    def get_output_file_name(self):
        fname = self.get_prefix_from_row(self.row)
        return fname

    def save(self):
        save_as = self.get_output_file_path()
        skio.imsave(f"{save_as}.tif", self.corrs)
        pd.DataFrame({"CellIds": self.row.CellIds}).to_csv(f"{save_as}.csv")
        return f"{save_as}.tif"

if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description='Batch correlation calculation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col=0)
    df = df.T
    df.CellIds = df.CellIds.astype(object)
    for index, row in df.iterrows():
        df.at[index,"CellIds"] = eval(row.CellIds)[:600]
    calculator = CorrelationCalculator(control)
    for _, row in df.iterrows():
        '''Concurrent processes inside. Do not use concurrent here.'''
        calculator.set_row(row)
        calculator.workflow()
        print(f"<REPS> = {calculator.reps.mean():.10f}")
        print(f"<CORR> = {calculator.corrs.mean():.10f}") 

