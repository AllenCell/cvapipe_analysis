import os
import sys
import argparse
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

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
        self.ni = len(indexes_i)
        self.nj = len(indexes_j)
        self.indexes = indexes_i.tolist() + indexes_j.tolist()
        self.corrs = np.zeros((self.ni, self.nj), dtype=np.float32)
        _, aliases = self.read_parameterized_intensity(indexes_i[0], True)
        if len(aliases) > 1:
            raise ValueError("Only single channel represenations are supported.")
        self.rep_alias = aliases[0]
        return

    def workflow(self):
        self.reps = self.load_representations(self.indexes)
        vsize = int(sys.getsizeof(self.reps)) / float(1 << 20)
        print(f"Data shape: {self.reps.shape} ({self.reps.dtype}, {vsize:.1f}Mb)")

        import pdb; pdb.set_trace()

        _ = Parallel(n_jobs=self.ncores, backend="threading")(
            delayed(self.get_ij_pair_correlation_and_save)(p)
            for p in tqdm(self.iter_over_ij_pairs(), total=self.ni*self.nj)
        )

        import pdb; pdb.set_trace()

        # with concurrent.futures.ProcessPoolExecutor(self.ncores) as executor:
        #         executor.map(self.get_ij_pair_correlation_and_save, self.iter_over_ij_pairs())
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

    def save_ij_pair_correlation(self, idxi, idxj, corr):
        index_i = self.indexes[idxi]
        index_j = self.indexes[self.ni + idxj]
        for index_folder, index_file in zip([index_i, index_j], [index_j, index_i]):
            path = self.control.get_staging() / f"{self.subfolder}/{index_folder}"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"{index_file}.{self.rep_alias}", "w") as ftxt:
                ftxt.write(f"{corr:.5f}")
        return

    def iter_over_ij_pairs(self):
        for idxi in range(self.ni):
            for idxj in range(self.nj):
                    yield (idxi, idxj)

    def get_ij_pair_correlation(self, idxi, idxj):
        r1 = self.reps[idxi]
        r2 = self.reps[self.ni + idxj]
        return self.correlate_representations(r1, r2)

    def get_ij_pair_correlation_and_save(self, pair):
        idxi, idxj = pair
        self.corrs[idxi, idxj] = self.get_ij_pair_correlation(idxi, idxj)
        # self.save_ij_pair_correlation(idxi, idxj, corr)
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
