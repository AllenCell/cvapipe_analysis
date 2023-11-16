import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io as skio
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

    WARNING: This class is optimized to compute
    correlations on binary representations only.
    """

    def __init__(self, control):
        super().__init__(control)
        self.ncores = control.get_ncores()
        self.subfolder = 'correlation/values'
        self.pilr_size = control.get_parameterized_representation_size() # 532610 in the paper

    def workflow(self):
        self.load_representations()
        self.correlate_all()
        return

    def get_output_file_name(self):
        '''
        This function has to return a file name with extension
        so that the resulting string can be used as output
        verification. If class writes multiple files, then the
        file extension have to be manually altered.
        '''
        fname = f"{self.get_prefix_from_row(self.row)}.tif"
        return fname

    def save(self):
        save_as = self.get_output_file_path()
        skio.imsave(save_as, self.corrs)
        pd.DataFrame({"CellIds": self.row.CellIds}).to_csv(str(save_as).replace(".tif",".csv"))
        return f"{save_as}.tif"

    def read_pilr(self, eCellId):
        i, CellId = eCellId
        pilr, names = self.read_parameterized_intensity(CellId, return_intensity_names=True)
        self.pilrs[i] = pilr[names.index(self.row.alias)].flatten()
        return

    def load_representations(self):
        self.ncells = len(self.row.CellIds)
        self.pilrs = np.zeros((self.ncells, self.pilr_size), dtype=np.float32)
        self.control.display_array_size_in_mb(self.pilrs, self.print)

        _ = Parallel(n_jobs=self.ncores, backend="threading")(
            delayed(self.read_pilr)(eindex)
            for eindex in tqdm(enumerate(self.row.CellIds), total=self.ncells)
        )
        return

    def correlate_all(self):
        self.corrs = np.corrcoef(self.pilrs, dtype=np.float32)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch single cell feature extraction.")
    parser.add_argument("--staging", help="Path to staging.", required=True)
    parser.add_argument("--csv", help="Path to the dataframe.", required=True)
    args = vars(parser.parse_args())

    config = general.load_config_file(args["staging"])
    control = controller.Controller(config)

    df = pd.read_csv(args['csv'], index_col=0)
    df = df.T
    df.CellIds = df.CellIds.astype(object)
    for index, row in df.iterrows():
        df.at[index,"CellIds"] = eval(row.CellIds)
    calculator = CorrelationCalculator(control)
    for _, row in df.iterrows():
        '''Concurrent processes inside. Do not use concurrent here.'''
        calculator.execute(row)

