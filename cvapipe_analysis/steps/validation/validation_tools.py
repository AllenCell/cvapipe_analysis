import argparse
import concurrent
import numpy as np
import pandas as pd
from aicsshparam import shparam, shtools
from skimage import measure as skmeasure
from skimage import morphology as skmorpho

from cvapipe_analysis.tools import io, general, controller, viz

class Validator(io.DataProducer):
    """
    Class for feature extraction.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """

    def __init__(self, control):
        super().__init__(control)
        self.subfolder = "validation/output"

    def workflow(self):
        self.print(f"Starting validation on cell {self.row.name}...")
        self.load_single_cell_data()
        self.print(f"Calculating reconstruction error...")
        self.compute_reconstruction_error()
        self.print("Done.")
        return

    def get_output_file_name(self):
        return f"{self.row.name}.csv"
    
    def save(self):
        save_as = self.get_output_file_path()
        self.df_err.to_csv(save_as, index=False)
        return save_as
    
    def compute_reconstruction_error(self):
        self.df_err = pd.DataFrame([])
        for alias in self.control.get_aliases_for_feature_extraction():
            if self.control.should_calculate_shcoeffs(alias):
                df_tmp = self.compute_reconstruction_error_for_alias(alias)
                self.df_err = self.df_err.append(df_tmp, ignore_index=True)
        return

    def compute_reconstruction_error_for_alias(self, alias):
        df_err = []
        channel = self.control.get_channel_from_alias(alias)
        chId = self.channels.index(channel)
        for lrec in range(2, 2*self.control.get_lmax()):
            (_, grid_rec), (_, mesh, grid, _) = shparam.get_shcoeffs(
                image=self.data[chId],
                lmax=lrec,
                sigma=self.control.get_sigma(alias),
                alignment_2d=False
            )
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            d12, d21 = viz.MeshToolKit.get_meshes_distance(mesh, mesh_rec)
            d12 = np.median(d12)
            d21 = np.median(d21)
            df_err.append({
                "lrec": lrec,
                "alias": alias,
                "d12": d12,
                "d21": d21,
                # Hard coding pixel size for now
                "error": 0.5*(d12+d21)*0.108,
                "CellId": self.row.name
            })
        df_err = pd.DataFrame(df_err)
        return df_err

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch single cell feature extraction.")
    parser.add_argument("--staging", help="Path to staging.", required=True)
    parser.add_argument("--csv", help="Path to the dataframe.", required=True)
    args = vars(parser.parse_args())

    config = general.load_config_file(args["staging"])
    control = controller.Controller(config)

    df = pd.read_csv(args["csv"], index_col="CellId")
    print(f"Processing dataframe of shape {df.shape}")

    validator = Validator(control)
    with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
        executor.map(validator.execute, [row for _, row in df.iterrows()])

