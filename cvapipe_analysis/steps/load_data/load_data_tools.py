import os
import uuid
import quilt3
import concurrent
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from ...tools import io, general

class DataLoader(io.LocalStagingIO):
    """
    Functionalities for downloading the variance
    dataset used in the paper or load a custom
    dataset specified as an input csv file.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    packages = {
        "default": "aics/hipsc_single_cell_image_dataset",
        "non-edges": "aics/hipsc_single_nonedge_cell_image_dataset",
        "edges": "aics/hipsc_single_edge_cell_image_dataset",
        "i1": "aics/hipsc_single_i1_cell_image_dataset",
        "i2": "aics/hipsc_single_i2_cell_image_dataset",
        "m1": "aics/hipsc_single_m1_cell_image_dataset",
        "m2": "aics/hipsc_single_m2_cell_image_dataset"
    }
    registry = "s3://allencell"
    required_df_columns = [
        'CellId',
        'structure_name',
        'crop_seg',
        'crop_raw'
    ]
    download_raw_data = True

    def __init__(self, control):
        super().__init__(control)
        self.subfolder = 'loaddata'

    def disable_download_of_raw_data(self):
        self.download_raw_data = False

    def load(self, parameters):
        if any(p in parameters for p in ["csv", "fmsid"]):
            df = self.download_local_data(parameters)
        else:
            df = self.download_quilt_data(parameters)
        df = self.drop_aliases_related_columns(df)

        return df

    def drop_aliases_related_columns(self, df):
        return df[[f for f in df.columns if not any(w in f for w in self.control.get_data_aliases())]]

    def download_quilt_data(self, parameters):

        print("Creating data folders...")
        seg_folder = self.control.get_staging()/f"{self.subfolder}/crop_seg"
        seg_folder.mkdir(parents=True, exist_ok=True)

        raw_folder = self.control.get_staging()/f"{self.subfolder}/crop_raw"
        raw_folder.mkdir(parents=True, exist_ok=True)

        pkg_name = "default"
        if "dataset" in parameters:
            pkg_name = parameters["dataset"]
        if pkg_name not in self.packages:
            raise ValueError(f"Package {pkg_name} not found. Packages available: {[k for k in self.packages.keys()]}.")
        self.pkg = quilt3.Package.browse(self.packages[pkg_name], self.registry)
        self.pkg["metadata.csv"].fetch(self.control.get_staging()/"manifest.csv")
        print("Reading manifest...")
        df_meta = pd.read_csv(self.control.get_staging()/"manifest.csv", index_col="CellId", low_memory=False)

        if "test" in parameters:
            ncells = 12
            if "ncells" in parameters:
                ncells = int(parameters["ncells"])
            print(f"Downloading test subset of {pkg_name} dataset.")
            df_meta = self.get_interphase_test_set(df_meta, n=ncells)
            for i, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
                self.pkg[row["crop_seg"]].fetch(self.control.get_staging()/f"loaddata/{row.crop_seg}")
                if self.download_raw_data:
                    self.pkg[row["crop_raw"]].fetch(self.control.get_staging()/f"loaddata/{row.crop_raw}")
        else:
            if self.download_raw_data:
                print("Downloading single cell raw images...")
                self.pkg["crop_raw"].fetch(raw_folder)
            print("Downloading single cell segmentations...")
            self.pkg["crop_seg"].fetch(seg_folder)

        print("Appending full path to file paths...")
        for index, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
            df_meta.at[index, "crop_seg"] = str(self.control.get_staging()/f"loaddata/{row.crop_seg}")
            if self.download_raw_data:
                df_meta.at[index, "crop_raw"] = str(self.control.get_staging()/f"loaddata/{row.crop_raw}")

        if not self.download_raw_data:
            df_meta = df_meta.drop(columns=["crop_raw"])

        return df_meta

    def download_local_data(self, parameters):
        use_fms = "fmsid" in parameters
        df = self.load_data_from_csv(parameters, use_fms)
        #self.is_dataframe_valid(df)
        df = df.set_index('CellId', drop=True)
        # if not use_fms:
        #     self.create_symlinks(df)
        return df

    def is_dataframe_valid(self, df):
        for col in self.required_df_columns:
            if col not in df.columns:
                raise ValueError(f"Input CSV is missing column: {col}.")
        return

    def create_symlinks(self, df):
        for col in ['crop_raw', 'crop_seg']:
            abs_path_data_folder = self.control.get_staging()/self.subfolder
            (abs_path_data_folder/col).mkdir(parents=True, exist_ok=True)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            idx = str(uuid.uuid4())[:12]
            for col in ['crop_raw', 'crop_seg']:
                src = Path(row[col])
                dst = abs_path_data_folder/f"{col}/{src.stem}_{idx}{src.suffix}"
                os.symlink(src, dst)
                df.loc[index, col] = dst
        return df

    @staticmethod
    def get_interphase_test_set(df, n=3):
        df_test = pd.DataFrame([])
        df = df.loc[df.cell_stage=='M0']# M0 = interphase
        for _, df_struct in df.groupby('structure_name'):
            df_test = df_test.append(df_struct.sample(n=n, random_state=666, replace=False))
        return df_test.copy()
