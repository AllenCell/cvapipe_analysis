import os
import uuid
import quilt3
import concurrent
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from cvapipe_analysis.tools import io

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

    package_name = "aics/hipsc_single_cell_image_dataset"
    registry = "s3://allencell"
    required_df_columns = [
        'CellId',
        'structure_name',
        'crop_seg',
        'crop_raw'
    ]

    def __init__(self, control):
        super().__init__(control)
        self.subfolder = 'loaddata'

    def load(self, parameters):
        if any(p in parameters for p in ["csv", "fmsid"]):
            return self.download_local_data(parameters)
        return self.download_quilt_data('test' in parameters)

    def download_quilt_data(self, test=False):
        self.pkg = quilt3.Package.browse(self.package_name, self.registry)
        # df_meta = pkg["metadata.csv"]()
        # Workaround the overflow error with the line above
        self.pkg["metadata.csv"].fetch(self.control.get_staging()/"manifest.csv")
        df_meta = pd.read_csv(self.control.get_staging()/"manifest.csv", index_col="CellId")
        if test:
            print('Downloading test dataset with 12 interphase cell images per structure.')
            df_meta = self.get_interphase_test_set(df_meta)

        seg_folder = self.control.get_staging()/f"{self.subfolder}/crop_seg"
        seg_folder.mkdir(parents=True, exist_ok=True)
        self.pkg["crop_seg"].fetch(seg_folder)

        raw_folder = self.control.get_staging()/f"{self.subfolder}/crop_raw"
        raw_folder.mkdir(parents=True, exist_ok=True)
        self.pkg["crop_raw"].fetch(raw_folder)

        for index, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
            df_meta.at[index, "crop_seg"] = str(self.control.get_staging()/f"loaddata/{row.crop_seg}")
            df_meta.at[index, "crop_raw"] = str(self.control.get_staging()/f"loaddata/{row.crop_raw}")

        return df_meta

    def download_local_data(self, parameters):
        use_fms = use_fms="fmsid" in parameters
        df = self.load_data_from_csv(parameters, use_fms)
        self.is_dataframe_valid(df)
        df = df.set_index('CellId', drop=True)
        if not use_fms:
            self.create_symlinks(df)
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
            idx = str(uuid.uuid4())[:8]
            for col in ['crop_raw', 'crop_seg']:
                src = Path(row[col])
                dst = abs_path_data_folder/f"{col}/{src.stem}_{idx}{src.suffix}"
                os.symlink(src, dst)
                df.loc[index, col] = dst
        return df

    @staticmethod
    def get_interphase_test_set(df):
        df_test = pd.DataFrame([])
        df = df.loc[df.cell_stage=='M0']# M0 = interphase
        for _, df_struct in df.groupby('structure_name'):
            df_test = df_test.append(df_struct.sample(n=12, random_state=666, replace=False))
        return df_test.copy()
