import os
import uuid
import quilt3
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
    subfolder = 'loaddata'
    required_df_columns = [
        'CellId',
        'structure_name',
        'crop_seg',
        'crop_raw',
        'name_dict'
    ]

    def __init__(self, control):
        super().__init__(control)

    def load(self, parameters):
        if 'csv' in parameters:
            return self.load_data_from_csv(parameters)
        return self.download_quilt_data('test' in parameters)

    def download_quilt_data(self, test=False):
        pkg = quilt3.Package.browse(self.package_name, self.registry)
        df_meta = pkg["metadata.csv"]()
        if test:
            print('Downloading test dataset with 12 interphase cell images per structure.')
            df_meta = self.get_interphase_test_set(df_meta)
        path = self.control.get_staging()/self.subfolder
        for i, row in df_meta.iterrows():
            pkg[row["crop_raw"]].fetch(path/row["crop_raw"])
            pkg[row["crop_seg"]].fetch(path/row["crop_seg"])
        return df_meta

    def is_dataframe_valid(self, df):
        for col in self.required_df_columns:
            if col not in df.columns:
                raise ValueError(f"Input CSV is missing column: {col}.")
        return

    def create_symlinks(self, df):
        for col in ['crop_raw', 'crop_seg']:
            abs_path_data_folder = self.control.get_staging()/f"{self.subfolder}"
            (abs_path_data_folder/col).mkdir(parents=True, exist_ok=True)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            idx = str(uuid.uuid4())[:8]
            for col in ['crop_raw', 'crop_seg']:
                src = Path(row[col])
                dst = abs_path_data_folder/f"{col}/{src.stem}_{idx}{src.suffix}"
                os.symlink(src, dst)
                df.loc[index, col] = dst
        return df

    # TODO: Merge this upstream LocalStagingIO.load_csv_file_as_dataframe?
    def load_data_from_csv(self, parameters):
        df = pd.read_csv(parameters['csv'])
        self.is_dataframe_valid(df)
        df = df[self.required_df_columns].set_index('CellId', drop=True)
        self.create_symlinks(df)
        return df

    @staticmethod
    def get_interphase_test_set(df):
        df_test = pd.DataFrame([])
        df = df.loc[df.cell_stage=='M0']# M0 = interphase
        for struct, df_struct in df.groupby('structure_name'):
            df_test = df_test.append(df_struct.sample(n=12, random_state=666, replace=False))
        return df_test.copy()
