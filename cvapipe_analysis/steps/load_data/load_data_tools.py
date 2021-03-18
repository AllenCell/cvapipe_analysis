import quilt3
import pandas as pd

from cvapipe_analysis.tools import general, cluster

class DataLoader(general.DataProducer):
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
    extra_columns = [
        'roi',
        'fov_path'
    ]

    def __init__(self, config):
        super().__init__(config)
        
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
        path = self.abs_path_local_staging/self.subfolder
        for i, row in df_meta.iterrows():
            pkg[row["crop_raw"]].fetch(path/row["crop_raw"])
            pkg[row["crop_seg"]].fetch(path/row["crop_seg"])
        return df_meta

    def is_dataframe_valid(self, df):
        for col in self.required_df_columns:
            if col not in df.columns:
                raise ValueError(f"Input CSV is missing column: {col}.")
        return

    def load_data_from_csv(self, parameters):
        df = pd.read_csv(parameters['csv'])
        self.is_dataframe_valid(df)
        df = df[self.required_df_columns+
                [f for f in self.extra_columns if f in df.columns]
               ]
        df = df.set_index('CellId', drop=True)
        return df
    
    @staticmethod
    def get_interphase_test_set(df):
        df_test = pd.DataFrame([])
        df = df.loc[df.cell_stage=='M0']# M0=interphase
        for struct, df_struct in df.groupby('structure_name'):
            df_test = df_test.append(df_struct.sample(n=12, random_state=666, replace=False))
        return df_test.copy()
