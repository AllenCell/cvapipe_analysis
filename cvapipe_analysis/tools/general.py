import os
import sys
import yaml
import json
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from aicsimageio import AICSImage
from contextlib import contextmanager

from cvapipe_analysis.tools import controller
from .shapespace import ShapeSpaceBasic, ShapeSpace

def load_config_file():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    return config

def save_config_file(config, abs_path_to_folder):
    abs_path_to_folder = Path(abs_path_to_folder)
    with open(abs_path_to_folder/'parameters.yaml', 'w') as f:
        yaml.dump(config, f)
    return

def create_workflow_file_from_config():
    config = load_config_file()
    local_staging = config['project']['local_staging']
    with open('workflow_config.json', 'w') as fj:
        json.dump({'project_local_staging_dir': local_staging}, fj)

def get_date_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
@contextmanager
def configuration(abs_path_to_folder=None):
    config = load_config_file()
    control = controller.Controller(config)
    try:
        control.log({"start": get_date_time()})
        yield control
    finally:
        if abs_path_to_folder is not None:
            control.log({"end": get_date_time()})
            save_config_file(control.config, abs_path_to_folder)
        else: pass
        
class LocalStagingWriter:
    """
    Support class. Should not be instantiated directly.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, control):
        self.control = control

    def get_abs_path_to_step_folder(self):
        '''Derived classes must set self.stepfolder.
        Maybe use metaclass here.'''
        path = self.control.get_staging() / self.stepfolder
        return path
        
class LocalStagingReader(LocalStagingWriter):
    """
    Class that provides functionalities to read results
    from local_staging.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, control, row):
        super().__init__(control)
        self.row = row
        
    def get_single_cell_images(self, imtype):
        segs = {}
        path = self.row[imtype]
        if str(self.control.get_staging()) not in path:
            path = self.control.get_staging()/f"loaddata/{self.row[imtype]}"
        imgs = AICSImage(path).data.squeeze()
        for ch, img in zip(eval(self.row.name_dict)[imtype], imgs):
            segs[ch] = img
        return segs
        
class DataProducer(LocalStagingWriter):
    """
    Functionalities for steps that produce results
    that should be saved in the local_staging folder.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, control):
        super().__init__(control)

    def set_row(self, row):
        self.row = row

    def set_row_with_cellids(self, row):
        self.row = row
        self.CellIds = self.row.CellIds
        if isinstance(self.CellIds, str):
            self.CellIds = eval(self.CellIds)

    def get_rel_output_file_path_as_str(self, row):
        file_name = self.get_output_file_name(row)
        return f"{self.control.get_staging().name}/{self.subfolder}/{file_name}"

    def check_output_exist(self, row):
        rel_path_to_output_file = self.get_rel_output_file_path_as_str(row)
        if Path(rel_path_to_output_file).is_file():
            return rel_path_to_output_file
        return None

    def load_shapemode_manifest(self):
        self.df = pd.read_csv(self.control.get_staging()/"shapemode/manifest.csv", index_col='CellId')
        print(f"Dataframe loaded: {self.df.shape}")

    def load_parameterization_manifest(self):
        self.df = pd.read_csv(self.control.get_staging()/"parameterization/manifest.csv", index_col='CellId')
        print(f"Dataframe loaded: {self.df.shape}")

    def read_parameterized_intensity(self, index, return_intensity_names=False):
        abs_path_to_rep_file = self.control.get_staging()/f"parameterization/representations/{index}.tif"
        code = AICSImage(abs_path_to_rep_file)
        intensity_names = code.get_channel_names()
        code = code.data.squeeze()
        if return_intensity_names:
            return code, intensity_names
        return code

    def read_agg_parameterized_intensity(self, row):
        abs_path_to_rep_file = self.control.get_staging()/f"aggregation/repsagg/{self.get_aggrep_file_name(row)}"
        agg_code = AICSImage(abs_path_to_rep_file).data.squeeze()
        return agg_code

    def execute(self, row):
        computed = False
        rel_path_to_output_file = self.check_output_exist(row)
        if (rel_path_to_output_file is None) or self.control.overwrite():
            try:
                self.workflow(row)
                computed = True
                rel_path_to_output_file = self.save()
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                rel_path_to_output_file = None
        self.status(row.name, rel_path_to_output_file, computed)
        return rel_path_to_output_file

    def load_results_in_single_dataframe(self):
        ''' Not sure this function is producing a column named index when
        the concordance results are loaded. Further investigation is needed
        here'''
        abs_path_to_output_folder = self.control.get_staging() / self.subfolder
        files = [abs_path_to_output_folder/f for f in os.listdir(abs_path_to_output_folder)]
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            df = pd.concat(
                tqdm(executor.map(self.load_csv_file_as_dataframe, files), total=len(files)),
                axis=0, ignore_index=True)
        return df

    @staticmethod
    def load_csv_file_as_dataframe(fpath):
        df = None
        try:
            df = pd.read_csv(fpath)
        except: pass
        return df

    @staticmethod
    def status(idx, output, computed):
        msg = "FAIL"
        if output is not None:
            msg = "COMPLETE" if computed else "SKIP"
        print(f"Index {idx} {msg}. Output: {output}")

    @staticmethod
    def get_aggrep_file_name(row):
        return f"{row.aggtype}-{row.intensity}-{row.structure_name}-{row.shapemode}-B{row.bin}-CODE.tif"

    @staticmethod
    def get_output_file_name(row):
        values = []
        for f in ['aggtype', 'intensity', 'structure_name', 'shapemode', 'bin']:
            if f in row:
                values.append(str(row[f]))
        return "-".join(values)
    
    @staticmethod
    def correlate_representations(rep1, rep2):
        pcor = np.corrcoef(rep1.flatten(), rep2.flatten())
        # Returns Nan if rep1 or rep2 is empty.
        return pcor[0,1]        
