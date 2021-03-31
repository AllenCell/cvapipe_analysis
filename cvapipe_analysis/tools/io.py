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

class LocalStagingIO:
    """
    Class that provides functionalities to read and
    write at local_staging.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, control):
        self.control = control

    def get_single_cell_images(self, row, imtype):
        segs = {}
        path = row[imtype]
        if str(self.control.get_staging()) not in path:
            path = self.control.get_staging()/f"loaddata/{row[imtype]}"
        imgs = AICSImage(path).data.squeeze()
        for ch, img in zip(eval(row.name_dict)[imtype], imgs):
            segs[ch] = img
        return segs

    def get_abs_path_to_step_manifest(self, step):
        return self.control.get_staging()/f"{step}/manifest.csv"

    def load_step_manifest(self, step, clean=False):
        df = pd.read_csv(
            self.get_abs_path_to_step_manifest(step),
            index_col="CellId", low_memory=False
        )
        if clean:
            feats = ['mem_', 'dna_', 'str_']
            df = df[[c for c in df.columns if not any(s in c for s in feats)]]
        return df

    def read_parameterized_intensity(self, index, return_intensity_names=False):
        path = f"parameterization/representations/{index}.tif"
        abs_path_to_rep_file = self.control.get_staging()/path
        code = AICSImage(abs_path_to_rep_file)
        intensity_names = code.get_channel_names()
        code = code.data.squeeze()
        if return_intensity_names:
            return code, intensity_names
        return code

    def read_agg_parameterized_intensity(self, row):
        path = f"aggregation/repsagg/{self.get_aggrep_file_name(row)}"
        abs_path_to_rep_file = self.control.get_staging()/path
        agg_code = AICSImage(abs_path_to_rep_file).data.squeeze()
        return agg_code

    def load_results_in_single_dataframe(self):
        ''' Not sure this function is producing a column named index when
        the concordance results are loaded. Further investigation is needed
        here'''
        path_to_output_folder = self.control.get_staging() / self.subfolder
        files = [path_to_output_folder/f for f in os.listdir(path_to_output_folder)]
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            df = pd.concat(
                tqdm(executor.map(self.load_csv_file_as_dataframe, files), total=len(files)),
                axis=0, ignore_index=True)
        return df
    
    def get_output_file_path(self):
        path = f"{self.subfolder}/{self.get_output_file_name()}"
        return self.control.get_staging()/path

    def check_output_exist(self):
        path_to_output_file = self.get_output_file_path()
        if path_to_output_file.is_file():
            return path_to_output_file
        return None
    
    @staticmethod
    def status(idx, output, computed):
        msg = "FAIL"
        if output is not None:
            msg = "COMPLETE" if computed else "SKIP"
        print(f"Index {idx} {msg}. Output: {output}")

    @staticmethod
    def load_csv_file_as_dataframe(fpath):
        df = None
        try:
            df = pd.read_csv(fpath)
        except: pass
        return df
        
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
    
class DataProducer(LocalStagingIO):
    """
    Functionalities for steps that perform calculations
    in a per row fashion.
    
    Derived classes should implement:
        def __init__(self, control)
        def workflow(self, row)
        def get_output_file_name(row):
        def save(self)
    
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
        if "CellIds" in row:
            self.CellIds = self.row.CellIds
            if isinstance(self.CellIds, str):
                self.CellIds = eval(self.CellIds)

    def execute(self, row):
        computed = False
        self.set_row(row)
        path_to_output_file = self.check_output_exist()
        if (path_to_output_file is None) or self.control.overwrite():
            try:
                self.workflow()
                computed = True
                path_to_output_file = self.save()
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                path_to_output_file = None
        self.status(row.name, path_to_output_file, computed)
        return path_to_output_file
        