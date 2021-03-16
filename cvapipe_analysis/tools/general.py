import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsimageio import AICSImage
import matplotlib.pyplot as plt

from .shapespace import ShapeSpace, ShapeSpaceBasic

def load_config_file():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    return config

def create_workflow_file_from_config():
    config = load_config_file()
    local_staging = config['project']['local_staging']
    with open("workflow_config.json", "w") as fj:
        json.dump({"project_local_staging_dir": local_staging}, fj)

def read_chunk_of_dataframe(cfg):
        
    # Keep the header
    skip = cfg['skip']
    if skip > 0:
        skip = range(1, skip+1)
    
    df = pd.read_csv(cfg['csv'], index_col='CellId', skiprows=skip, nrows=cfg['nrows'])
    
    return df
    

def get_segmentations(path_seg, channels):

    """
    Find the segmentations that should be used for features
    calculation.

    Parameters
    --------------------
    path_seg: str
        Path to the 4D binary image.
    channels: list of str
        Name of channels of the 4D binary image.

    Returns
    -------
    result: seg_nuc, seg_mem, seg_str
        3D binary images of nucleus, cell and structure.
    """

    seg_channel_names = ['dna_segmentation',
                         'membrane_segmentation',
                         'struct_segmentation_roof']

    if not all(name in channels for name in seg_channel_names):
        raise ValueError("One or more segmentation channels was\
        not found.")

    ch_dna = channels.index('dna_segmentation')
    ch_mem = channels.index('membrane_segmentation')
    ch_str = channels.index('struct_segmentation_roof')
    
    segs = AICSImage(path_seg).data.squeeze()

    return segs[ch_dna], segs[ch_mem], segs[ch_str]
    
def get_raws(path_raw, channels):

    """
    Find the raw images.

    Parameters
    --------------------
    path_raw: str
        Path to the 4D raw image.
    channels: list of str
        Name of channels of the 4D raw image.

    Returns
    -------
    result: nuc, mem, struct
        3D raw images of nucleus, cell and structure.
    """

    ch_dna = channels.index('dna')
    ch_mem = channels.index('membrane')
    ch_str = channels.index('structure')
    
    raw = AICSImage(path_raw).data.squeeze()

    return raw[ch_dna], raw[ch_mem], raw[ch_str]

class LocalStagingWriter:
    """
    Support class. Should not be instantiated directly.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, config):
        self.config = config
        self.set_abs_path_to_local_staging_folder(config['project']['local_staging'])
        
    def set_abs_path_to_local_staging_folder(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self.abs_path_local_staging = path
        
class DataProducer(LocalStagingWriter):
    """
    DESC
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, config):
        super().__init__(config)

    def set_row(self, row):
        self.row = row

    def set_row_with_cellids(self, row):
        self.row = row
        self.CellIds = self.row.CellIds
        if isinstance(self.CellIds, str):
            self.CellIds = eval(self.CellIds)
        
    def get_rel_output_file_path_as_str(self, row):
        file_name = self.get_output_file_name(row)
        return f"{self.abs_path_local_staging.name}/{self.subfolder}/{file_name}"

    def check_output_exist(self, row):
        rel_path_to_output_file = self.get_rel_output_file_path_as_str(row)
        if Path(rel_path_to_output_file).is_file():
            return rel_path_to_output_file
        return None

    # Might not be needed.
    def load_parameterization_manifest(self):
        self.df = pd.read_csv(self.abs_path_local_staging/"parameterization/manifest.csv", index_col='CellId')
        print(f"Dataframe loaded: {self.df.shape}")
    
    def get_available_parameterized_intensities(self):
        return [k for k in self.config['parameterization']['intensities'].keys()]
    
    def read_parameterized_intensity(self, index, return_intensity_names=False):
        abs_path_to_rep_file = self.abs_path_local_staging/f"parameterization/representations/{index}.tif"
        code = AICSImage(abs_path_to_rep_file)
        intensity_names = code.get_channel_names()
        code = code.data.squeeze()
        if return_intensity_names:
            return code, intensity_names
        return code

    def read_agg_parameterized_intensity(self, row):
        abs_path_to_rep_file = self.abs_path_local_staging/f"aggregation/repsagg/{self.get_aggrep_file_name(row)}"
        agg_code = AICSImage(abs_path_to_rep_file).data.squeeze()
        return agg_code
    
    def execute(self, row):
        rel_path_to_output_file = self.check_output_exist(row)
        if (rel_path_to_output_file is None) or self.config['project']['overwrite']:
            try:
                self.workflow(row)
                rel_path_to_output_file = self.save()
            except Exception as ex:
                rel_path_to_output_file = None
            except KeyboardInterrupt:
                sys.exit()
        self.status(row.name, rel_path_to_output_file)
        return rel_path_to_output_file
    
    def load_results_in_single_dataframe(self):
        df = pd.DataFrame()
        abs_path_to_output_folder = self.abs_path_local_staging / self.subfolder
        for f in tqdm(os.listdir(abs_path_to_output_folder)):
            try:
                df_tmp = pd.read_csv(abs_path_to_output_folder/f)
                df = df.append(df_tmp, ignore_index=True)
            except Exception as e:
                print(f"ERROR {e}, file: {f}")
        return df
    
    @staticmethod
    def status(idx, output):
        msg = "FAILED" if output is None else "complete"
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
