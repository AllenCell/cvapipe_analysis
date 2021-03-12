import yaml
import json
import pandas as pd
from pathlib import Path
from aicsimageio import AICSImage

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

class DataProducer:
    """
    Desc
    
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
        
    def get_rel_output_file_path_as_str(self, row):
        file_name = self.get_output_file_name(row)
        return f"{self.abs_path_local_staging.name}/{self.subfolder}/{file_name}"

    def check_output_exist(self, row):
        rel_path_to_output_file = self.get_rel_output_file_path_as_str(row)
        if Path(rel_path_to_output_file).is_file():
            return rel_path_to_output_file
        return None

    def load_parameterization_manifest(self):
        self.df = pd.read_csv(self.abs_path_local_staging/"parameterization/manifest.csv", index_col='CellId')
        print(f"Dataframe loaded: {self.df.shape}")
    
    @staticmethod
    def status(idx, output):
        msg = "FAILED" if output is None else "complete"
        print(f"Index {idx} {msg}. Output: {output}")
