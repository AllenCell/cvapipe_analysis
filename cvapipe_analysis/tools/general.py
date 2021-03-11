import yaml
import json
import pandas as pd
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