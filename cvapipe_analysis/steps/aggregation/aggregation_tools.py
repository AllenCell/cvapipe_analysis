import vtk
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ..shapemode.avgshape import digitize_shape_mode

def _read_parameterization(row):
    
    """
    Read parameterized intensity representations
    AICSImage.

    Parameters
    --------------------
    df: pd.Series
        Pandas dataframe row that constain the column
        CellRepresentationPath that points to the
        cell representation generated in step
        parameterization.

    Returns
    -------
    code: AICSImage
        AICSImage of the TIF file that represents the
        parameterized intensity representation.
    """
    
    # Delayed Image Reading. Dimension S is kept and
    # used to run over different cell ids. Z=0
    # since the representation is always 2D.
    code = AICSImage(row['CellRepresentationPath'])
    code = code.get_image_dask_data("SCYX", T=0, Z=0)
    
    return code
    
def aggregate_intensity_representations(
    df: pd.DataFrame,
    distributed_executor_address: Optional[str] = None
):

    """
    Aggregate the parameterized intensity representation
    of all cells contained in the input dataframe. All
    cells are processed in parallel via Dask.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains one cell per row and
        the column CellRepresentationPath with the path to
        the corresponding representation.

    Returns
    -------
    agg: dict
        Dict with one key for each representation encoded
        in the images and another key for the aggregation
        type: avg for average and std for standard deviation.
    """
    
    # Process each row
    with DistributedHandler(distributed_executor_address) as handler:
        representations = handler.batched_map(
            _read_parameterization,
            [row for _, row in df.iterrows()]
        )
    # Stack images together
    representations = np.vstack(representations)
    
    # Use dask to compute mean and std
    avg_rep = representations.mean(axis=0).compute()
    std_rep = representations.std(axis=0).compute()

    # Load first image to find channel names
    index = df.index[0]
    img_path = df.at[index,'CellRepresentationPath']
    channel_names = AICSImage(img_path).get_channel_names()
    
    agg = {}
    for ch, ch_name in enumerate(channel_names):
        agg[ch_name] = {
            'avg': avg_rep[ch],
            'std': std_rep[ch]
        }

    return agg

def aggregate_intensities_of_shape_mode(
    df: pd.DataFrame,
    pc_names: List,
    pc_idx: int,
    save_dir: Path,
    nbins: Optional[int] = 9
):

    """
    Aggregate the representations per srtructure name and
    per bin number. The final aggregation is saved as a
    hyperstack image with format: TC1YX, where T is used
    to store the different bins and C is used to store the
    different structures.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains one cell per row and
        the column CellRepresentationPath with the path to
        the corresponding representation.
    pc_names: List
        List with all features that should be used to
        discretize the PC space (shape space).
    pc_idx: int
        Index of the feature in the list pc_names that will
        be aggregated.
    save_dir: Path
        Path where to save the results.
    nbins: int
        Number of bins that was used to discretize the shape
        space.

    Returns
    -------
    result: List
        List of dict with keys for the pc_name, representation
        name, aggregation type and path to aggregated
        representation.
        
    TBD:
    -----
    
        - Save a CSV with the numberof cells used in the
        aggregation process.
    
    """
    
    # Name of the PC to be processed
    pc_name = pc_names[pc_idx]
    
    # Find the indexes of cells in each bin
    df_filtered, bin_indexes, _ = digitize_shape_mode(
        df = df,
        feature = pc_name,
        nbins = nbins,
        filter_based_on = pc_names
    )

    # Loop over different structures
    agg_struct = {}
    n_structs = len(df_filtered.structure_name.unique())
    for struct, df_struct in tqdm(df_filtered.groupby('structure_name'), total=n_structs):
        # Loop over bins
        agg_bins = {}
        for b in df_filtered.bin.unique():

            df_struct_bin = df_struct.loc[df_struct.bin==b]

            ncells = df_struct_bin.shape[0]

            if ncells > 0:

                agg = aggregate_intensity_representations(df_struct_bin)

                agg_bins[b] = agg
                
        agg_struct[struct] = agg_bins

    # Find all names of the representations in this data
    s = list(agg_struct.keys())[0] # temp structure name
    b = list(agg_struct[s].keys())[0] # temp bin value
    # Every pair (s,b) should share the same names of
    # representation and aggregation types (avg and std).
    rep_names = list(agg_struct[s][b].keys())
    agg_types = list(agg_struct[s][b][rep_names[0]])
    
    # Find dimensions of 2D representations:
    ny, nx = agg_struct[s][b][rep_names[0]][agg_types[0]].shape
    
    STRUCT_SEQ = ["FBL","NPM1","SON","SMC1A","HIST1H2BJ","LMNB1" ,"NUP153" ,"SEC61B","ATP2A2","TOMM20","SLC25A17","RAB5A","LAMP1","ST6GAL1","CETN2","TUBA1B","AAVS1","ACTB","ACTN1","MYH10","GJA1","TJP1","DSP","CTNNB1","PXN"]
    
    # Create 5D hyperstacks
    result = []
    for rn in rep_names:
        for at in agg_types:
            # Empty hypermatrix
            data = np.zeros((1,nbins,n_structs,1,ny,nx), np.float32)
            # Populate hypermatrix with representations
            for sid, struct in enumerate(STRUCT_SEQ):
                if struct in agg_struct:
                    for b in range(1, nbins+1):
                        if b in agg_struct[struct]:
                            data[0,b-1,sid,0] = agg_struct[struct][b][rn][at]
    
            # Convert hypermatrix to hyperstack
            data = AICSImage(data)
            data.channel_names = STRUCT_SEQ
            
            # Save hyperstack
            save_as = save_dir / f"{pc_name}_{rn}_{at}.tif"
            with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
                writer.save(
                    data.get_image_data('TCZYX', S=0),
                    dimension_order = 'TCZYX',
                    image_name = f"{pc_name}_{rn}_{at}",
                    channel_names = STRUCT_SEQ
                )

            # Store file name
            result.append({
                'feature': pc_name,
                'representation': rn,
                'aggtype': at,
                'AggFilePath': save_as
            })
            
    return result

