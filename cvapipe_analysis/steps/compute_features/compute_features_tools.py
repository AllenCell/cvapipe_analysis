import json
import argparse
import numpy as np
from aicsimageio import AICSImage
from aicsshparam import shtools, shparam
from skimage import measure as skmeasure
from skimage import morphology as skmorpho

        
def cast_features(features):
    
    """
    Cast feature values from numpy type to python so that the
    Json dict can be serialized.

    Parameters
    --------------------
    features: dict
        Dictionary of features.

    Returns
    -------
    features: dict
        Dictionary of features converted into python type.
    """
    
    for key, value in features.items():
        if isinstance(value, np.integer):
            features[key] = int(value)
        elif isinstance(value, np.float):
            features[key] = float(value)
        elif isinstance(value, np.ndarray):
            features[key] = value.tolist()
        else:
            # To deal with the case value = np.nan
            features[key] = int(value)
            
    return features

def get_surface_area(input_img):

    """
    Calculates the surface area of a binary shape by counting the
    number of boundary faces.

    Parameters
    --------------------
    input_image: ndarray
        3D input image representing the single cell segmentation that
        represents either dna, cell or structure.

    Returns
    -------
    result: int
        Number of boundary faces.
    """
    
    # Forces a 1 pixel-wide offset to avoid problems with binary
    # erosion algorithm

    input_img[:,:,[0,-1]] = 0
    input_img[:,[0,-1],:] = 0
    input_img[[0,-1],:,:] = 0

    input_img_surface = np.logical_xor(input_img, skmorpho.binary_erosion(input_img)).astype(np.uint8)

    # Loop through the boundary voxels to calculate the number of
    # boundary faces. Using 6-neighborhod.

    pxl_z, pxl_y, pxl_x = np.nonzero(input_img_surface)

    dx = np.array([ 0, -1,  0,  1,  0,  0])
    dy = np.array([ 0,  0,  1,  0, -1,  0])
    dz = np.array([-1,  0,  0,  0,  0,  1])
    
    surface_area = 0

    for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
        surface_area += 6 - input_img_surface[k+dz,j+dy,i+dx].sum()

    return int(surface_area)

def get_features(input_image, input_reference_image, compute_shcoeffs=True):
    
    """
    Extracts single cell features used in the variance paper.

    Parameters
    --------------------
    input_image: ndarray
        3D image representing the single cell segmentation that
        represents either dna, cell or structure.
    input_reference_image: ndarray
        3D image representing the reference that should be used
        align the input image. This is only used for the spherical
        harmonics expansion. In case None is provided, the input
        image is not aligned.

    Returns
    -------
    result: dict
        Dictionary with feature names and values.
    """

    features = {}
    
    # Binarize the input and cast to 8-bit
    input_image = (input_image>0).astype(np.uint8)
    
    input_image_lcc = skmeasure.label(input_image)
    
    # Number of connected components
    features[f'connectivity_cc'] = input_image_lcc.max()
    
    if features[f'connectivity_cc'] > 0:
    
        # Find largest connected component (lcc)
        counts = np.bincount(input_image_lcc.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])

        input_image_lcc[input_image_lcc!=lcc] = 0
        input_image_lcc[input_image_lcc==lcc] = 1
        input_image_lcc = input_image_lcc.astype(np.uint8)

        # Basic features
        for img, suffix in zip([input_image,input_image_lcc],['','_lcc']):

            z, y, x = np.where(img)

            features[f'shape_volume{suffix}'] = img.sum()
            features[f'position_depth{suffix}'] = 1 + np.ptp(z)
            for uname, u in zip(['x','y','z'],[x,y,z]):
                features[f'position_{uname}_centroid{suffix}'] = u.mean()
            features[f'roundness_surface_area{suffix}'] = get_surface_area(img)

    else:
        # If no foreground pixels are found
        for img, suffix in zip([input_image,input_image_lcc],['','_lcc']):
            features[f'shape_volume{suffix}'] = np.nan
            features[f'position_depth{suffix}'] = np.nan
            for uname in ['x','y','z']:
                features[f'position_{uname}_centroid{suffix}'] = np.nan
            features[f'roundness_surface_area{suffix}'] = np.nan
        
    if not compute_shcoeffs:
        features = cast_features(features)
        return features
    
    # Spherical harmonics expansion
    angle = np.nan
    if input_reference_image is not None:

        # Get alignment angle based on the reference image. Variance
        # paper uses make_unique = False
        input_ref_image_aligned, angle = shtools.align_image_2d(
            image = input_reference_image,
            make_unique = False
        )
                
        # Rotate input image according the reference alignment angle
        input_image_lcc_aligned = shtools.apply_image_alignment_2d(
            image = input_image_lcc,
            angle = angle
        ).squeeze()
                
    else:
        
        input_image_lcc_aligned = input_image_lcc
        
    (coeffs, _), (_, _, _, transform) = shparam.get_shcoeffs(
        image = input_image_lcc_aligned,
        lmax = 16,
        sigma = 2,
        alignment_2d = False
    )
    
    if transform is not None:
        transform = {
            'shcoeffs_transform_xc_lcc': transform[0],
            'shcoeffs_transform_yc_lcc': transform[1],
            'shcoeffs_transform_zc_lcc': transform[2],
            'shcoeffs_transform_angle_lcc': angle,
        }
    else:
        transform = {
            'shcoeffs_transform_xc_lcc': np.nan,
            'shcoeffs_transform_yc_lcc': np.nan,
            'shcoeffs_transform_zc_lcc': np.nan,
            'shcoeffs_transform_angle_lcc': np.nan,
        }

    # Add suffix to identify coeffs have been calculated on the
    # largest connected component
    coeffs = dict(
        (f'{key}_lcc',value) for (key,value) in coeffs.items()
    )
        
    features.update(coeffs)
    features.update(transform)

    for key, value in features.items():
        features = cast_features(features)
    
    return features


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
    ch_str = channels.index('structucture')
    
    raw = AICSImage(path_raw).data.squeeze()

    return raw[ch_dna], raw[ch_mem], raw[ch_str]


def load_images_and_calculate_features(path_seg, channels, path_output):
    
    """
    Load segmentation images, compute features and saves features
    as a JSON file.

    Parameters
    --------------------
    path_seg: Path
        Path to the 4D binary image.
    channels: list of str
        Name of channels of the 4D raw image.
    path_outout: Path
        Path to save the features.

    """
    
    # Find the correct segmentation for nucleus,
    # cell and structure
    seg_dna, seg_mem, seg_str = get_segmentations(
        path_seg=path_seg,
        channels=channels
    )

    # Compute nuclear features
    features_dna = get_features(
        input_image=seg_dna, input_reference_image=seg_mem
    )

    # Compute cell features
    features_mem = get_features(
        input_image=seg_mem, input_reference_image=seg_mem
    )

    # Compute structure features
    features_str = get_features(
        input_image=seg_str, input_reference_image=None, compute_shcoeffs=False
    )

    # Append prefix to features names
    features_dna = dict(
        (f"dna_{key}", value) for (key, value) in features_dna.items()
    )
    features_mem = dict(
        (f"mem_{key}", value) for (key, value) in features_mem.items()
    )
    features_str = dict(
        (f"str_{key}", value) for (key, value) in features_str.items()
    )

    # Concatenate all features for this cell
    features = features_dna.copy()
    features.update(features_mem)
    features.update(features_str)

    # Save to JSON
    with open(path_output, "w") as f:
        json.dump(features, f)

    return


if __name__ == "__main__":
    
    """
    You can also run feature calculation from a config file.
    """
    
    parser = argparse.ArgumentParser(description='Manage parallel cytoplasm parameterization')
    parser.add_argument('--config', help='Path to the JSON config file.', required=True)
    args = vars(parser.parse_args())

    with open(args['config'], 'r') as f:
        config = json.load(f)
    
    load_images_and_calculate_features(
        path_seg=config['path'],
        channels=config['channels'],
        path_output=config['output']
    )
