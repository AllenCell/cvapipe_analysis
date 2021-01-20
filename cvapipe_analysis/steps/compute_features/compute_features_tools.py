import numpy as np
from aicsshparam import shtools
from skimage import measure as skmeasure
from skimage import morphology as skmorpho

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

def get_features(input_image):
    
    """
    Extracts single cell features used in the variance paper.

    Parameters
    --------------------
    input_image: ndarray
        3D input image representing the single cell segmentation that
        represents either dna, cell or structure.

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
    
    # Find largest connected component (lcc)
    counts = np.bincount(input_image_lcc.reshape(-1))
    lcc = 1 + np.argmax(counts[1:])

    input_image_lcc[input_image_lcc!=lcc] = 0
    input_image_lcc[input_image_lcc==lcc] = 1
    input_image_lcc = input_image_lcc.astype(np.uint8)

    # Basic features
    for img, suffix in zip([input_image,input_image_lcc],['','_lcc']):

        z, _, _ = np.where(img)
                
        features[f'shape_volume{suffix}'] = img.sum()
        features[f'position_depth{suffix}'] = 1 + np.ptp(z)
        features[f'roundness_surface_area{suffix}'] = get_surface_area(img)
    
    # Spherical harmonics expansion
    
    '''
    Implement spherical harmonics expansion using aics-shparam here.
    
    Code snippet used for the paper:
    --------------------------------
    
    if seg2 is not None:

        img_aligned, (angle, flip_x, flip_y) = shtools.align_image_2d(
            image = seg2,
            alignment_channel = None,
            preserve_chirality = chirality)

        seg = shtools.apply_image_alignment_2d(
            image = seg,
            angle = angle,
            flip_x = flip_x,
            flip_y = flip_y).squeeze()

        if _aicsfeature_debug_:
            print("Alignment done!")
            pdb.set_trace()

    (features, grid_rec), (image_, mesh, grid_down, transform) = shparam.get_shcoeffs(
        image = seg,
        lmax = lmax,
        sigma = sigma,
        alignment_2d = False)

    if transform is not None:
        transform = {
            'shcoeffs_transform_xc': transform[0],
            'shcoeffs_transform_yc': transform[1],
            'shcoeffs_transform_zc': transform[2],
            ## This need to be adjusted to handle the case when seg2 = None
            'shcoeffs_transform_angle': angle,
            'shcoeffs_transform_xflip': flip_x,
            'shcoeffs_transform_yflip': flip_y
        }
    else:
        transform = {
            'shcoeffs_transform_xc': np.nan,
            'shcoeffs_transform_yc': np.nan,
            'shcoeffs_transform_zc': np.nan,
            'shcoeffs_transform_angle': np.nan,
            'shcoeffs_transform_xflip': np.nan,
            'shcoeffs_transform_yflip': np.nan 
        }

    if _aicsfeature_debug_:
        print("Running on: {0}".format(_running_on_flag_))
        pdb.set_trace()

    features.update(transform)
    
    return features, (image_, mesh, grid_down, grid_rec)    
    '''
    
    return features