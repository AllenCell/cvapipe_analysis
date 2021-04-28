# cvapipe_analysis

## Analysis Pipeline for Cell Variance

![Shape modes](docs/logo.png)

---

## Installation

First, create a conda environment for this project:
```
conda create --name cvapipe python=3.8
conda activate cvapipe
```

then clone this repo
```
git clone https://github.com/AllenCell/cvapipe_analysis.git
```

and install it with
```
cd cvapipe_analysis
pip install -e .
```

## Types of usage

This package can be used to reproduce main results shown in [1] or to generate similar results using your own data. However, before applying to your dataset, we highly recommend you to first run it for reproducibility in our test dataset to understand how the package works.

[1] - [Viana, Matheus P., et al. "Robust integrated intracellular organization of the human iPS cell: where, how much, and how variable?." bioRxiv (2020)](https://www.biorxiv.org/content/10.1101/2020.12.08.415562v1).


## The YAML configuration file

This package is fully configured through the file `config.yaml`. This file is divided into sections that more or less has a one-to-one mapping to existing workflow steps. Here are the main things you need to know about the configuration file:

**Project**
```
appName: cvapipe_analysis
project:
    # Sufix to append to local_staging
    local_staging: "path_to_your/local_staging"
    overwrite: on
```

Set the full path where you want data and results to be stored in `local_staging`.

**Data**
```
data:
    nucleus:
        channel: "dna_segmentation"
        alias: "NUC"
        color: "#3AADA7"
    cell:
        channel: "membrane_segmentation"
        alias: "MEM"
        color: "#F200FF"
    structure:
        channel: "struct_segmentation_roof"
        alias: "STR"
        color: "#000000"
    structure-raw:
        channel: "structure"
        alias: "STRRAW"
        color: "#000000"
```

Here we provide a description of the data. Aliases must be unique and they are used in the rest of the configuration file to specify which data we are referring to. In case you are using this package on your own data, be aware that the values used in the field `channel` must be found in the column `name_dict`of your input manifets file (see the section "Running the pipeline on your own data").

**Features**
```
features:
    aliases: ["NUC", "MEM", "STR"]
    # SHE - Spherical harmonics expansion
    SHE:
        alignment:
            align: on
            unique: off
            reference: "cell"
        aliases: ["NUC", "MEM"]
        # Size of Gaussian kernal used to smooth the
        # images before SHE coefficients calculation
        sigma: 2
        # Number of SHE coefficients used to describe cell
        # and nuclear shape
        lmax: 16
```
This section is used to specify which aliases we should compute features on. In addition, which aliases we should calculate the spherical harmonics coefficies on and which type of alignment should be used.

**Pre-processing**
```
preprocessing:
    remove_mitotics: on
    remove_outliers: on
```

Here we set whether or not to remove mitotic cells or outlier from the dataset. You can turn this off when running `cvapipe_analysis` on your own data.

**Shape Space**
```
shapespace:
    # Specify the a set of aliases here
    aliases: ["NUC", "MEM"]
    # Sort shape modes by volume of
    sorter: "MEM"
    # Percentage of exteme points to be removed
    removal_pct: 1.0
    # Number of principal components to be calculated
    number_of_shape_modes: 8
    # Map points
    map_points: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    plot:
        swapxy_on_zproj: off
        # limits of x and y axies in the animated GIFs
        limits: [-150, 150, -80, 80]
```

Here we specify which aliases should be used to create a shape space. This must be a subset of the aliases specified above to have their spherical harmonics coefficients computed. In case os small datasets with only hundreds of cells, you may want to reduce the number of map points of your shape soace. The number of map points must be odd.

**Intensity Parameterization**
```
parameterization:
    inner: "NUC"
    outer: "MEM"
    parameterize: ["RAWSTR", "STR"]
    number_of_interpolating_points: 32
```

First we specify which alias should be used as internal and external references and the aliases that we obtain parameterization for.

**Structures**
```
structures:
    "FBL": ["nucleoli [DFC)", "#A9D1E5"]
    "NPM1": ["nucleoli [GC)", "#88D1E5"]
    "SON": ["nuclear speckles", "#3292C9"]
    "SMC1A": ["cohesins", "#306598"]
    "HIST1H2BJ": ["histones", "#305098"]
    "LMNB1": ["nuclear envelope", "#084AE7"]
    "NUP153": ["nuclear pores", "#0840E7"]
    "SEC61B": ["ER [Sec61 beta)", "#FFFFB5"]
    "ATP2A2": ["ER [SERCA2)", "#FFFFA0"]
    "TOMM20": ["mitochondria", "#FFD184"]
    "SLC25A17": ["peroxisomes", "#FFD16E"]
    "RAB5A": ["endosomes", "#FFD150"]
    "LAMP1": ["lysosomes", "#AD952A"]
    "ST6GAL1": ["Golgi", "#B7952A"]
    "CETN2": ["centrioles", "#9D7000"]
    "TUBA1B": ["microtubules", "#6B4500"]
    "AAVS1": ["plasma membrane", "#FFD2FF"]
    "ACTB": ["actin filaments", "#FFB1FF"]
    "ACTN1": ["actin bundles", "#FF96FF"]
    "MYH10": ["actomyosin bundles", "#FF82FF"]
    "GJA1": ["gap junctions", "#A850C0"]
    "TJP1": ["tight junctions", "#A850D4"]
    "DSP": ["desmosomes", "#A850E8"]
    "CTNNB1": ["adherens junctions", "#CB1CCC"]
    "PXN": ["matrix adhesions", "#77207C"]
```

Here we specify a dictionary with the gene names, description and color for each structure. Again, in case you are applying to your own data, make sure you specify here the values you use in the column `structure_name` of your manifest file (see the section "Running the pipeline on your own data").

## Running the pipeline to reproduce the paper

This analysis is currently not configured to run as a workflow. Please run steps individually.

### 1. Download the single-cell image dataset manifest including raw GFP and segmented cropped images
```
cvapipe_analysis loaddata run
```

This command downloads the whole dataset of ~7Tb. For each cell in the dataset, we provide a raw 3-channels image containing fiducial markers for cell membrane and nucleus, toghether with a FP marker for one intracellular structure. We also provide segmentations for each cell in the format of 5-channels binary images. The extra two channels corresponds to roof-augmented versions of cell and intracellular structures segmentations. For more information about this, please refer to our paper [1]. Metadata about each cell can be found in the file `manifest.csv`. This is a table where each row corresponds to a cell.

**Importantly**, you can download a *small test dataset composed by 300 cells chosen at random* from the main dataset. To do so, please run
```
cvapipe_analysis loaddata run --test
```

This step saves the single-cell images in the folders `local_staging/loaddata/crop_raw` and `local_staging/loaddata/crop_seg`.

### 2. Compute single-cell features
```
cvapipe_analysis computefeatures run
```

This step extract single-cell features, including cell, nuclear and intracellular volumes and other basic features. Here we also use `aics-shparam` [(link)](https://github.com/AllenCell/aics-shparam) to compute the spherical harmonics coefficients for cell and nuclear shape. This step depends on step 1.

This step saves the features in the file `local_staging/computefeatures/manifest.csv`.

### 3. Pre-processing dataset
```
cvapipe_analysis preprocessing run
```

This step removes outliers and mitotic cells from the single cell dataset. This step saves results in the file `local_staging/preprocessing/manifest.csv` and

**Folder: `local_staging/shapemode/outliers/`**

- `xx.png`: Diagnostic plots for outlier detection.

### 4. Compute shapemodes
```
cvapipe_analysis shapemode run
```

Here we implement a few pre-processing steps. First, all mitotic cells are removed from the dataset. Next we use a feature-based outlier detection to detect and remove outliers form the dataset. The remaining dataset is used as input for principal component analysis. Finally, we compute cell and nuclear shape modes. This step depends on step 2.

A couple of output files are produced on this step:

**Folder: `local_staging/shapemode/pca/`**

- `explained_variance.png`: Explained variance by each principal component.
- `feature_importance.txt`: Importance of first few features of each principal component.

**Folder: `local_staging/shapemode/avgcell/`**

- `xx.vtk`: vtkPolyData files corresponding to 3D cell and nuclear meshes. We recommend [Paraview](https://www.paraview.org) to open this files.
- `xx.gif`: Animated GIF illustrating cell and nuclear shape modes from 3 different projections.
- `combined.tif`: Multichannel TIF that combines all animated GIFs in the same image.

### 5. Create parameterized intensity representation
```
cvapipe_analysis parameterization run
```

Here we use `aics-cytoparam` [(link)](https://github.com/AllenCell/aics-cytoparam) to create parameterization for all the single-cell data. This steps depends on step 2.

**Folder: `local_staging/parameterization/representations/`**

- `xx.tif`: Multichannels TIFF image with the cell representation.

### 6. Create aggregated parameterized intensity representations
```
cvapipe_analysis aggregation run
```

This step generates aggregation of multiple cells representations and morph them into idealized shapes from the shape space. This step depends on steps 3 and 4.

**Folder: `local_staging/aggregation/`**

- `manifest.csv`: Manifest with combinations of parameters used for aggregation and path to TIF file generated.

**Folder: `local_staging/aggregation/repsagg/`**

- `avg-SEG-TUBA1B-DNA_MEM_PC4-B5-CODE.tif`: Example of file generated. This represents the average parameterized intensity representation generated from segmented images of all TUBA1B cells that fall into bin number 5 from shape mode 4.

**Folder: `local_staging/aggregation/aggmorph/`**

- `avg-SEG-TUBA1B-DNA_MEM_PC4-B5.tif`: Same as above but the representation has been morphed into the cell shape corresponding to bin number 5 of shape mode 4.

### 7. Stereotypy analysis
```
cvapipe_analysis stereotypy run
```

This calculates the extent to which a structure’s individual location varied. This step depends on step 4.

**Folder: `local_staging/stereotypy/values`**

- `*.csv*`: Stereotypy values.

**Folder: `local_staging/stereotypy/plots`**

- Resulting plots.

### 8. Concordance analysis
```
cvapipe_analysis concordance run
```

This calculates the extent to which the structure localized relative to all the other cellular structures. This step depends on step 5.

**Folder: `local_staging/concordance/values/`**

- `*.csv*`: Concordance values

**Folder: `local_staging/concordance/plots/`**

- Resulting plots.

## Running the pipeline on your own data

You need to specify the format of your data using a `manifest.csv` file. Each row of this file corresponds to a cell in your dataset. This file is requred to have the following columns:

`CellId`: Unique ID of the cell. Example: `AB98765`.

`structure_name`: FP structure tagged in the cell. Add something like "NA" if you don't have anything tagged for the cell. Example: `TOMM20`.

`crop_seg`: Full path to the multichannel single cell segmentation.

`crop_raw`: Full path to the multichannel single cell raw image.

`name_dict`: Dictionary that specifies the names of each channel in the two images above. Example: `"{'crop_raw': ['dna_dye', 'membrane', 'gfp'], 'crop_seg': ['dna_seg', 'cell_seg', 'gfp_seg', 'gfp_seg2']}"`. In this case, your `crop_raw` images must have 3 channels once this is the number of names you provide in `name_dict`. Similarly, `crop_seg` must have 4 channels in this example.

You are ready to start using `cvapipe_analysis` once you have this manifest file created. To do so, you should run the step `loaddata` with the additional flag `--csv path_to_manifest`, where `path_to_manifest` is the full path to the manifest file that you juest created:

`cvapipe_analysis loaddata run --csv path_to_manifest`

All the other steps can be ran without modifications.

## Running the pipeline on a cluster with `sbatch` capabilities

If you are running `cvapipe_analysis` on a Slurm cluster or any other cluster with `sbatch` capabilities, each step can be called with a flag `--distribute`. This will spawn many jobs to run in parallel in the cluster. Specific parameters can be set in the `resources` section of the YAML config file.

***Free software: Allen Institute Software License***

