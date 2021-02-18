# cvapipe_analysis

## Analysis Pipeline for Cell Variance

Here you will find all the code necessary to i) reproduce the results shown in our paper [1] or ii) apply our methodology to you own dataset.

[1] - [Viana, Matheus P., et al. "Robust integrated intracellular organization of the human iPS cell: where, how much, and how variable?." bioRxiv (2020)](https://www.biorxiv.org/content/10.1101/2020.12.08.415562v1).

---

## Installation

First, create a conda environment for this project:

```
conda create --name cvapipe_analysis python=3.7
conda activate cvapipe_analysis
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

This step extract single-cell features, including cell, nuclear and intracellular volumes and spherical harmonics coefficients for cell and cnuclear shape. This step depends on step 1.

This step saves the features in the file `local_staging/computefeatures/manifest.raw`.

### 3. Compute shapemodes
```
cvapipe_analysis shapemode run
```

Preprocessing???? Here we compute cell and nuclear shape modes. Shape modesThis step depends on step 2.

### 4. Create single-cell parameterization
```
cvapipe_analysis parameterize run
```

This step depends on step 1.

### 5. Create 5D hypterstacks
```
cvapipe_analysis aggregation run
```

This step depends on steps 3 and 4.

## Running the pipeline on your own data

**TBD**

***Free software: Allen Institute Software License***

