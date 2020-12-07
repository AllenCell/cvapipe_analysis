# cvapipe_analysis

Analysis of data produced by cvapipe for the variance paper

---

## Installation

First, create a conda environment

```
conda create --name cvapipe_analysis_conda_env python=3.7
conda activate cvapipe_analysis_conda_env
```

then

```
pip install -e .
```

## Running the workflow

This analysis is currently not configured to run as a workflow. Please run steps indivudually. For example, to download the single cell image dataset manifest including raw GFP and segmented cropped images, run

```
cvapipe_analysis loaddata run
```

To commpute shapemodes on the downloaded dataset, run

```
cvapipe_analysis shapemode run
```

To compute multi resolution structure correlations for stereotypy, prepare a csv containing paths to pairs of morphed cell images of the same structure corresponding to specific bins and shapemodes (to be added). Then run

```
cvapipe_analysis multiresstructcompare run --input_csv_loc "/path/to/csv"
```

To compute concordance, prepare a 5d image stack that represents an average morphed cell (to be added). Then run

```
cvapipe_analysis multiresstructcompare run --input_5d_stack "/path/to/tif"
```

***Free software: Allen Institute Software License***

