#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from cvapipe_analysis.steps import MultiResStructCompare

EXPECTED_COLUMNS = [
    "Pearson Correlation",
    "Resolution (micrometers)",
]

from cvapipe_analysis.steps.multi_res_struct_compare.constants import (
    DatasetFieldsMorphed,
    DatasetFieldsAverageMorphed,
)


def test_morphed_cells_run(data_dir):

    multiresstructcompare = MultiResStructCompare()

    output = multiresstructcompare.run(
        input_csv_loc=Path(
            "/allen/aics/modeling/ritvik/projects/cvapipe/"
            + "FinalMorphedStereotypyDatasetPC1.csv"
        ),
        max_rows=10,
    )

    fig_plus_data_manifest = pd.read_csv(output)
    similarity_score_manifest = pd.read_csv(fig_plus_data_manifest["path"][0])

    # Check pearson corr and res columns
    assert all(
        expected_col in similarity_score_manifest.columns
        for expected_col in [
            *EXPECTED_COLUMNS,
            DatasetFieldsMorphed.StructureName1,
            DatasetFieldsMorphed.StructureName2,
            DatasetFieldsMorphed.Bin1,
            DatasetFieldsMorphed.Bin2,
            DatasetFieldsMorphed.CellId1,
            DatasetFieldsMorphed.CellId2,
            DatasetFieldsMorphed.SourceReadPath1,
            DatasetFieldsMorphed.SourceReadPath2,
        ]
    )


def test_avg_morphed_cell_run(data_dir):

    multiresstructcompare = MultiResStructCompare()
    output = multiresstructcompare.run(
        input_5d_stack=Path(
            "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/"
            + "assay-dev-cytoparam/avgcell/DNA_MEM_PC1_seg_avg.tif"
        ),
        max_rows=10,
    )

    fig_plus_data_manifest = pd.read_csv(output)
    similarity_score_manifest = pd.read_csv(fig_plus_data_manifest["path"][0])

    # Check expected columns
    assert all(
        expected_col in similarity_score_manifest.columns
        for expected_col in [
            *EXPECTED_COLUMNS,
            DatasetFieldsAverageMorphed.StructureIndex1,
            DatasetFieldsAverageMorphed.StructureIndex2,
            DatasetFieldsAverageMorphed.StructureName1,
            DatasetFieldsAverageMorphed.StructureName2,
            DatasetFieldsAverageMorphed.PC_bin,
        ]
    )
