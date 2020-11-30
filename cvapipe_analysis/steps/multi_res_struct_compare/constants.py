#!/usr/bin/env python
# -*- coding: utf-8 -*-


class DatasetFieldsMorphed:
    CellId1 = "CellId_1"
    CellId2 = "CellId_2"
    StructureName1 = "structure_name_1"
    StructureName2 = "structure_name_2"
    Bin1 = "bin_1"
    Bin2 = "bin_2"
    SourceReadPath1 = "path_raw_morph_1"  # path_raw_morph_1
    SourceReadPath2 = "path_raw_morph_2"  # path_raw_morph_2


class DatasetFieldsIC:
    CellId = "CellId"
    CellIndex = "CellIndex"
    FOVId = "FOVId"
    StructureName1 = "GeneratedStructureName_i"
    StructureName2 = "GeneratedStructureName_j"
    GeneratedStructureInstance_i = "GeneratedStructureInstance_i"
    GeneratedStructureInstance_j = "GeneratedStructureInstance_j"
    SourceReadPath1 = "GeneratedStructuePath_i"
    SourceReadPath2 = "GeneratedStructuePath_j"
    SaveDir = "save_dir"
    SaveRegPath = "save_reg_path"


class DatasetFieldsAverageMorphed:
    NumStructs = 26
    NumBins = 9
    StructureIndex1 = "ChannelIndex_i"
    StructureIndex2 = "ChannelIndex_j"
    StructureName1 = "ChannelGeneName_i"
    StructureName2 = "ChannelGeneName_j"
    PC_bin = "PC_bin"


class StructureGenes:
    # This is the order that data is stored in the 5d stack
    CENT2 = "CENT2"
    TUBA1B = "TUBA1B"
    PXN = "PXN"
    TJP1 = "TJP1"
    LMNB1 = "LMNB1"
    NUP153 = "NUP153"
    ST6GAL1 = "ST6GAL1"
    LAMP1 = "LAMP1"
    ACTB = "ACTB"
    DSP = "DSP"
    FBL = "FBL"
    NPM1 = "NPM1"
    TOMM20 = "TOMM20"
    PMP34 = "PMP34"
    ACTN1 = "ACTN1"
    GJA1 = "GJA1"
    H2B = "H2B"
    SON = "SON"
    SEC61B = "SEC61B"
    RAB5A = "RAB5A"
    MYH10 = "MYH10"
    AAVS1 = "AAVS1"
    CTNNB1 = "CTNNB1"
    ATP2A2 = "ATP2A2"
    SMC1A = "SMC1A"
    CELL_NUC = "CELL+NUCLEUS"


class StructureGeneFullNames:
    # This is the order that data is stored in the 5d stack
    CENT2 = "Centrioles"
    TUBA1B = "Microtubules"
    PXN = "Matrix adhesions"
    TJP1 = "Tight junctions"
    LMNB1 = "Nuclear envelope"
    NUP153 = "Nuclear pores"
    Golgi = "Golgi"
    LAMP1 = "Lysosome"
    ACTB = "Filamentous actin"
    DSP = "Desmosomes"
    FBL = "Nucleolus (Dense Fibrillar Component)"
    NPM1 = "Nucleolus (Granular Component)"
    TOMM20 = "Mitochondria"
    PMP34 = "PMP34"
    ACTN1 = "Peroxisomes"
    GJA1 = "Gap junctions"
    H2B = "Histone"
    SON = "Nuclear Speckles"
    SEC61B = "Endoplasmic Reticulum"
    RAB5A = "Endosomes"
    MYH10 = "Actomyosin bundles"
    AAVS1 = "Plasma membrane"
    CTNNB1 = "Adherens junctions"
    ATP2A2 = "Sarcoplasmic reticulum"
    SMC1A = "Cohesin"
    CELL_NUC = "CELL+NUCLEUS"
