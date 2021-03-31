import os
import sys
import yaml
import json
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from aicsimageio import AICSImage
from contextlib import contextmanager
        
class Controller:
    """
    Functionalities for communicating with the config
    file.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    def __init__(self, config):
        self.config = config
        self.config['log'] = {}
        self.set_abs_path_to_local_staging_folder(config['project']['local_staging'])
        self.data_seg_section = self.config['data']['segmentation']
        self.features_section = self.config['features']
        self.space_section = self.config['shapespace']
        self.distribute_section = self.config['distribute']

    def set_abs_path_to_local_staging_folder(self, path):
        self.abs_path_local_staging = Path(path)
    def get_abs_path_to_local_staging_folder(self):
        return self.abs_path_local_staging
    def get_staging(self):# shortcut
        return self.get_abs_path_to_local_staging_folder()
    def overwrite(self):
        return self.config['project']['overwrite']

    def get_data_seg_names(self):
        return [k for k in self.data_seg_section.keys()]
    def get_data_seg_aliases(self):
        return [v['alias'] for k, v in self.data_seg_section.items()]
    def get_data_seg_channels(self):
        return [v['channel'] for k, v in self.data_seg_section.items()]
    def get_data_seg_name_alias_dict(self):
        return self.data_seg_section
    def get_data_seg_alias_channel_dict(self):
        aliases = self.get_data_seg_aliases()
        channels = self.get_data_seg_channels()
        return dict(zip(aliases, channels))
    def iter_data_seg_aliases(self):
        for alias in self.get_data_seg_aliases():
            yield alias

    def remove_mitotics(self):
        return self.config['preprocessing']['remove_mitotics']
    def remove_outliers(self):
        return self.config['preprocessing']['remove_outliers']

    def run_alignment(self):
        return self.features_section['alignment']['align']
    def make_alignment_unique(self):
        return self.features_section['alignment']['unique']
    def get_alignment_reference_name(self):
        return self.features_section['alignment']['reference']
    def get_alignment_reference_alias(self):
        name = self.get_alignment_reference_name()
        return self.get_data_seg_name_alias_dict()[name]['alias']
    def get_alignment_reference_channel(self):
        name = self.get_alignment_reference_name()
        return self.get_data_seg_name_alias_dict()[name]['channel']
    def should_calculate_shcoeffs(self, alias):
        return alias in self.features_section['SHE']['aliases']
    def get_lmax(self):
        return self.features_section['SHE']['lmax']
    def get_sigma(self):
        return self.features_section['SHE']['sigma']

    def get_aliases_for_pca(self):
        return self.space_section['aliases']
    def get_alias_for_sorting_pcs(self):
        return self.space_section['sorter']
    def get_removal_pct(self):
        return self.space_section['removal_pct']
    def get_number_of_shape_modes(self):
        return self.space_section['number_of_shape_modes']
    def get_map_points(self):
        return self.space_section['map_points']
    def get_number_of_map_points(self):
        return len(self.get_map_points())
    def get_plot_limits(self):
        return self.space_section['plot']['limits']
    def swapxy_on_zproj(self):
        return self.space_section['plot']['swapxy_on_zproj']
    def iter_map_points_index(self):
        for index in range(1, 1+self.get_number_of_map_points()):
            yield index
    def iter_map_points(self):
        for m in self.get_map_points():
            yield m
    def iter_shape_modes(self):
        p = "_".join(self.get_aliases_for_pca())
        for s in range(1, 1+self.get_number_of_shape_modes()):
            yield f"{p}_PC{s}"

    def get_available_parameterized_intensities(self):
        return [k for k in self.config['parameterization']['intensities'].keys()]

    # Misc
    def log(self, info):
        if not isinstance(info, dict):
            raise ValueError("Only dict can be logged.")
        for k, v in info.items():
            self.config["log"].setdefault(k, []).append(v)

    @staticmethod
    def get_ncores():
        return len(os.sched_getaffinity(0))
    def get_distributed_python_env_as_str(self):
        path = Path(self.distribute_section['python_env'])/"bin/python"
        return str(path)
    def get_distributed_cores(self):
        return self.distribute_section['cores']
    def get_distributed_number_of_workers(self):
        return self.distribute_section['number_of_workers']
    def get_distributed_queue(self):
        return self.distribute_section['queue']
    def get_distributed_walltime(self):
        return self.distribute_section['walltime']
    def get_distributed_memory(self):
        return self.distribute_section['memory']
        