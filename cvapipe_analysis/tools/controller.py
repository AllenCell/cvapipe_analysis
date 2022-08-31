import yaml
import numpy as np
import multiprocessing
from pathlib import Path
from aicsimageio import AICSImage


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
        self.data_section = self.config['data']
        self.features_section = self.config['features']
        self.space_section = self.config['shapespace']
        self.distribute_section = self.config['distribute']
        self.param_section = self.config['parameterization']

    def set_abs_path_to_local_staging_folder(self, path):
        self.abs_path_local_staging = Path(path)

    def get_abs_path_to_local_staging_folder(self):
        return self.abs_path_local_staging

    def get_staging(self):  # shortcut
        return self.get_abs_path_to_local_staging_folder()

    def overwrite(self):
        return self.config['project']['overwrite']

    def get_data_names(self):
        return [k for k in self.data_section.keys()]

    def get_data_aliases(self):
        return [v['alias'] for k, v in self.data_section.items()]

    def get_data_channels(self):
        return [v['channel'] for k, v in self.data_section.items()]

    def get_data_name_alias_dict(self):
        return self.data_section

    def get_data_alias_channel_dict(self):
        aliases = self.get_data_aliases()
        channels = self.get_data_channels()
        return dict(zip(aliases, channels))

    def get_data_alias_name_dict(self):
        named_aliases = self.get_data_name_alias_dict()
        return dict([(v['alias'], k) for k, v in named_aliases.items()])

    def get_name_from_alias(self, alias):
        return self.get_data_alias_name_dict()[alias]

    def get_channel_from_alias(self, alias):
        return self.get_data_alias_channel_dict()[alias]

    def get_name_from_channel(self, channel):
        return self.get_data_alias_name_dict()[channel]

    def get_color_from_alias(self, alias):
        return self.data_section[self.get_name_from_alias(alias)]['color']

    def get_alias_from_channel(self, channel):
        return self.data_section[self.get_name_from_channel(channel)]['alias']

    def remove_mitotics(self):
        return self.config['preprocessing']['remove_mitotics']

    def remove_outliers(self):
        return self.config['preprocessing']['remove_outliers']

    def is_filtering_on(self):
        return self.config["preprocessing"]["filtering"]["filter"]

    def get_filtering_csv(self):
        return self.config["preprocessing"]["filtering"]["csv"]

    def get_filtering_specs(self):
        return self.config["preprocessing"]["filtering"]["specs"]

    def get_aliases_for_feature_extraction(self):
        return self.features_section['aliases']

    def should_align(self):
        return self.features_section['SHE']['alignment']['align']

    def make_alignment_unique(self):
        return self.features_section['SHE']['alignment']['unique']

    def get_alignment_reference_name(self):
        return self.features_section['SHE']['alignment']['reference']

    def get_alignment_reference_alias(self):
        name = self.get_alignment_reference_name()
        if name:
            return self.get_data_name_alias_dict()[name]['alias']
        return None

    def get_alignment_reference_channel(self):
        name = self.get_alignment_reference_name()
        return self.get_data_name_alias_dict()[name]['channel']

    def get_alignment_moving_aliases(self):
        ref = self.get_alignment_reference_alias()
        aliases = self.get_aliases_with_shcoeffs_available()
        return [a for a in aliases if a != ref]

    def get_aliases_with_shcoeffs_available(self):
        return self.features_section['SHE']['aliases']

    def should_calculate_shcoeffs(self, alias):
        return alias in self.get_aliases_with_shcoeffs_available()

    def should_calculate_intensity_features(self, alias):
        if "intensity" in self.features_section:
            return alias in [k for k in self.features_section['intensity'].keys()]
        return False

    def get_mask_alias(self, alias):
        return [v for (k, v) in self.features_section['intensity'].items() if k==alias][0]

    def get_lmax(self):
        return self.features_section['SHE']['lmax']

    def get_sigma(self, alias):
        return self.features_section['SHE']['sigma'][alias]

    def get_aliases_for_pca(self):
        return self.space_section['aliases']

    def get_features_for_pca(self, df):
        prefixes = [f"{alias}_shcoeffs_L" for alias in self.get_aliases_for_pca()]
        return [f for f in df.columns if any(w in f for w in prefixes)]

    def get_shape_modes_prefix(self):
        return "_".join(self.get_aliases_for_pca())

    def get_alias_for_sorting_pca_axes(self):
        return self.space_section['sorter']

    def get_features_for_sorting_pca_axes(self):
        ranker = self.get_alias_for_sorting_pca_axes()
        return f"{ranker}_shape_volume"

    def get_removal_pct(self):
        return self.space_section['removal_pct']

    def get_number_of_shape_modes(self):
        return self.space_section['number_of_shape_modes']

    def get_shape_modes(self):
        p = self.get_shape_modes_prefix()
        return [f"{p}_PC{s}" for s in range(1, 1 + self.get_number_of_shape_modes())]

    def get_map_points(self):
        return self.space_section['map_points']

    def get_map_point_indexes(self):
        return np.arange(1, 1 + self.get_number_of_map_points())

    def get_number_of_map_points(self):
        return len(self.get_map_points())

    def get_center_map_point_index(self):
        return int(0.5 * (self.get_number_of_map_points() + 1))

    def get_extreme_opposite_map_point_indexes(self, off=0):
        mpIds = self.get_map_point_indexes()
        return [mpIds[0 + off], mpIds[-1 - off]]

    def get_plot_limits(self):
        return self.space_section['plot']['limits']

    def get_plot_frame(self):
        return self.space_section['plot']['frame']

    def swapxy_on_zproj(self):
        return self.space_section['plot']['swapxy_on_zproj']

    def iter_map_point_indexes(self):
        for index in range(1, 1 + self.get_number_of_map_points()):
            yield index

    def iter_map_points(self):
        for m in self.get_map_points():
            yield m

    def iter_shape_modes(self):
        for s in self.get_shape_modes():
            yield s

    def get_inner_most_alias_to_parameterize(self):
        return self.param_section['inner']

    def get_outer_most_alias_to_parameterize(self):
        return self.param_section['outer']

    def get_aliases_to_parameterize(self):
        return self.param_section['parameterize']

    def get_number_of_interpolating_points(self):
        return self.param_section['number_of_interpolating_points']

    def get_variables_values_for_aggregation(self, include_genes=True):
        variables = {}
        variables['shape_mode'] = self.get_shape_modes()
        variables['mpId'] = self.get_map_point_indexes()
        variables['aggtype'] = self.config['aggregation']['type']
        variables['alias'] = self.param_section['parameterize']
        if include_genes:
            structs = self.config['structures']
            variables['structure'] = [k for k in structs.keys()]
        return variables

    def duplicate_variable(self, variables, v):
        vals = variables.pop(v)
        variables[f"{v}1"] = vals
        variables[f"{v}2"] = vals
        return variables

    @staticmethod
    def get_filtered_dataframe(df, filters):
        for k, v in filters.items():
            values = v if isinstance(v, list) else [v]
            df = df.loc[df[k].isin(values)]
        return df

    # Misc
    def log(self, info):
        if not isinstance(info, dict):
            raise ValueError("Only dict can be logged.")
        for k, v in info.items():
            self.config["log"].setdefault(k, []).append(v)

    def get_gene_names(self):
        return [k for k in self.config['structures'].keys()]

    def get_structure_names(self):
        return [v[0] for k, v in self.config['structures'].items()]

    def get_structure_name(self, gene):
        return self.config["structures"][gene][0]

    def get_gene_color(self, gene):
        return self.config["structures"][gene][1]

    def get_optimal_seg_contrast(self, gene):
        return eval(self.config["structures"][gene][2])["seg"]

    def get_optimal_avgseg_contrast(self, gene):
        return eval(self.config["structures"][gene][2])["avgseg"]

    def get_optimal_raw_contrast(self, gene):
        return eval(self.config["structures"][gene][2])["raw"]

    @staticmethod
    def get_ncores():
        return multiprocessing.cpu_count()

    def get_distributed_python_env_as_str(self):
        path = Path(self.distribute_section['python_env']) / "bin/python"
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
