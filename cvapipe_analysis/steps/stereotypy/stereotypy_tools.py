import random
import argparse
import concurrent
import pandas as pd

from cvapipe_analysis.tools import io, general, controller


class StereotypyCalculator(io.DataProducer):
    """
    Provides the functionalities necessary for
    calculating the stereotypy of cells using their
    parameterized intensity representation.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, config):
        super().__init__(config)
        self.subfolder = 'stereotypy/values'

    def workflow(self):
        self.shuffle_target_cellids()
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            self.pcorrs = list(executor.map(self.correlate, self.iter_over_pairs()))
        return

    def get_output_file_name(self):
        fname = self.get_prefix_from_row(self.row)
        return f"{fname}.csv"

    def save(self):
        df = pd.DataFrame(
            list(zip(self.CellIds, self.CellIdsTarget, self.pcorrs)),
            columns=['CellId1', 'CellId2', 'Pearson']
        ).dropna(subset=["Pearson"])
        self.row.pop('CellIds')
        for k in self.row.keys():
            df[k] = self.row[k]
        save_as = self.get_output_file_path()
        df.to_csv(save_as, index=False)
        return save_as

    def shuffle_target_cellids(self):
        if len(self.CellIds) < 2:
            raise RuntimeError(f"Not enough cells to compute stereotypy.")
        self.CellIdsTarget = self.CellIds.copy()
        while True:
            random.shuffle(self.CellIdsTarget)
            for id1, id2 in zip(self.CellIds, self.CellIdsTarget):
                if id1 == id2:
                    break
            else:
                return

    def iter_over_pairs(self):
        for id1, id2 in zip(self.CellIds, self.CellIdsTarget):
            yield (id1, id2)

    def correlate(self, indexes):
        rep1, aliases = self.read_parameterized_intensity(indexes[0], True)
        rep2 = self.read_parameterized_intensity(indexes[1])
        rep1 = rep1[aliases.index(self.row.alias)]
        rep2 = rep2[aliases.index(self.row.alias)]
        return self.correlate_representations(rep1, rep2)

    @staticmethod
    def append_configs_from_stereotypy_result_file_name(df, filename):
        for name, value in zip(
            ['intensity', 'structure_name', 'shapemode', 'bin'], filename.split('-')
        ):
            df[name] = value if name != 'bin' else int(value[1])
        return df


if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description='Batch stereotypy calculation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col=0)

    calculator = StereotypyCalculator(control)
    for _, row in df.iterrows():
        '''Concurrent processes inside. Do not use concurrent here.'''
        calculator.execute(row)
