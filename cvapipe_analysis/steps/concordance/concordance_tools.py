import argparse
import concurrent
import pandas as pd

from cvapipe_analysis.tools import io, general, controller


class ConcordanceCalculator(io.DataProducer):
    """
    Provides the functionalities necessary for
    calculating the concordance of cells using
    their parameterized intensity representation.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, config):
        super().__init__(config)
        self.subfolder = 'concordance/values'

    def workflow(self):
        agg_rep1 = self.read_agg_parameterized_intensity(
            self.row.rename({'structure1': 'structure'}))
        agg_rep2 = self.read_agg_parameterized_intensity(
            self.row.rename({'structure2': 'structure'}))
        self.row['Pearson'] = self.correlate_representations(agg_rep1, agg_rep2)
        return

    def get_output_file_name(self):
        fname = self.get_prefix_from_row(self.row)
        return f"{fname}.csv"

    def save(self):
        save_as = self.get_output_file_path()
        pd.DataFrame([self.row]).to_csv(save_as, index=False)
        return save_as


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch single cell feature extraction.")
    parser.add_argument("--staging", help="Path to staging.", required=True)
    parser.add_argument("--csv", help="Path to the dataframe.", required=True)
    args = vars(parser.parse_args())

    config = general.load_config_file(args["staging"])
    control = controller.Controller(config)

    df = pd.read_csv(args['csv'], index_col=0)

    calculator = ConcordanceCalculator(control)
    with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
        executor.map(calculator.execute, [row for _, row in df.iterrows()])
