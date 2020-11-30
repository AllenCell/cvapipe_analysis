#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union, Optional
from pathlib import Path

import dask.dataframe as dd
import pandas as pd


class MissingDataError(Exception):
    def __init__(
        self, dataset: Union[pd.DataFrame, dd.DataFrame], missing_fields: List[str]
    ):
        # Run base exception init
        super().__init__()

        # Store params for display
        self.dataset = dataset
        self.missing_fields = missing_fields

    def __str__(self):
        return (
            f"Dataset provided does not have the required columns for this operation. "
            f"Missing fields: {self.missing_fields} "
        )


def check_required_fields(
    dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
    required_fields: List[str],
) -> Optional[MissingDataError]:
    # Handle dataset provided as string or path
    if isinstance(dataset, (str, Path)):
        dataset = Path(dataset).expanduser().resolve(strict=True)

        # Read dataset
        dataset = dd.read_csv(dataset)

    # Check that all columns provided as required are in the dataset
    missing_fields = set(required_fields) - set(dataset.columns)
    if len(missing_fields) > 0:
        raise MissingDataError(dataset, missing_fields)
