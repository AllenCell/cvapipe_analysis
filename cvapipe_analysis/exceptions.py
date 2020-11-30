#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

###############################################################################


class MissingDataError(Exception):
    def __init__(self, missing_fields: List[str]):
        # Run base exception init
        super().__init__()

        # Store params for display
        self.missing_fields = missing_fields

    def __str__(self):
        return (
            f"Dataset provided does not have the required columns for this operation. "
            f"Missing fields: {self.missing_fields}"
        )
