#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

# From Helpyr
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from helpyr_misc import print_entire_df
from logger import Logger
from data_loading import DataLoader


class QsGrapher:

    def __init__(self):
        # File locations
        self.root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
        self.pickle_source = f"{self.root_dir}/processed-pickles"
        self.log_filepath = "./log-files/Qs_grapher.txt"
        
        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("Qs Grapher")

        # Start up loader
        self.loader = DataLoader(self.pickle_source, logger=self.logger)

    def make_experiment_plots(self):
        self.logger.write(["Making experiment plots..."])

        self.experiments = {}

        indent_function = self.logger.run_indented_function

        indent_function(self.load_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        self.logger.end_output()

    def load_data(self):
        for experiment in self.experiments.values():
            experiment.sort_ranks()
            experiment.load_data(self.loader)


if __name__ == "__main__":
    # Run the script
    grapher = QsGrapher()
    grapher.make_experiment_plots()
