#!/usr/bin/env python3

# This script will load Qs# chunk pickles according to the metapickle and 
# combine them into complete Qs pickles
#
# Qs_metapickle = {period_path: Qs_pickle_paths}
#
# Qs# pickles are panda dataframes directly translated from the raw txt files

import os
import numpy as np
from time import asctime

# From Helpyr
import data_loading
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from logger import Logger

# Outline:
# 1) load Qs_metapickle
# 2) load Qs pickles for a period
# 3) error check raw qs dataframes
#   - conflict between qs and qs#?
# 4) combine qs dataframes
# 5) error check combined qs dataframe

error_codes = {
        0 : "Conflicting Qs files",
        }

class QsPickleProcessor:

    def __init__(self):
        # File locations
        self.root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
        self.pickle_source = f"{self.root_dir}/raw-pickles"
        self.pickle_destination = f"{self.root_dir}/processed-pickles"
        self.log_filepath = "./log-files/Qs_pickle_processor.txt"
        self.metapickle_name = "Qs_metapickle"

        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        ensure_dir_exists(self.pickle_destination, self.logger)
        self.logger.write_section_break()
        self.logger.write(["Begin pickle processor output", asctime()])

        # Start up loader
        self.loader = data_loading.DataLoader(self.pickle_source, 
                self.pickle_destination, self.logger)

    def run(self):
        self.logger.write(["Running pickle processor..."])
        indent_function = self.logger.run_indented_function

        # Load Qs_metapickle
        self.metapickle = self.loader.load_pickle(self.metapickle_name)

        for period_path in self.metapickle:
            # attribute data to be reset every period
            self.lingering_errors = [] # error for secondary check to look at
            self.Qs_path_list = [] # list of Qs#.txt file paths
            self.Qs0_data = None # data for Qs.txt
            self.Qsn_data = [] # data for Qs#.txt
            self.Qsn_names = [] # Names of Qs# files
            self.current_period_path = period_path # is also the metapickle key

            # Get meta info
            _, experiment, step, rperiod = nsplit(self.current_period_path, 3)
            period = rperiod[8:]
            msg = f"Processing {experiment} {step} {period}..."
            indent_function(self.process_period, before_msg=msg)

    def process_period(self):

        indent_function = self.logger.run_indented_function
        # Load data
        indent_function(self.load_data,
                        "Loading data...", "Finished loading data!")

        # Primary Error Checks
        indent_function(self.primary_error_check,
                        "Running primary error checks...",
                        "Finished primary error checks!")

        # Combining Qsn chunks
        indent_function(self.combine_Qsn_chunks,
                        "Combining Qs chunks...",
                        "Finished combining Qs chunks!")

        # Secondary Error Checks
        indent_function(self.secondary_error_check,
                        "Running secondary error checks...",
                        "Finished secondary error checks!")


    def load_data(self):
        # Load the sorted list of paths for this period
        self.Qs_path_list = self.metapickle[self.current_period_path]
        # Load the associated data
        Qs_period_data = self.loader.load_pickles(self.Qs_path_list, add_path=False)

        for Qs_path in self.Qs_path_list:
            pkl_name = nsplit(Qs_path, 1)[1]
            stripped_name = pkl_name.split('.')[0]
            Qs_name = stripped_name.split('_')[-1]
            bedload_data = Qs_period_data[Qs_path]

            if Qs_name == 'Qs':
                assert(Qs0_data is None)
                self.Qs0_data = bedload_data
            else:
                assert(Qs_name[2:].isdigit())
                self.Qsn_data.append(bedload_data)
                self.Qsn_names.append(Qs_name)

    def primary_error_check(self):
        # 3) error check raw qs dataframes
        #   - conflict between qs and qs#?

        if self.Qs0_data and self.Qsn_data:
            name_list = ', '.join(Qsn_names)
            error_msg = error_codes[0]
            self.logger.warning([error_msg,
                "Qs.txt and Qs#.txt files both exist",
                "Qs#.txt: {}"])
            self.lingering_errors.append(error_msg)

    def combine_Qsn_chunks(self):
        # 4) combine qs dataframes
        def get_zeros_with_meta():
            zeros = np.zeros_like(self.Qsn_data[0])
        overlap = get_zeros_with_meta()
        summed = get_zeros_with_meta()

        for chunk in self.Qsn_data:
            print(chunk)
            print(summed)
            raise NotImplementedError

    def secondary_error_check(self):
        # 5) error check combined qs dataframe
        pass

if __name__ == "__main__":
    # Run the script
    processor = QsPickleProcessor()
    processor.run()
