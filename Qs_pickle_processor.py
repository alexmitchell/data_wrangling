#!/usr/bin/env python3

# This script will load Qs# chunk pickles according to the metapickle and 
# combine them into complete Qs pickles
#
# Qs_metapickle = {period_path: Qs_pickle_paths}
#
# Qs# pickles are panda dataframes directly translated from the raw txt files

import os
import numpy as np
import pandas as pd
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

def print_entire_df(df):
    # Print all rows in a dataframe
    with pd.option_context('display.max_rows', None):
        print(df)


class QsPickleProcessor:

    error_codes = {
            'CQF' : "Conflicting Qs files",
            'NDF' : "No Data Found",
            'MMD' : "Mismatched Data",
            }

    def __init__(self):
        # File locations
        self.root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
        self.pickle_source = f"{self.root_dir}/raw-pickles"
        self.pickle_destination = f"{self.root_dir}/processed-pickles"
        self.log_filepath = "./log-files/Qs_pickle_processor.txt"
        self.metapickle_name = "Qs_metapickle"
        
        # tolerance for difference between files
        self.difference_tolerance = 0.02

        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        ensure_dir_exists(self.pickle_destination, self.logger)
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
            self.combined_Qs = None
            self.accumulating_overlap = None

            # Get meta info
            _, experiment, step, rperiod = nsplit(self.current_period_path, 3)
            period = rperiod[8:]
            msg = f"Processing {experiment} {step} {period}..."
            self.pkl_name = '-'.join(['Qs', experiment, step, period])

            indent_function(self.process_period, before_msg=msg)
        self.logger.end_output()


    def process_period(self):

        if self.loader.is_pickled(self.pkl_name):
            self.logger.write(["Nothing to do"])
            return

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

        # Write to pickle
        indent_function(self.produce_processed_pickle,
                        "Producing processed pickles...",
                        "Processed pickles produced!")

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
                assert(self.Qs0_data is None)
                self.Qs0_data = bedload_data
            else:
                assert(Qs_name[2:].isdigit())
                self.Qsn_data.append(bedload_data)
                self.Qsn_names.append(Qs_name)

    def primary_error_check(self):
        # 3) error check raw qs dataframes
        #   - conflict between qs and qs#?

        if self.Qs0_data is not None and self.Qsn_data:
            name_list = ', '.join(self.Qsn_names)
            error_msg = QsPickleProcessor.error_codes['CQF']
            self.logger.warning([error_msg,
                "Qs.txt and Qs#.txt files both exist",
               f"Qs#.txt: {name_list}"])
            self.lingering_errors.append(error_msg)

    def combine_Qsn_chunks(self):
        # 4) combine qs dataframes
        # The data is split up into multiple chunks (each Qs# file is a chunk).  
        # This functions assembles them into a complete Qs dataframe.  
        # Overlapping rows are converted to nan because I can't see any way to 
        # choose which one to keep.
        if not self.Qsn_data:
            self.logger.write("No chunks to combine.")
            return

        combined = self._make_like_df(self.Qsn_data[0], ['timestamp'])
        accumulating_overlap = None

        exclude_cols = ['timestamp', 'missing ratio', 'vel', 'sd vel', 'number vel']
        target_cols = [col for col in combined.columns.values
                           if col not in exclude_cols]
        #bedload_columns = [   # Columns containing bedload data
        #        'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
        #        'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
        #        'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
        #        'Bedload 22', 'Bedload 32', 'Bedload 45',
        #        ]

        # Set up a few lambda functions
        get_num = lambda s: int(s[2:]) # get the file number from the name
        #
        #get_bedload_subset = lambda c: c.loc[:, bedload_columns]
        get_target_subset = lambda c: c.loc[:, target_cols]
        #
        # Find rows with data. Should remove meta columns beforehand
        # Will select rows with non-null values (selects zero rows
        #find_data_rows = lambda df: ((df != 0) & df.notnull()).any(axis=1)
        find_data_rows = lambda df: df.notnull().all(axis=1)

        for raw_chunk, name in zip(self.Qsn_data, self.Qsn_names):
            # Each raw dataframe contains only a chunk of the overall data.  
            # However they contain zero values for all times outside of the 
            # valid chunk time. Some chunks overlap too. 
            ch_num, max_num = get_num(name), get_num(self.Qsn_names[-1])
            self.logger.write(f"Processing chunk {ch_num} of {max_num}")

            # Get bedload subsets
            bedload_chunk = get_target_subset(raw_chunk)
            bedload_combined = get_target_subset(combined)

            # Find rows with data
            chunk_rows = find_data_rows(bedload_chunk)
            combined_rows = find_data_rows(bedload_combined)

            # Find overlap
            overlap_rows = chunk_rows & combined_rows
            #overlap_A = raw_chunk[overlap_rows]
            #overlap_B = combined[overlap_rows]

            # Add chunk to combined array
            combined.loc[chunk_rows, 1:] = raw_chunk[chunk_rows]
            combined.loc[overlap_rows, 1:] = np.nan

            # Keep track of overlap rows
            if accumulating_overlap is None:
                accumulating_overlap = overlap_rows
            else:
                accumulating_overlap = accumulating_overlap | overlap_rows

        self.combined_Qs = combined
        self.accumulating_overlap = accumulating_overlap

    def _make_like_df(self, like_df, columns_to_copy=[], fill_val=np.nan):
        # Make a dataframe like the Qs data with a few columns copied and the 
        # rest filled with a default value

        np_like = np.empty_like(like_df.values)
        np_like.fill(fill_val)
        pd_like = pd.DataFrame(np_like,
                columns=like_df.columns, index=like_df.index)

        for column in columns_to_copy:
            pd_like.loc[:, column] = like_df.loc[:, column]
        return pd_like

    def secondary_error_check(self):
        # 5) error check combined qs dataframe
        
        self.final_output = None

        # Check for diff between raw_Qs and Qs_combined
        raw_Qs = self.Qs0_data
        raw_exists = raw_Qs is not None
        combined_Qs = self.combined_Qs
        combined_exists = combined_Qs is not None
        if raw_exists and combined_exists:
            self._difference_check()
        elif not(raw_exists or combined_exists):
            error_msg = QsPickleProcessor.error_codes['NDF']
            self.logger.warning([error_msg,
                "Both the raw Qs pickle and combined Qs df are missing."])
        else:
            using = "raw Qs" if raw_exists else "combined Qs"
            self.final_output = raw_Qs if raw_exists else combined_Qs
            self.logger.write(f"Only {using} found." +
                              "No difference check needed.")

        # Set rows with any Nan values to entirely Nan values
        nan_rows = self.final_output.isnull().any(axis=1)
        self.final_output.loc[nan_rows, 1:] = np.nan

        # Check for accumulated overlap
        overlap = self.accumulating_overlap
        if combined_exists and overlap.any():
            overlap_times = self.combined_Qs.loc[overlap,'timestamp']
            str_overlap_times = overlap_times.to_string(float_format="%f")

            self.logger.write(["The following timestamps were overlapped: "])
            self.logger.write(str_overlap_times.split('\n'), local_indent=1)

    def _difference_check(self):
        # Look at the difference between the Qs.txt and Qs combined data.
        raw_Qs = self.Qs0_data
        combined_Qs = self.combined_Qs

        # Element-wise bool difference between dataframes
        Qs_diff = (combined_Qs != raw_Qs)
        # Rows that have Nan values in both dataframes will be thrown out and 
        # should not count towards the difference.
        # Rows that started with a value and ended with Nan should count. (such 
        # as overlap rows)
        Qs_both_nan = combined_Qs.isnull() & raw_Qs.isnull()
        both_nan_rows = Qs_both_nan.any(axis=1)
        Qs_diff.loc[both_nan_rows, :] = False
        #Qs_either_nan = combined_Qs.isnull() | raw_Qs.isnull()
        #Qs_same = (combined_Qs == raw_Qs) | Qs_both_nan
        #Qs_diff = ~Qs_same.loc[~both_nan_rows, :]

        # Ignore columns that are likely to be different and don't seem to have 
        # any practical value. (I think....?)
        exclude_cols = ['missing ratio', 'vel', 'sd vel', 'number vel']
        Qs_diff.loc[:, exclude_cols] = False

        # Isolate the rows and columns where values are different
        #Qs_diff.loc[0,:] = False # ignore first row
        diff_rows = Qs_diff.any(axis=1)
        diff_cols = Qs_diff.any(axis=0)
        any_diff = diff_rows.any()

        if any_diff:
            # Get some metrics on difference
            diff_rows_count = diff_rows.sum()
            rows_count = diff_rows.shape[0]
            diff_ratio = diff_rows_count / rows_count
            tolerance = self.difference_tolerance

            is_tolerant = '' if diff_ratio < tolerance else ' NOT'
            error_msg = QsPickleProcessor.error_codes['MMD']
            msgs = [error_msg,
                    f"Difference ratio of {diff_ratio:.3f} is{is_tolerant} within tolerance of {tolerance}.",
                    f"{diff_rows_count} conflicting rows found out of {rows_count}",
                    f"Using combined Qs data",
                    ]
            self.logger.warning(msgs)

            # Write differing rows/cols to log
            diff_raw_Qs = raw_Qs.loc[diff_rows, diff_cols]
            diff_combined = combined_Qs.loc[diff_rows, diff_cols]
            self.logger.write_dataframe(diff_raw_Qs, "Raw Qs")
            self.logger.write_dataframe(diff_combined, "Combined Qs")

            #if diff_ratio < diff_tolerance:
            #    raise NotImplementedError

            self.final_output = combined_Qs
            # default to using the combined output

        else:
            self.logger.write(["Qs.txt matches combined Qs chunk data",
                              "(Excluding velocity columns and missing ratio)"])
            self.final_output = combined_Qs
    def produce_processed_pickle(self):
        if self.final_output is not None:
            prepickles = {self.pkl_name : self.final_output}
            # prepickles is a dictionary of {'pickle name':data}
            self.loader.produce_pickles(prepickles)
        else:
            error_msg = QsPickleProcessor.error_codes['NDF']
            self.logger.warning([error_msg,
                f"Pickle not created for {self.pkl_name}"])



if __name__ == "__main__":
    # Run the script
    processor = QsPickleProcessor()
    processor.run()
