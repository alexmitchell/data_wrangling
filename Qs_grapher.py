#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from time import asctime

# From Helpyr
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from helpyr_misc import print_entire_df
from logger import Logger
from crawler import Crawler
from data_loading import DataLoader


class QsGrapher:

    def __init__(self):
        # File locations
        self.root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
        self.pickle_source = f"{self.root_dir}/processed-pickles"
        self.log_filepath = "./log-files/Qs_grapher.txt"
        
        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.write(["Begin Qs Grapher output", asctime()])

        # Start up loader
        self.loader = DataLoader(self.pickle_source, logger=self.logger)

    def make_experiment_plots(self):
        self.logger.write(["Making experiment plots..."])

        self.experiment_pickles = {}

        indent_function = self.logger.run_indented_function

        indent_function(self.get_picklepaths,
                before_msg="Getting pickle paths", after_msg="Finished!")

        indent_function(self.load_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        self.logger.end_output()

    def get_picklepaths(self):
        # Fill out the experiment_pickles dict with {experiment : [step, 
        # period, path]} info
        crawler = Crawler(self.logger)
        crawler.set_root(self.pickle_source)
        pkl_filepaths = crawler.get_target_files("*.pkl", verbose_file_list=False)

        for picklepath in pkl_filepaths:
            pkl_name = nsplit(picklepath, 1)[1][:-4]
            _, experiment, step, period = pkl_name.split('_')

            if experiment in self.experiment_pickles:
                self.experiment_pickles[experiment].append((step, period, picklepath))
            else:
                self.experiment_pickles[experiment] = [(step, period, picklepath)]

    def load_data(self):
        ranking = { "rising"  : 0,
                  "falling" : 1,
                  "50L"  : 00,
                  "62L"  : 10,
                  "75L"  : 20,
                  "87L"  : 30,
                  "100L" : 40,
                  "t00-t10" : 0,
                  "t10-t20" : 1,
                  "t00-t20" : 2,
                  "t20-t30" : 3,
                  "t30-t40" : 4,
                  "t20-t40" : 5,
                  "t40-t50" : 6,
                  "t50-t60" : 7,
                  "t40-t60" : 8,
                  "t00-t60" : 9,
                  }
        exp_pickles = self.experiment_pickles
        for experiment in exp_pickles.keys():
            for period_data in exp_pickles[experiment]:
                step, period, picklepath = period_data
                limb, discharge = step.split('-')

                get_ranking = lambda l, d, p: d + p + l*(40-d)
                rank = get_ranking(*[ranking[k] for k in (limb, discharge, period)])

                print(rank, picklepath[-29:])

            raise NotImplementedError
        #for Qs_path in self.Qs_path_list:
        #    pkl_name = nsplit(Qs_path, 1)[1]
        #    stripped_name = pkl_name.split('.')[0]
        #    Qs_name = stripped_name.split('_')[-1]
        #    bedload_data = Qs_period_data[Qs_path]

        #    if Qs_name == 'Qs':
        #        assert(self.Qs0_data is None)
        #        self.Qs0_data = bedload_data
        #    else:
        #        assert(Qs_name[2:].isdigit())
        #        self.Qsn_data.append(bedload_data)
        #        self.Qsn_names.append(Qs_name)



if __name__ == "__main__":
    # Run the script
    grapher = QsGrapher()
    grapher.make_experiment_plots()
