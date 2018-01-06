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


class PeriodData:
    # The PeriodData class is meant as a simple container to store all the 
    # information associated with each period

    period_ranking = {"rising" : 0, "falling" : 1,
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
        
    def __init__(self, picklepath):
        self.picklepath = picklepath
        self.pkl_name = nsplit(picklepath, 1)[1].split('.',1)[0]
        _, self.exp_code, self.step, self.period = self.pkl_name.split('_')
        
        # Calculate period ranks
        limb, flow = self.step.split('-')
        ranking = PeriodData.period_ranking
        get_ranking = lambda l, d, p: d + p + 2*l*(40-d)
        self.rank = get_ranking(*[ranking[k] for k in (limb, flow, self.period)])

        self.data = None

    def load_data(self, loader):
        self.data = loader.load_pickle(self.picklepath, add_path=False)


class Experiment:

    def __init__(self, experiment_code):
        self.code = experiment_code
        self.periods = {} # { rank : PeriodData}
        self.sorted_ranks = []

    def save_period_data(self, period_data):
        try:
            assert(period_data.rank not in self.periods)
        except AssertionError:
            print(f"AssertionError in {self.code}")
            print("new: ", period_data.rank, period_data.pkl_name)
            other = self.periods[period_data.rank]
            print("old: ", other.rank, other.pkl_name)
            raise
        self.periods[period_data.rank] = period_data

    def sort_ranks(self):
        self.sorted_ranks = [ r for r in self.periods.keys()]
        self.sorted_ranks.sort()

    def load_data(self, loader):
        for period_data in self.periods.values():
            period_data.load_data(loader)

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

        self.experiments = {}

        indent_function = self.logger.run_indented_function

        indent_function(self.store_pickle_info,
                before_msg="Getting pickle info", after_msg="Finished!")

        indent_function(self.load_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.accumulate_time,
                before_msg="Accumulating time", after_msg="Time accumulated!")

        self.logger.end_output()

    def store_pickle_info(self):
        # Fill out the experiments dict with {experiment code : Experiment}
        # Create PeriodData and Experiment objects

        # Find files
        crawler = Crawler()
        crawler.set_root(self.pickle_source)
        pkl_filepaths = crawler.get_target_files("*.pkl", verbose_file_list=False)
        crawler.end()
        
        for picklepath in pkl_filepaths:
            # Create PeriodData objects and hand off to Experiment objects
            period_data = PeriodData(picklepath)

            exp_code = period_data.exp_code
            if exp_code not in self.experiments:
                self.experiments[exp_code] = Experiment(exp_code)

            self.experiments[exp_code].save_period_data(period_data)

    def load_data(self):
        for experiment in self.experiments.values():
            experiment.sort_ranks()
            experiment.load_data(self.loader)

    def accumulate_time(self):
        raise NotImplementedError


if __name__ == "__main__":
    # Run the script
    grapher = QsGrapher()
    grapher.make_experiment_plots()
