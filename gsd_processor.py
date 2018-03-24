#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from time import asctime

from data_loading import DataLoader
from logger import Logger
from crawler import Crawler
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from helpyr_misc import exclude_df_cols

from omnipickle_manager import OmnipickleManager
from crawler import Crawler
import global_settings as settings

class GSDProcessor:
    """ Collects combines the distributed GrainSize.txt files. Updates the omnipickle."""

    def __init__(self):
        self.root = settings.cart_data_dir
        self.pickle_destination = settings.cart_pickles_dir
        self.log_filepath = f"{settings.log_dir}/gsd_processor.txt"
        
        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.write(["Begin GSD Processor output", asctime()])

        # Start up loader
        self.loader = DataLoader(self.pickle_destination, logger=self.logger)
        
        # Relead omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()
        self.logger.write("Updating experiment definitions")
        self.omnimanager.update_tree_definitions()

    def run(self):
        indent_function = self.logger.run_indented_function

        indent_function(self.find_gsd_txt_files,
                before_msg="Finding GrainSize files", after_msg="Finished!")

        indent_function(self.load_data,
                before_msg="Loading and merging data", after_msg="Finished!")
        
        indent_function(self.update_omnipickle,
                before_msg="Updating omnipickle", after_msg="Finished!")

        indent_function(self.omnimanager.store,
                before_msg="Storing omnipickle", after_msg="Finished!")

        print(self.omnimanager.experiments['1B'].periods[35].gsd_picklepath)

    def find_gsd_txt_files(self):
        # Find all the GrainSize.txt files
        self.logger.write("")
        crawler = Crawler(logger=self.logger)
        crawler.set_root(self.root)
        self.gsd_txt_filepaths = crawler.get_target_files(
                "??-*L-t??-8m_sta-*_GrainSize.txt", verbose_file_list=False)
        # example filename: 3B-r87L-t60-8m_sta-2000_GrainSize.txt
        crawler.end()

    def load_data(self):
        # Load all the GrainSize.txt files and combine
        gsd_txt_kwargs = {
                'index_col' : None,
                'header'    : 0,
                'skiprows'  : [1],
                }
        self.all_data = pd.DataFrame
        run_data_frames = []
        for gsd_filepath in self.gsd_txt_filepaths:
            # Pull apart provided filepath to GrainSize.txt to get run info 
            gsd_dir, gsd_name = nsplit(gsd_filepath, 1)
            gsd_name = gsd_name.split('.', 1)[0]
            data_name, sta_str, _ = gsd_name.split('_')
            exp_code, step, period, scan_length = data_name.split('-')

            # Generate name to grain size fraction file
            gsf_name = f"{data_name}_{sta_str}_GrainSizeFractions.txt"
            gsf_filepath = os.path.join(gsd_dir, gsf_name)

            # Both data files are read exactly the same, do in a loop
            run_data = pd.DataFrame()
            for filepath in [gsd_filepath, gsf_filepath]:
                # Load data and set the index label
                data = self.loader.load_txt(filepath,
                        gsd_txt_kwargs, add_path=False)
                #data.index = run_multiindex
                run_data = pd.concat([run_data, data], axis=1)

            # Add columns that will be later used for a multiindex
            var_names = ['exp_code', 'step', 'period', 'sta_str', 'scan_length']
            var_vals = [exp_code, step, period, sta_str, scan_length]
            for var_name, var_val in zip(var_names + ['data_name'],
                                         var_vals + [data_name]):
                run_data[var_name] = var_val

            run_data_frames.append(run_data)

        # Add data to combined data
        self.all_data = pd.concat(run_data_frames, ignore_index=True)
        self.all_data.set_index(var_names, inplace=True)

    def update_omnipickle(self):
        # Add gsd data to omnipickle
        self.omnimanager.add_gsd_data(settings.cart_data_dir, self.all_data)


if __name__ == "__main__":
    # Run the script
    gsd_processor = GSDProcessor()
    gsd_processor.run()
