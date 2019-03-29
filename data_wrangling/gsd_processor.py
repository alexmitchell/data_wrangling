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

# currently set to handed 2m gsd
# Doing it the quick way....
# Search for: self.gsd_txt_filepaths = 

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
        
        # Reload omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()
        self.logger.write("Updating experiment definitions")
        self.omnimanager.update_tree_definitions()

        self.gsd_txt_filepaths = []

    def run(self):
        indent_function = self.logger.run_indented_function

        indent_function(self.find_gsd_txt_files,
                before_msg="Finding GrainSize files", after_msg="Finished!")

        indent_function(self.load_data,
                before_msg="Loading and merging data", after_msg="Finished!")
        
        indent_function(self.update_omnipickle,
                before_msg="Updating omnipickle", after_msg="Finished!")

        indent_function(self.omnimanager.store,
                kwargs={'overwrite':{'gsd':True}},
                before_msg="Storing omnipickle", after_msg="Finished!")

        #print(self.omnimanager.experiments['1B'].periods[35].gsd_picklepath)

    def find_gsd_txt_files(self):
        # Find all the GrainSize.txt files
        self.logger.write("")
        #crawler = Crawler(logger=self.logger)
        #crawler.set_root(self.root)
        #self.gsd_txt_filepaths = crawler.get_target_files(
        #        "??-*L-t??-8m_sta-*_GrainSize.txt", verbose_file_list=False)
        # example filename: 3B-r87L-t60-8m_sta-2000_GrainSize.txt
        #crawler.end()
        #gsd_list_path = '/home/alex/feed-timing/code/matlab/supporting-files/2m_2B_final_gsd_list.txt'
        gsd_list_paths = [
                '/home/alex/feed-timing/code/matlab/supporting-files/8m_final_gsd_list.txt',
                '/home/alex/feed-timing/code/matlab/supporting-files/2m_final_gsd_list.txt',
                ]
                
        for gsd_list_path in gsd_list_paths:
            with open(gsd_list_path) as f:
                self.logger.write(f'Reading from {gsd_list_path}')
                self.gsd_txt_filepaths.extend(f.read().splitlines())

    def load_data(self):
        # Load all the GrainSize.txt files and combine
        gsd_txt_kwargs = {
                'index_col' : None,
                'header'    : 0,
                'skiprows'  : [1],
                }
        run_data_frames = []
        for gsd_filepath in self.gsd_txt_filepaths:
            # Pull apart provided filepath to GrainSize.txt to get run info 
            gsd_dir, gsd_name = nsplit(gsd_filepath, 1)
            gsd_name = gsd_name.split('.', 1)[0]
            scan_name, sta_str, _ = gsd_name.split('_')
            exp_code, step, period, scan_length = scan_name.split('-')

            # Calculate experiment time based on step and period
            is_falling = step[0] == 'f'
            discharge = int(step[1:-1])
            period_time = int(period[1:])

            discharge_order = [50, 62, 75, 87, 100]
            discharge_index = discharge_order.index(discharge)
            n_discharges = len(discharge_order)

            calc_time = lambda l, d, t: t + 60*(d + 2*l*(n_discharges-1-d))
            exp_time = calc_time(is_falling, discharge_index, period_time)

            # Generate name to grain size fraction file
            gsf_name = f"{scan_name}_{sta_str}_GrainSizeFractions.txt"
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
            index_names = ['exp_code', 'step', 'period', 'sta_str', 'scan_length']
            index_vals = [exp_code, step, period, sta_str, scan_length]
            var_names = index_names + ['scan_name', 'exp_time']
            var_vals = index_vals + [scan_name, exp_time]
            for var_name, var_val in zip(var_names, var_vals):
                run_data[var_name] = var_val

            run_data_frames.append(run_data)

        # Add data to combined data
        self.all_data = pd.concat(run_data_frames, ignore_index=True)
        self.all_data.set_index(index_names, inplace=True)
        self.all_data.sort_index(inplace=True)

        # Convert size classes from string to float to make consistent with 
        # sieve data
        col_conv = {
                '0.5'  : 0.5 , '0.71' : 0.71, '1'    : 1   , '1.4'  : 1.41,
                '2'    : 2   , '2.8'  : 2.83, '4'    : 4   , '5.6'  : 5.66,
                '8'    : 8   , '11.3' : 11.2, '16'   : 16  , '22.6' : 22.3,
                '32'   : 32  ,
                }
        self.all_data.columns = [col_conv[c] if c in col_conv else c \
                for c in self.all_data.columns]

    def update_omnipickle(self):
        # Add gsd data to omnipickle
        ensure_dir_exists(settings.cart_pickles_dir)
        self.omnimanager.add_gsd_data(settings.cart_pickles_dir, self.all_data)


if __name__ == "__main__":
    # Run the script
    gsd_processor = GSDProcessor()
    gsd_processor.run()
