
from os.path import join as pjoin
from os.path import split as psplit
import numpy as np
import pandas as pd
from time import asctime
from xlrd.biffh import XLRDError

#import re
#import matplotlib.pyplot as plt
#import scipy as scp
#import statsmodels.api as sm

# From helpyr
from helpyr import data_loading
from helpyr import logger
from helpyr import crawler as helpyr_crawler
from helpyr import helpyr_misc as hm

from omnipickle_manager import OmnipickleManager
import global_settings as settings


class DEMProcessor:

    def __init__(self):
        self.root = settings.cart_data_dir
        self.pickle_destination = settings.cart_pickles_dir
        self.log_filepath = pjoin(settings.log_dir, "dem_processor.txt")
        
        # Start up logger
        self.logger = logger.Logger(self.log_filepath, default_verbose=True)
        self.logger.write(["Begin DEM Processor output", asctime()])

        # Start up loader
        self.loader = data_loading.DataLoader(self.pickle_destination, logger=self.logger)
        
        # Reload omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()
        self.omnimanager.update_tree_definitions()

    def run(self):
        indent_function = self.logger.run_indented_function

        indent_function(self.find_data_files,
                before_msg="Finding manual data files", after_msg="Finished!")

        indent_function(self.load_data,
                before_msg="Loading data", after_msg="Finished!")
        
        indent_function(self.update_omnipickle,
                before_msg="Updating omnipickle", after_msg="Finished!")

        indent_function(self.omnimanager.store,
                before_msg="Storing omnipickle", after_msg="Finished!",
                kwargs={'overwrite' : {'dem': True}})

    def find_data_files(self):
        # Find all the dem text files
        self.logger.write("")
        crawler = helpyr_crawler.Crawler(logger=self.logger)
        crawler.set_root(self.root)
        self.dem_txt_filepaths = crawler.get_target_files(
                "*_clean_dem.txt", verbose_file_list=True)
        crawler.end()

    def load_data(self):
        # Load all the manual data files and combine
        indent_function = self.logger.run_indented_function

        self.all_dem_data = {}
        indent_function(self._load_dem_txt,
                before_msg="Loading dem data", after_msg="DEM data extraction complete")

    def _load_dem_txt(self):
        kwargs = {}

        for dem_filepath in self.dem_txt_filepaths:

            ### Extract experiment info from filepath
            # example path: /home/alex/feed-timing/data/cart/3B/\
            # rising-75L/full-bed-laser/3B-r75L-t60-8m_clean_dem.txt
            # One file per experiment
            dem_dir, dem_filename = psplit(dem_filepath)
            period_info, dem_type = dem_filename.split('_', 1)
            exp_code, step, time, length = period_info.split('-')

            self.logger.write(f"Extracting {dem_filename}")

            # Read and prep raw data
            dem = self.loader.load_txt_np(dem_filepath, kwargs, add_path=False)

            if exp_code in self.all_dem_data:
                self.all_dem_data[exp_code][period_info] = dem
            else:
                self.all_dem_data[exp_code] = {period_info : dem}

    def update_omnipickle(self):
        # Add manual data to omnipickle
        hm.ensure_dir_exists(self.pickle_destination)
        self.omnimanager.add_dem_data(self.pickle_destination,
                self.all_dem_data)



if __name__ == "__main__":
    dem_processor = DEMProcessor()
    dem_processor.run()
