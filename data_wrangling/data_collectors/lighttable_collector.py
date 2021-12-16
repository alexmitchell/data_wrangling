#!/usr/bin/env python3
from data_wrangling.data_collectors.data_collector_base import DataCollectorBase

class LightTableCollector(DataCollectorBase):
    def __init__(self, data_root_dir, output_dir):
        super().__init__(data_root_dir, output_dir)
        raise NotImplementedError

    def is_my_file(self, filepath):
        # returns true if filepath is a light table file
        raise NotImplementedError

    def load_data_file(self, filepath):
        # returns data as a pandas dataframe
        # will be entered into a data dict as {filename:pd_data}
        raise NotImplementedError

    def combine_data(self, data_dict):
        # Returns a pandas dataframe containing all the entries in data_dict 
        # {filename:pd_data}.
        raise NotImplementedError

    def save_file(self, data, ext):
        # save the data to a file with the extension ext
        raise NotImplementedError


