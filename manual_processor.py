
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
from data_loading import DataLoader
from logger import Logger
from crawler import Crawler
from helpyr_misc import ensure_dir_exists

from omnipickle_manager import OmnipickleManager
from crawler import Crawler
import global_settings as settings


class ManualProcessor:

    def __init__(self):
        self.root = settings.manual_data_dir
        self.pickle_destination = settings.manual_pickles_dir
        self.log_filepath = pjoin(settings.log_dir, "manual_processor.txt")
        
        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.write(["Begin Manual Processor output", asctime()])

        # Start up loader
        self.loader = DataLoader(self.pickle_destination, logger=self.logger)
        
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
                kwargs={'overwrite' : {'depth': True}})

    def find_data_files(self):
        ### Find all the flow depth and trap masses text files
        self.logger.write("")
        crawler = Crawler(logger=self.logger)
        crawler.set_root(self.root)
        
        # example filename: 3B-flow-depths.xlsx  3B-masses.xlsx
        #self.depth_xlsx_filepaths = crawler.get_target_files(
        #        "??-flow-depths.xlsx", verbose_file_list=True)
        #self.masses_xlsx_filepaths = crawler.get_target_files(
        #        "??-masses.xlsx", verbose_file_list=True)

        # Filenames for sieve data do not have a unique key word. Grab all 
        # excel files in the SampleSieveData directory
        self.sieve_xlsx_filepaths = crawler.get_target_files(
                target_names="*.xlsx",
                target_dirs="SampleSieveData",
                verbose_file_list=False)

        crawler.end()

    def load_data(self):
        # Load all the manual data files and combine
        indent_function = self.logger.run_indented_function

        self.all_depth_data = {} # {exp_code : dataframe}
        #indent_function(self._load_depth_xlsx,
        #        before_msg="Loading depth data", after_msg="Depth data 
        #        extraction complete")

        self.all_masses_data = {} # {exp_code : dataframe}
        #indent_function(self._load_masses_xlsx,
        #        before_msg="Loading mass data...",
        #        after_msg="Finished loading mass data!")

        self.all_sieve_data = {} # {exp_code : dataframe}
        indent_function(self._load_sieve_xlsx,
                before_msg="Loading sieve data...",
                after_msg="Finished loading sieve data!")

    def _load_sieve_xlsx(self):
        kwargs = {
                'sheetname'  : 'Clean',
                'header'     : 0,
                'skiprows'   : 5,
                'skip_footer': 3,
                'index_col'  : None,
                'parse_cols' : 'A:B',
                }

        #self.logger.write("Removing old references")
        #self.omnimanager._remove_datasets([f"sieve-s{i}" for i in (1,2)])

        sieve_data = {}
        for sieve_filepath in self.sieve_xlsx_filepaths:

            ### Extract experiment from filepath
            # example filename: 3B-sieves.xlsx
            # One file per experiment
            sieve_path, sieve_filename = psplit(sieve_filepath)
            parts = (sieve_filename.split('.')[0]).split('_')

            if 'feed' in sieve_filename:
                self.logger.write(f"Skipping feed sample {sieve_filename}.")
                continue

            exp_code, step = parts[0:2]
            ptime = int(''.join(c for c in parts[2] if c.isdigit()))

            if len(parts) == 3:
                # No sample label, assume sample 1
                #print('Sample 1: ', parts)
                sample = 1
            elif '1' in parts[3]:
                #print('Sample 1: ', parts)
                sample = 1
            elif '2' in parts[3]:
                #print('Sample 2: ', parts)
                sample = 2
            elif '75acc' == parts[3]:
                #print('Sample 1: ', parts, 'Special case')
                sample = 1
            else:
                print('Unhandled case')
                print(parts)
                assert(False)

            meta_kwargs = {
                    'exp_code' : exp_code,
                    'step'     : step,
                    'ptime'    : ptime,
                    'sample'   : sample,
                    }

            self.logger.write(f"Extracting {sieve_filename}")

            # Read and prep raw data
            data = self.loader.load_xlsx(sieve_filepath, kwargs, add_path=False)
            data = self._reformat_sieve_data(data, **meta_kwargs)

            if exp_code in sieve_data:
                sieve_data[exp_code].append(data)
            else:
                sieve_data[exp_code] = [data]

        for exp_code, frames in sieve_data.items():
            combined_data = pd.concat(frames)
            self.all_sieve_data[exp_code] = combined_data.sort_index()

    def _reformat_sieve_data(self, data, **kwargs):
        # Clean up the data by providing experiment timestamps, sorting, and 
        # fixing zero values.
        # kwargs must have exp_code, step, ptime, and sample variables
        self.logger.write("Reformatting data")

        exp_code = kwargs['exp_code']
        step = kwargs['step']
        limb = 'rising' if step[0] == 'r' else 'falling'
        discharge = int(''.join(c for c in step if c.isdigit()))
        ptime = kwargs['ptime']
        sample = kwargs['sample']

        # Rename column labels and names
        data.columns = ['size (mm)', 'mass (g)']

        # Shift sizes so it represents pass through rather than retained
        pan_index = data['size (mm)'] == 'pan'
        pan = data.loc[pan_index, 'mass (g)'].iloc[0]
        data['size (mm)'].values[1:] = data['size (mm)'].values[:-1]

        # Identify rows outside of the acceptable size range
        sizes = data['size (mm)']
        outside = (sizes < 0.5) | (64 <= sizes)

        outside_data = data.loc[outside & ~pan_index, 'mass (g)']
        if (outside_data.notnull() & (outside_data != 0)).any():
            print('Impossible masses found')
            print(f"{exp_code} {step} {ptime}")
            print(data)
            assert(False)

        # Remove >= 64 because they are impossible
        # Remove < 0.5 because they are not measured, but keep pan
        data.drop(data.index[outside], inplace=True)
        # set 45mm (0th item) to 0g if it is nan
        data.iloc[0, 1] = 0 if np.isnan(data.iloc[0,1]) else data.iloc[0,1]
        # replace the pan value
        data.iloc[-1, 1] = pan

        # Because the files do not use the same # decimals for the sizes,
        # hardcode a consistent size column
        data['size (mm)'] = np.array([
            45, 32, 22.3, 16, 11.2, 8, 5.66,
            4, 2.83, 2, 1.41, 1, 0.71, 0.5])

        # Transpose and sort
        data.set_index('size (mm)', inplace=True)
        data = (data.T).sort_index(axis=1)

        # Add a multiindex to the rows
        key = limb, discharge, ptime, sample
        key_names = 'limb', 'discharge', 'time', 'sample'
        data.index = pd.MultiIndex.from_tuples([key], names=key_names)

        # Add a new level providing the experiment time
        def _get_exp_time(args):
            limb, discharge, period_time, _ = args
            weights = {'rising'  : 0,
                       'falling' : 1,
                       50  : 0,
                       62  : 1,
                       75  : 2,
                       87  : 3,
                       100 : 4
                      }
            w_Qmax = weights[100]
            w_limb = weights[limb]
            w_Q = weights[discharge]
            order = w_Q  + 2 * w_limb * (w_Qmax - w_Q)
            try:
                p_time = int(period_time)
            except ValueError:
                p_time = 60
            exp_time = order * 60 + p_time
            return exp_time
        name = 'exp_time'
        data[name] = data.index
        data[name] = data[name].map(_get_exp_time)
        data.set_index('exp_time', append=True, inplace=True)

        #if (exp_code, step, ptime) == ('3B', 'r62', 60):
        #    print(data)
        #    assert(False)

        # Check to make sure the data is numeric
        try:
            assert((data.dtypes == np.float64).all())
        except AssertionError:
            print(data)
            print("Data is not all float64")
            #print(data.apply(pd.to_numeric))
            #print(data.apply(pd.to_numeric).dtypes)
            raise

        return data


    def _load_masses_xlsx(self):
        kwargs = {
                'sheetname'  : 'Sheet1',
                'header'     : [0,1],
                'skiprows'   : 0,
                'index_col'  : [0, 1, 2],
                'parse_cols' : 'B:M',
                'na_values'  : ['#DIV/0!', '#VALUE!'],
                'skip_footer': 4,
                }

        self.logger.write("Removing old references")
        self.omnimanager._remove_datasets([f"masses-s{i}" for i in (1,2)])

        masses_data = {}
        for masses_filepath in self.masses_xlsx_filepaths:

            ### Extract experiment from filepath
            # example filename: 3B-massess.xlsx
            # One file per experiment
            masses_path, masses_filename = psplit(masses_filepath)
            exp_code, source_str = masses_filename.split('-', 1)

            self.logger.write(f"Extracting {masses_filename}")

            # Read and prep raw data
            data = self.loader.load_xlsx(masses_filepath, kwargs, add_path=False)
            #try:
            #    data = self.loader.load_xlsx(masses_filepath, kwargs, add_path=False)
            #except XLRDError:
            #    kwargs['sheetname'] = 'All'
            #    data = self.loader.load_xlsx(masses_filepath, kwargs, add_path=False)
            #    kwargs['sheetname'] = 'Sheet1'
            data = self._reformat_masses_data(data)

            self.all_masses_data[exp_code] = data

    def _reformat_masses_data(self, data):
        # Clean up the data by providing experiment timestamps, sorting, and 
        # fixing zero values.
        self.logger.write("Reformatting data")

        # Rename index names
        new_names = ['limb', 'discharge', 'period_time']
        data.index.names = new_names
        
        # Rename column labels and names
        # Add spaces between name and (kg)
        def add_space(s):
            if '(kg)' in s:
                i = s.rindex('(kg)')
                s = s[:i] + ('' if s[i-1] == ' ' else ' ') + s[i:]
            return s
        primary = ['Total wet'] + ['Sample 1'] * 4 + ['Sample 2'] * 4
        secondary = [add_space(c) for c in list(data.columns.get_level_values(1))]
        columns = pd.MultiIndex.from_tuples(list(zip(primary, secondary)),
                names=['Sample', 'Measurement'])
        data.columns = columns

        # Delete 'Total' rows (ie. 'Total' in period_time level)
        # They are duplicate data and makes slicing a pain in the butt
        data.drop(labels='Total', level='period_time', inplace=True)

        # Add a new level providing the experiment time
        def _get_exp_time(args):
            limb, discharge, period_time = args
            weights = {'rising'  : 0,
                       'falling' : 1,
                       50  : 0,
                       62  : 1,
                       75  : 2,
                       87  : 3,
                       100 : 4
                      }
            w_Qmax = weights[100]
            w_limb = weights[limb]
            w_Q = weights[discharge]
            order = w_Q  + 2 * w_limb * (w_Qmax - w_Q)
            try:
                p_time = int(period_time)
            except ValueError:
                p_time = 60
            exp_time = order * 60 + p_time
            return exp_time
        name = 'exp_time'
        data[name] = data.index
        data[name] = data[name].map(_get_exp_time)

        #data.sort_values(by=name, inplace=True)
        data.set_index(name, append=True, inplace=True)
        #data.reset_index(col_level=1, col_fill='meta', inplace=True)
        #data.set_index(name, inplace=True)
        #data.sort_index(axis=0, level='exp_time', inplace=True)
        data.sort_index(axis=0, inplace=True)
        data.sort_index(axis=1, inplace=True)

        # Fix 0 values where appropriate
        idx = pd.IndexSlice
        tot_wet = data.loc[: , idx['Total wet', 'total wet (kg)']]
        for sample in 'Sample 1', 'Sample 2',:
            if (data == 0).any().any():
                # None of the values should be zero
                # Just crash it for manual inspection
                self.logger.write('Zero value found! Crashing!')
                print(data)
                assert(False)
            wet = data.loc[: , idx[sample, 'subset wet (kg)']]
            dry = data.loc[: , idx[sample, 'subset dry (kg)']]
            tot_dry = data.loc[: , idx[sample, 'total dry (kg)']]

            zero_subset_rows = (wet == 0) | (dry == 0)
            null_subset_rows = wet.isnull() | dry.isnull()
            bad_tot_dry_rows = (tot_dry == 0) | tot_dry.isnull()

            if ((zero_subset_rows | null_subset_rows) & ~bad_tot_dry_rows).any().any():
                # Missing subset data calculation
                self.logger.write(f"Warning: Some total dry masses exist without subset data. Continuing...")

            if (~(zero_subset_rows | null_subset_rows) & bad_tot_dry_rows).any().any():
                # incomplete calculation
                # Again, should not happen, so just crash it for manual 
                # inspection
                self.logger.write('Incomplete calculation! Crashing!')
                print(data)
                assert(False)

            ## make sure empty rows are nan
            #bad_subset_rows = zero_subset_rows | null_subset_rows
            #data.loc[bad_subset_rows, idx[sample, :]] = np.nan

            ## fix the rows that have incomplete calculations
            #tot_dry = data.loc[: , idx[sample, 'total dry (kg)']]
            #incomplete_data = tot_dry == 0 # find rows that are still zero
            #mass_ratio = dry[incomplete_data] / wet[incomplete_data]
            #data.loc[incomplete_data , idx[sample, 'mass ratio']] = mass_ratio

            #tot_dry = mass_ratio * tot_wet[incomplete_data]
            #data.loc[incomplete_data , idx[sample, 'total dry (kg)']] = tot_dry

        return data


    def _load_depth_xlsx(self):
        kwargs = {
                'sheetname'  : 'Sheet1',
                'header'     : 0,
                'skiprows'   : 1,
                'index_col'  : [0, 1, 2, 3],
                'parse_cols' : range(1,18)
                }

        surf_data = {}
        bed_data = {}
        for depth_filepath in self.depth_xlsx_filepaths:

            ### Extract experiment from filepath
            # example filename: 3B-flow-depths.xlsx
            # One file per experiment
            depth_path, depth_filename = psplit(depth_filepath)
            exp_code, source_str = depth_filename.split('-', 1)
            source = source_str.split('.')[0]

            self.logger.write(f"Extracting {depth_filename}")

            # Read and prep raw data
            try:
                data = self.loader.load_xlsx(depth_filepath, kwargs, add_path=False)
            except XLRDError:
                kwargs['sheetname'] = 'All'
                data = self.loader.load_xlsx(depth_filepath, kwargs, add_path=False)
                kwargs['sheetname'] = 'Sheet1'
            data = self._reformat_depth_data(data)
            data.sort_index(inplace=True)

            self.all_depth_data[exp_code] = data

    def _reformat_depth_data(self, data):
        # Rename the row labels and reorder all the rows to chronologically 
        # match my experiment design. Without reordering, falling comes before 
        # rising and the repeated discharged could get confused.
        self.logger.write("Reformatting data")
        data.sort_index(inplace=True)

        # Rename provided level names
        new_names = ['limb', 'discharge', 'period_time', 'location']
        data.index.names = new_names

        def orderizer(args):
            weights = {'rising'  : 0,
                       'falling' : 1,
                       50  : 0,
                       62  : 1,
                       75  : 2,
                       87  : 3,
                       100 : 4
                      }
            w_Qmax = weights[100]
            w_limb = weights[args[0]]
            w_Q = weights[args[1]]
            order = w_Q  + 2 * w_limb * (w_Qmax - w_Q)
            exp_time = order * 60 + args[2]
            return exp_time
            
        # Add a new level providing the row order
        name = 'exp_time'
        data[name] = data.index
        data[name] = data[name].map(orderizer)
        data.sort_values(by=name, inplace=True)

        #data.set_index(name, append=True, inplace=True)
        ## Make the Order index level zero
        #new_names.insert(0, name)
        #data = data.reorder_levels(new_names)
        #data.sort_index(inplace=True)

        return data


    def update_omnipickle(self):
        # Add manual data to omnipickle
        ensure_dir_exists(self.pickle_destination)
        if self.all_depth_data:
            self.omnimanager.add_depth_data(
                    self.pickle_destination, self.all_depth_data)

        if self.all_masses_data:
            self.omnimanager.add_masses_data(
                    self.pickle_destination, self.all_masses_data)

        if self.all_sieve_data:
            self.omnimanager.add_sieve_data(
                    self.pickle_destination, self.all_sieve_data)


if __name__ == "__main__":
    manual_processor = ManualProcessor()
    manual_processor.run()
