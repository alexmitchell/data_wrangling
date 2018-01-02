import os
import numpy as np

# From Helpyr
import data_loading
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from logger import Logger
from crawler import Crawler

from xlrd.biffh import XLRDError



# ISSUE TO ADDRESS: Some Qs.txt and Qs1.txt files appear to be nearly identical 
# copies. The first row is usually different and sometimes a random row where a 
# grain count is 1 off. Preference is given to Qs1.txt?
#
# 4A/rising-87L/results-t40-t60 -> Multiple Qs files are wrong.  Full 
# of zero values. Maybe linked to minimum particles needed for 
# velocity?? Might need to reanalyze many files....

class ExtractionCrawler (Crawler):
    # The Extraction Crawler does the initial work of finding all the data 
    # files and converting them to pickles

    def __init__(self, log_filepath="./log-files/extraction-crawler.txt"):
        logger = Logger(log_filepath, default_verbose=True)
        Crawler.__init__(self, logger)

        self.mode_dict['extract-manual']      = self.run_extract_manual_data
        self.mode_dict['extract-light-table'] = self.run_extract_light_table_data

    def make_pickle(self, pkl_name, data):
        self.logger.write(["Performing picklery on {}".format(pkl_name)])
        self.logger.increase_global_indent()
        picklepaths = self.loader.produce_pickles({pkl_name:data})
        self.logger.decrease_global_indent()
        return picklepaths

    def generic_intro(self, start_run_msgs=[]):
        # Do some common intro stuff, like writing to the log file and starting 
        # the data loader.
        self.logger.write_section_break()
        self.logger.write(start_run_msgs)

        data_path = self.root
        pickle_path = os.path.join(data_path, 'raw-pickles')
        ensure_dir_exists(pickle_path, self.logger)
        self.loader = data_loading.DataLoader(data_path, pickle_path, self.logger)


    def run_extract_manual_data(self):
        # Extract the depth and data from excel files and save them as  
        # pickles.
        self.generic_intro(["Extracting manual data"])

        self.logger.write(["Finding files"])
        depths_xlsx = self.get_target_files(['flow-depths-??.xlsx'])
        masses_xlsx = self.get_target_files(['masses-??.xlsx'])

        self.extract_depths(depths_xlsx)

        self.end()

    def extract_depths(self, depths_xlsx):
        self.logger.write(["Extracting depth data"])
        self.logger.increase_global_indent()
        kwargs = {
                'sheetname'  : 'Sheet1',
                'header'     : 0,
                'skiprows'   : 1,
                'index_col'  : [0, 1, 2, 3],
                'parse_cols' : range(1,18)
                }

        for depth_filepath in depths_xlsx:
            self.logger.write(["Extracting {}".format(depth_filepath)])

            data_dic = {} # For temp processing... delete later

            # Extract experiment from filepath
            depth_path, depth_filename = os.path.split(depth_filepath)
            source_name, experiment = depth_filename.rsplit('-', 1)
            experiment = experiment.split('.')[0]

            # Read and prep raw data
            try:
                data = self.loader.load_xlsx(depth_filepath, kwargs, is_path=True)
            except XLRDError:
                kwargs['sheetname'] = "All"
                data = self.loader.load_xlsx(depth_filepath, kwargs, is_path=True)
            data = self.reformat_data(data)

            # Extract surface and bed measurements then save them as pickles
            slice_all = slice(None)
            for location in "surface", "bed":
                # Pull out all rows with a location then drop the location 
                # label
                loc_data = data.loc[(slice_all, slice_all, slice_all, slice_all, location),:]
                loc_data.index = loc_data.index.droplevel(4) # Drop location category

                # Make pickles
                pkl_name = "{}-{}-{}".format(experiment, "profile", location)
                self.make_pickle(pkl_name, loc_data)
                data_dic[location] = loc_data
                self.logger.decrease_global_indent()

        self.logger.decrease_global_indent()
        self.logger.write(["Depth data extraction complete"])

    def reformat_data(self, data):
        # Rename the row labels and reorder all the rows to chronologically 
        # match my experiment design. Without reordering, falling comes before 
        # rising and the repeated discharged could get confused.
        self.logger.write(["Reformatting data"])
        data.sort_index(inplace=True)

        # Rename provided level names
        new_names = ['Limb', 'Discharge (L/s)', 'Time (min)', 'Location']
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
        name = 'Exp time'
        data[name] = data.index
        data[name] = data[name].map(orderizer)
        data.set_index(name, append=True, inplace=True)

        # Make the Order index level zero
        new_names.insert(0, name)
        data = data.reorder_levels(new_names)

        data.sort_index(inplace=True)
        return data


    def run_extract_light_table_data(self):
        # Extract the Qs data from text files and save them as pickles
        self.generic_intro(["Extracting light table data"])

        self.logger.write(["Finding files"])
        sediment_flux_files = self.get_target_files(['Qs*.txt'],
                verbose_file_list=False)

        self.extract_light_table(sediment_flux_files)

        self.end()

    def extract_light_table(self, sediment_flux_txt_files):
        self.logger.write(["Extracting light table data"])
        self.logger.increase_global_indent()

        # Prepare kwargs for reading Qs text files
        Qs_column_names = [
                # Timing and meta data
                #'elapsed-time sec', <- Calculate this column later
                'timestamp', 'missing ratio', 'vel', 'sd vel', 'number vel',
                # Bedload transport masses (g)
                'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                'Bedload 22', 'Bedload 32', 'Bedload 45',
                # Grain counts
                'Count all', 'Count 0.5', 'Count 0.71', 'Count 1', 'Count 1.4',
                'Count 2', 'Count 2.8', 'Count 4', 'Count 5.6', 'Count 8',
                'Count 11.2', 'Count 16', 'Count 22', 'Count 32', 'Count 45',
                # Statistics
                'D10', 'D16', 'D25', 'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax'
                ]
        Qs_kwargs = {
                'index_col' : None,
                'header'    : None,
                'names'     : Qs_column_names,
                }

        period_dict = self.build_period_dict(sediment_flux_txt_files)
        pickle_dict = {}

        # Create new pickles if necessary
        for period_path in period_dict:
            self.logger.write([f"Extracting {period_path}"])
            self.logger.increase_global_indent()

            fnames = period_dict[period_path]
            pickle_dict[period_path] = self.pickle_Qs_text_files(
                    period_path, fnames, Qs_kwargs)
            self.logger.decrease_global_indent()

        # Update the metapickle
        # it describes which pkl files belong to which periods
        metapickle_name = "Qs_metapickle"
        if self.loader.is_pickled(metapickle_name):
            # Pickled metapickle already exists.
            # Update the metapickle
            self.logger.write([f"Updating {metapickle_name}..."])
            existing = self.loader.load_pickle(metapickle_name, use_source=False)
            pickle_dict = self._merge_metapickle(pickle_dict, existing)
        self.make_pickle(metapickle_name, pickle_dict)

        self.logger.decrease_global_indent()
        self.logger.write(["Light table data extraction complete"])
        #self.check_for_Qs_errors(data)
        #self.check_for_Qs_duplication
    
    def _merge_metapickle(self, new_dict, old_dict):
        merge = lambda a, b: list(set(a + b))
        file_name = lambda path: path.rsplit('_',1)[1]
        file_num = lambda name: int(name[2:-4]) if name[2:-4].isdigit() else 0
        sort_key = lambda path: file_num(file_name(path))

        #period_dict[period_path].sort(key=file_num)

        nd = new_dict
        od = old_dict
        old_dict.update(
            {nk: merge(nd[nk], od[nk] if nk in od else []) for nk in nd.keys()})
        for key in old_dict:
            old_dict[key].sort(key=sort_key)
        print(old_dict)
        return old_dict

    def build_period_dict(self, sediment_flux_txt):
        # Build a dict of associated Qs#.txt files per 20 minute periods
        # Dict values are sorted lists of Qs#.txt file paths
        # Dict keys are the results directory paths
        period_dict = {}
        self.logger.write(["Building dict of Qs files"])
        for Qs_filepath in sediment_flux_txt:

            # Extract meta data from filepath
            fpath, fname = nsplit(Qs_filepath, 1)

            if fpath in period_dict:
                period_dict[fpath].append(fname)
            else:
                period_dict[fpath] = [fname]

        # Sort the associated files
        for period_path in period_dict:
            file_num = lambda s: int(s[2:-4]) if s[2:-4].isdigit() else 0
            period_dict[period_path].sort(key=file_num)
            #print(period_path[-30:], period_dict[period_path])

        return period_dict

    def pickle_Qs_text_files(self, period_path, Qs_names, Qs_kwargs):
        # Check for preexisting pickles.
        # If they don't exist yet, create new ones and return the filepaths
        _, experiment, step, rtime = nsplit(period_path, 3)

        picklepaths = []
        for name in Qs_names:
            pkl_name = f"{experiment}_{step}_{rtime[8:]}_{name[:-4]}"

            if self.loader.is_pickled(pkl_name):
                self.logger.write([f'Pickle {name} preexists. Nothing to do.'])
            else:
                self.logger.write([f'Pickling {name}'])
                
                # Read and prep raw data
                filepath = os.path.join(period_path, name)
                data = self.loader.load_txt(filepath, Qs_kwargs, is_path=True)

                # Make pickles
                picklepaths += self.make_pickle(pkl_name, data)

        return picklepaths

    def check_for_Qs_errors(self, Qs_data):
        # Check for basic errors in the Qs_data
        
        # Check for all zero values
        nrows = Qs_data.shape[0]
        nzeros = np.count_nonzero(Qs_data, axis=0)

        percent_zeros = nzeros / nrows
        print_options = np.get_printoptions()
        np.set_printoptions(precision=2)
        self.logger.write(["Percent zeros", percent_zeros.__str__()])
        np.set_printoptions(precision=print_options['precision'])



if __name__ == "__main__":
    crawler = ExtractionCrawler()
    exp_root = '/home/alex/ubc/research/feed-timing/data/{}'
    #crawler.set_root(exp_root.format('data-links/manual-data'))
    #crawler.run('extract-manual')
    crawler.set_root(exp_root.format('extracted-lighttable-results'))
    crawler.run('extract-light-table')
