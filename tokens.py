#!/usr/bin/env python3

from os.path import join as pjoin
from time import asctime
import numpy as np
import pandas as pd

# From Helpyr
from helpyr_misc import nsplit

import global_settings as settings

# These classes were originally coded for the Qs secondary processor, but have 
# been extended to elsewhere. Therefore, they have some legacy functions that 
# should be cleaned up eventually to fit the current framework.
#
# Tokens (and Omnipickle) are originally created in the Qs_pickle_processor.  
# Remove this dependency in future version.

# To handle more and more data, perhaps change PeriodData to store a dictionary 
# of DataObjects (have to define a new token)?

# Is this tree structure even useful? Maybe split data to experiment level 
# only. It seems that a lot of effort is put into splitting the data, but the 
# first step when using the data is to add them all together again. Would be 
# easier to add new types of data or adapt to new experiments without the tree.
# Speaking of which, I've come to realize that the current version is totally 
# useless for other people's experiments.

class DataSet:
    # DataSet is a container for a type of data and the filepath (picklepath) 
    # where it is stored. Does not load the data until directed or first 
    # attempted use.
    #
    # Once a picklepath is provided, the user can access the data simply by 
    # using the .data property (ie. data_set.data).

    # Used for creation or updating
    def __init__(self, name):
        self.name = name
        self.__picklepath = None
        self.__data = None
        self.misc = {} # Miscellaneous info

    def from_existing(old_dataset):
        nds = DataSet(old_dataset.name)
        nds.picklepath = old_dataset.picklepath
        return nds

    def reload_data(self, loader):
        # Load the data from the picklepath
        self.__data = loader.load_pickle(self.picklepath, add_path=False)
        #print(self.__picklepath)
        #print(self.__data)
        #assert(False)
    def save_data(self, loader, overwrite={}):
        # Save the data to its picklepath using provided loader class.
        # overwrite = {name : overwrite_bool}
        # ^ indicates which names should or should not be overwritten. Default 
        # is to not overwrite. "all" will overwrite all pickles.
        if self.data is not None:
            name = self.name
            overwrite_bool = True if 'all' in overwrite else \
                    overwrite[self.name] if self.name in overwrite else \
                    False
            loader.produce_pickles({self.picklepath: self.data},
                    add_path=False, overwrite=overwrite_bool)

    def wipe_data(self):
        # Delete data (but keep picklepath) so DataSet can be saved without 
        # duplicating data
        self.__data = None


    # Attributes
    @property
    def picklepath(self):
        return self.__picklepath

    @picklepath.setter
    def picklepath(self, picklepath):
        self.__picklepath = picklepath

    @property
    def data(self):
        # Access the data.
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data


class PeriodData:
    # The PeriodData class is meant as a simple container to store all the 
    # information associated with each period.
    # Data includes measurements that happened during or at the end of the 
    # period
    # This class is specific to Alex's experiment, but the framework could be 
    # reused for other projects.

    # Creation and Token I/O
    def __init__(self, exp_code, limb, discharge, period_range):
        # exp_code similar to '1A', '3B', etc.
        # limb is 'rising' or 'falling'
        # discharge is int like 62, 100, etc.
        # period_range similar to 't30-t40'
       
        self.exp_code = exp_code
        self.limb = limb
        self.discharge = discharge
        self.discharge_int = int(self.discharge[:-1])
        self.period_range = period_range

        # Calculate/extract more info
        self.step = f"{limb[0]}{self.discharge_int}L" # eg. 'f75L'
        self.period_end = self.period_range.split('-')[-1] # grab the end time str
        duration_fu = lambda t0, t1: int(t1[1:]) - int(t0[1:])
        self.duration = duration_fu(*self.period_range.split('-'))
        discharge_index = settings.discharge_order.index(self.step)
        self.feed_rate = settings.feed_rates[self.exp_code][discharge_index]
        self.exp_time_start = self.calc_exp_time()
        self.exp_time_end = self.calc_exp_time(60)

        # Calculate period ranks
        period_ranking = {"rising" : 0, "falling" : 1,
                "50L" : 00, "62L" : 10, "75L" : 20, "87L" : 30, "100L" : 40,
                "t00-t10" : 0, "t10-t20" : 1, "t00-t20" : 2,
                "t20-t30" : 3, "t30-t40" : 4, "t20-t40" : 5,
                "t40-t50" : 6, "t50-t60" : 7, "t40-t60" : 8, "t00-t60" : 9,
                          }
        get_ranking = lambda l, d, p: d + p + 2*l*(40-d)
        self.rank = get_ranking(*[period_ranking[k] for k in
                                  (self.limb, self.discharge, self.period_range)])

        # Create the data dictionary {data_name : DataSet}
        # data is wiped before storage, but can be reloaded from the 
        # picklepath.
        self.data_dict = {}

    def from_existing(old_period):
        # Create a new PeriodData obj from an existing one. Returns the new 
        # PeriodData object. Used for updating the tree to new definitions of 
        # Experiment and PeriodData classes (ie. changes to this file)
        op = old_period
        np = PeriodData(op.exp_code, op.limb, op.discharge, op.period_range)
        for okey, ods in op.data_dict.items():
            np.data_dict[okey] = DataSet.from_existing(ods)
        return np
        #try:
        #    np.depth_picklepath = op.depth_picklepath 
        #except AttributeError:
        #    np.depth_picklepath = None

    def from_Qs_picklepath(Qs_picklepath):
        # generate a PeriodData object from a given Qs_picklepath

        # Get some meta data from the Qs path (### Change this someday)
        Qs_pkl_name = nsplit(Qs_picklepath, 1)[1].split('.',1)[0]
        _, exp_code, step, period_range = Qs_pkl_name.split('_')
        limb, discharge = step.split('-')

        npd = PeriodData(exp_code, limb, discharge, period_range)
        npd.add_Qs_data(Qs_picklepath)
        npd._Qs_dataset.misc['Qs_pkl_name'] = Qs_pkl_name
        return npd

    def save_data(self, loader, overwrite={}):
        # Save data prior to wiping. Only really needed when building or 
        # updating the omnipickle. (ie. for the first time the data is added to 
        # the omnipickle and isn't already pickled)
        for dataset in self.data_dict.values():
            dataset.save_data(loader, overwrite)

    def wipe_data(self):
        # So I can save the omnipickle/experiment/period objects without 
        # repickling the underlying data
        for dataset in self.data_dict.values():
            dataset.wipe_data()


    def calc_exp_time(self, partial_time=0):
        # Estimate the overall experiment time during this period.
        # Returns minutes since experiment started (based on design not 
        # necessarily data duration)
        weights = {'rising'  : 0,
                   'falling' : 1,
                   50  : 0,
                   62  : 1,
                   75  : 2,
                   87  : 3,
                   100 : 4
                  }
        w_Qmax = weights[100] # bc 100L/s is max in my experiment
        w_limb = weights[self.limb]
        w_Q = weights[self.discharge_int]
        order = w_Q  + 2 * w_limb * (w_Qmax - w_Q)
        exp_time = order * 60 + partial_time
        return exp_time
            
        ## Add a new level providing the row order
        #name = 'exp_time'
        #data[name] = data.index
        #data[name] = data[name].map(orderizer)
        #data.sort_values(by=name, inplace=True)


    # Adding data
    def _make_new_dataset(self, name, picklepath, kwargs):
        # If data is provided, then hand that off to the new DataSet to be 
        # pickled later (ie. if data is provided, then there is no pickled 
        # version yet.)
        # "specific_data" is exactly the data to hand off to the DataSet
        # If "loader" is provided, then save the data
        nds = DataSet(name)
        nds.picklepath = picklepath

        if 'specific_data' in kwargs:
            nds.data = kwargs['specific_data']
            if 'loader' in kwargs:
                nds.save_data(kwargs['loader'], overwrite={name : True})

        if 'misc' in kwargs:
            nds.misc = kwargs['misc']

        self.data_dict[name] = nds

    def add_Qs_data(self, Qs_picklepath, **kwargs):
        # See _make_new_dataset for kwargs
        self._make_new_dataset('Qs-primary', Qs_picklepath, kwargs)

    def add_Qs_secondary_data(self, Qs_picklepath, **kwargs):
        # See _make_new_dataset for kwargs
        self._make_new_dataset('Qs-secondary', Qs_picklepath, kwargs)

    def add_gsd_data(self, gsd_picklepath, **kwargs):
        # "all_data" is all the data and will need to be sliced
        # "generate_path" indicates that the picklepath provided is actually 
        # the pickle directory and a path need to be generated.
        # See _make_new_dataset for more kwargs
        #
        # return the name if failed, otherwise return ''
        if 'all_data' in kwargs and 'specific_data' not in kwargs:
            # Get values to select from multiindex
            exp_code = self.exp_code
            period = self.period_end
            step = self.step
            length = '8m' if period == 't60' else '2m'

            idx = pd.IndexSlice
            index_slicer = idx[exp_code, step, period, :, length]
            col_slicer = idx[:]
            all_data = kwargs['all_data']
            try:
                kwargs['specific_data'] = all_data.loc[index_slicer, col_slicer]
            except KeyError:
                #print(period)
                #print(gsd_picklepath)
                #print(index_slicer)
                #print(col_slicer)
                #print(f"{exp_code}-{step}-{period}-{length}")
                #print(all_data)
                #assert(False)
                #print(all_data.loc[index_slicer, col_slicer])
                return f"{exp_code}-{step}-{period}-{length}"

        if 'generate_path' in kwargs:
            # Picklepath is actually the pickle dir
            pkl_filename = f"{exp_code}-{step}-{period}_gsd.pkl"
            gsd_picklepath = pjoin(gsd_picklepath, pkl_filename)
        self._make_new_dataset('gsd', gsd_picklepath, kwargs)
        return ''

    def add_depth_data(self, depth_picklepath, **kwargs):
        if 'all_data' in kwargs and 'specific_data' not in kwargs:
            # Get values to select from multiindex
            exp_code = self.exp_code
            period = self.period_end
            time = int(self.period_end[1:]) - 10
            step = self.step
            flow = self.discharge_int
            limb = self.limb

            idx = pd.IndexSlice
            index_slicer = idx[limb, flow, time, :]
            col_slicer = idx[:]
            all_data = kwargs['all_data']
            #print(f"{exp_code}-{step}-t{time}_flow-depth")
            #print(all_data)
            #print(all_data.loc[index_slicer, col_slicer])
            #assert(False)
            try:
                kwargs['specific_data'] = all_data.loc[index_slicer, col_slicer]
            except KeyError:
                return f"{exp_code}-{step}-t{time}_flow-depth"
        if 'generate_path' in kwargs:
            # Picklepath is actually the pickle dir
            pkl_filename = f"{exp_code}-{step}-t{time}_flow-depth.pkl"
            depth_picklepath = pjoin(depth_picklepath, pkl_filename)

        self._make_new_dataset('depth', depth_picklepath, kwargs)
        return ''

    def add_dem_data(self, dem_picklepath, **kwargs):
        exp_code = self.exp_code
        step = self.step
        time = self.period_end
        length = '8m' if self.period_end == 't60' else '2m'
        period_info_str = f"{exp_code}-{step}-{time}-{length}"

        if 'all_data' in kwargs and 'specific_data' not in kwargs:
            # Get values to select from all_data dict {period_info : dem 
            # npArray}
            all_data = kwargs['all_data']

            if period_info_str in all_data:
                kwargs['specific_data'] = all_data[period_info_str]
            else:
                return period_info_str

        if 'generate_path' in kwargs:
            # Picklepath is actually the pickle dir
            pkl_filename = f"{period_info_str}_dem.pkl"
            dem_picklepath = pjoin(dem_picklepath, pkl_filename)

        self._make_new_dataset('dem', dem_picklepath, kwargs)
        return ''


    # Reloading
    def reload_Qs_data(self, loader):
        # Loads the most up to date version of the Qs pickle
        self._Qs_dataset.reload_data(loader)
        #print(self._Qs_dataset.picklepath)
        #assert(False)

    def reload_gsd_data(self, loader):
        # Loads the gsd data
        if 'gsd' in self.data_dict:
            self.data_dict['gsd'].reload_data(loader)

    def reload_depth_data(self, loader):
        # Loads the depth data
        if 'depth' in self.data_dict:
            self.data_dict['depth'].reload_data(loader)

    def reload_dem_data(self, loader):
        # Loads the dem data
        if 'dem' in self.data_dict:
            self.data_dict['dem'].reload_data(loader)

    
    # Attributes
    @property
    def _Qs_dataset(self):
        # provides the most up to date Qs DataSet
        if 'Qs-secondary' in self.data_dict:
            return self.data_dict['Qs-secondary']
        else:
            return self.data_dict['Qs-primary']

    @property
    def Qs_picklepath(self):
        # provides the path for the most up to date version of the Qs pickle
        # (ie. primary vs secondary processed)
        return self._Qs_dataset.picklepath

    @property
    def Qs_data(self):
        return self._Qs_dataset.data

    @Qs_data.setter
    def Qs_data(self, data):
        self._Qs_dataset.data = data

    @property
    def gsd_data(self):
        try:
            return self.data_dict['gsd'].data
        except KeyError:
            return None

    @property
    def depth_data(self):
        try:
            return self.data_dict['depth'].data
        except KeyError:
            return None

    @property
    def dem_data(self):
        try:
            return self.data_dict['dem'].data
        except KeyError:
            return None


    # Processing
    def apply_period_function(self, fu, kwargs):
        # Call fu with self as an argument
        fu(self, kwargs)


class Experiment:

    code_name_dict = {
            '1A' : 'No Feed',
            '1B' : 'No Feed',
            '2A' : 'Constant Feed',
            '2B' : 'Constant Feed',
            '3A' : 'Rising Limb Feed',
            '3B' : 'Rising Limb Feed',
            '4A' : 'Falling Limb Feed',
            '5A' : 'Capacity Feed',
            }

    """ Experiment is a quasi-container class representing all the data associated with one feed scenario. Functions here should be more general purpose. Experiment pickles (via PeriodData objects) are not intended to contain the actual data. However, they will have paths to the data pickles and functions to reload that data. """

    # Creation
    def __init__(self, experiment_code):
        self.name = Experiment.code_name_dict[experiment_code]
        self.code = experiment_code
        self.periods = {} # { rank : PeriodData}
        self.sorted_ranks = []
        self.accumulated_data = None
        self.init_time = asctime() # asctime of last instance

    def from_existing(old_experiment, logger):
        # Create a new experiment from an existing one. Returns the new 
        # experiment object. Used for updating the tree to new definitions of 
        # Experiment and PeriodData classes (ie. changes to this file)
        oe = old_experiment
        ne = Experiment(oe.code)
        logger.write(f"Updating exp {oe.code} from {oe.init_time} to {ne.init_time}")
        for rank, op in oe.periods.items():
            #print(f"period {op.exp_code}: {op.gsd_picklepath}")
            ne.periods[rank] = PeriodData.from_existing(op)
        ne.sort_ranks()

        return ne

    def add_period_data(self, period_data):
        # Store a new PeriodData object in the self.periods dict
        # Complains if an object already exists
        assert(period_data.rank not in self.periods)
        self.periods[period_data.rank] = period_data

    def sort_ranks(self):
        # Sort the ranks. Likely won't need to call this more than once, but 
        # just in case
        self.sorted_ranks = [ r for r in self.periods.keys()]
        self.sorted_ranks.sort()


    # Adding new data sets
    def add_generic_data(self, period_fu, **kwargs):
        # Add an arbitrary dataset to the PeriodData objects
        # period_fu should be one of the add_*_data functions from PeriodData
        # kwargs are passed directly to the period_fu
        failed_list = []
        for period_data in self.periods.values():
            failed = period_fu(period_data, **kwargs)
            if failed:
                failed_list.append(failed)
        return failed_list

    def add_gsd_data(self, gsd_pickledir, gsd_data, generate_path=True):
        # Passes the gsd_data to each period for extraction
        failed_list = []
        for period_data in self.periods.values():
            failed = period_data.add_gsd_data(gsd_pickledir,
                    all_data=gsd_data, generate_path=True)
            if failed:
                failed_list.append(failed)
        return failed_list

    def add_depth_data(self, depth_pickledir, depth_data):
        # Passes the depth_data to each period for extraction
        failed_list = []
        for period_data in self.periods.values():
            failed = period_data.add_depth_data(depth_pickledir,
                    all_data=depth_data[self.code], generate_path=True)
            if failed:
                failed_list.append(failed)
        return failed_list

    def add_dem_data(self, dem_pickledir, dem_data):
        period_fu = PeriodData.add_dem_data
        kwargs = {'dem_picklepath': dem_pickledir,
                  'all_data'      : dem_data[self.code],
                  'generate_path' : True}
        return self.add_generic_data(period_fu, **kwargs)

    
    # Processing
    def apply_period_function(self, fu, kwargs={}):
        # Pass a function on to the period data.
        # period data will call the function and provide itself as an argument 
        # then non-expanded kwargs.
        for period_data in self.periods.values():
            period_data.apply_period_function(fu, kwargs)

    def accumulate_Qs_data(self, kwargs):
        # Accumulate the data for all the periods in this experiment.  
        # Simplifies subsequent functions (like rolling averages).
        # Note: this function won't work on Qs-primary data because 
        # 'exp_time_hrs' has not been created. KeyError is handled in calling 
        # function (in omnipickle).
        self.apply_period_function(self._accumulate_Qs_data, kwargs)
        #print(self.accumulated_data)
        #assert(False)
        
        # Must sort data or many things will not make sense
        self.accumulated_data.sort_values(by='exp_time_hrs', inplace=True)

    def _accumulate_Qs_data(self, period_data, kwargs):
        if 'check_ignored_fu' in kwargs:
            check_ignore = kwargs['check_ignored_fu']
            if check_ignore(period_data):
                return

        if 'cols_to_keep' in kwargs:
            columns_to_keep = accumulate_Qs_kwargs['cols_to_keep']
            data = period_data.Qs_data.loc[:, columns_to_keep]
        else:
            data = period_data.Qs_data

        #print(period_data._Qs_dataset)
        #print(data)
        #assert(False)
        if self.accumulated_data is None:
            self.accumulated_data = data
        else:
            self.accumulated_data = self.accumulated_data.append(data)

    # Saving
    def save_data(self, loader, overwrite={}):
        # Tell periods to (over)write pickles for the data
        # Likely won't be used unless you are updating the pickles
        for period_data in self.periods.values():
            period_data.save_data(loader, overwrite=overwrite)

    def wipe_data(self):
        # So I can pickle experiment objects without repickling the data
        # Likely won't be used unless you are updating the experiment pickle
        for period_data in self.periods.values():
            period_data.wipe_data()
        self.accumulated_data = None

    
    # Loading
    def reload_Qs_data(self, loader):
        # Load Qs data from pickles. PeriodData objects should already have 
        # individual picklepaths to load.
        for period_data in self.periods.values():
            period_data.reload_Qs_data(loader)

    def reload_gsd_data(self, loader):
        # Load gsd data from pickles.
        for period_data in self.periods.values():
            period_data.reload_gsd_data(loader)

    def reload_depth_data(self, loader):
        # Load depth data from pickles.
        for period_data in self.periods.values():
            period_data.reload_depth_data(loader)

    def reload_dem_data(self, loader):
        # Load dem data from pickles.
        for period_data in self.periods.values():
            period_data.reload_dem_data(loader)


