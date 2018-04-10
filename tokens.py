#!/usr/bin/env python3

from os.path import join as pjoin
from time import asctime

# From Helpyr
from helpyr_misc import nsplit

# These classes were originally coded for the Qs secondary processor, but have 
# been extended to elsewhere. Therefore, they have some legacy functions that 
# should be cleaned up eventually to fit the current framework.

# To handle more and more data, perhaps change PeriodData to store a dictionary 
# of DataObjects (have to define a new token)?

class PeriodData:
    # The PeriodData class is meant as a simple container to store all the 
    # information associated with each period.
    # Data includes measurements that happened during or at the end of the 
    # period

    """ PeriodData is a quasi-container class representing the data associated with one period. Functions here should be more general purpose. PeriodData pickles are not intended to contain the actual data. However, they will have paths to the data pickles and functions to reload that data. """

    period_ranking = {"rising" : 0, "falling" : 1,
            "50L" : 00, "62L" : 10, "75L" : 20, "87L" : 30, "100L" : 40,
            "t00-t10" : 0, "t10-t20" : 1, "t00-t20" : 2,
            "t20-t30" : 3, "t30-t40" : 4, "t20-t40" : 5,
            "t40-t50" : 6, "t50-t60" : 7, "t40-t60" : 8, "t00-t60" : 9,
                      }
        
    # Used for creation or updating
    def __init__(self, Qs_picklepath):
        # Set up Qs
        self.Qs_primary_picklepath = Qs_picklepath
        self.Qs_secondary_picklepath = None
        self.Qs_pkl_name = nsplit(Qs_picklepath, 1)[1].split('.',1)[0]
        _, self.exp_code, self.step, self.period = self.Qs_pkl_name.split('_')
        self.Qs_data = None

        # Set up gsd
        self.gsd_picklepath = None
        self.gsd_data = None
        
        # Calculate period ranks
        self.limb, self.discharge = self.step.split('-')
        self.discharge_int = int(self.discharge[:-1])
        ranking = PeriodData.period_ranking
        get_ranking = lambda l, d, p: d + p + 2*l*(40-d)
        self.rank = get_ranking(*[ranking[k] for k in
                                  (self.limb, self.discharge, self.period)])

    def from_existing(old_period):
        # Create a new PeriodData obj from an existing one. Returns the new 
        # PeriodData object. Used for updating the tree to new definitions of 
        # Experiment and PeriodData classes (ie. changes to this file)
        op = old_period
        np = PeriodData(op.Qs_primary_picklepath)
        np.Qs_secondary_picklepath = op.Qs_secondary_picklepath 
        np.gsd_picklepath = op.gsd_picklepath 
        #print(op.gsd_picklepath)
        #print(np.gsd_picklepath)
        #assert(False)

        return np

    def save_data(self, loader):
        # Save data prior to wiping. Only really needed when building or 
        # updating the omnipickle. (ie. for the first time the data is added to 
        # the omnipickle and isn't already pickled)
        path_data_pairs = [
                (self.Qs_picklepath             , self.Qs_data),
                (self.gsd_picklepath            , self.gsd_data),
                ]
        data_to_save = {}
        for path, data  in path_data_pairs:
            if data is not None and path is not None:
                data_to_save[path] = data

        loader.produce_pickles(data_to_save, add_path=False)

    def wipe_data(self):
        # So I can save the omnipickle/experiment/period objects without 
        # repickling the underlying data
        self.Qs_data = None
        self.gsd_data = None

    def add_gsd_data(self, gsd_pickledir, gsd_data):
        # Extract the proper set of gsd_data if exists
        # gsd_data has all the data and uses a multiindex to keep them separate
        # return the name if failed (keep?)

        # Get values to select from multiindex
        exp_code = self.exp_code
        period = self.period.split('-')[-1] # grab the second time str
        limb, flow = self.step.split('-')
        step = f"{limb[0]}{flow}"
        length = '8m' if period == 't60' else '2m'

        try:
            pkl_filename = f"{exp_code}-{step}-{period}_gsd.pkl"
            self.gsd_data = gsd_data.loc[exp_code, step, period, :, length]
            self.gsd_picklepath = pjoin(gsd_pickledir, pkl_filename)
        except KeyError:
            name = f"{exp_code}-{step}-{period}-{length}"
            return name

        return ''


    # Attributes
    @property
    def Qs_picklepath(self):
        # provides the path for the most up to date version of the Qs pickle
        # (ie. primary vs secondary processed)
        primary = self.Qs_primary_picklepath
        secondary = self.Qs_secondary_picklepath
        return primary if secondary is None else secondary


    # Reloading
    def reload_Qs_data(self, loader):
        # Loads the most up to date version of the Qs pickle
        self.Qs_data = loader.load_pickle(self.Qs_picklepath, add_path=False)

    def reload_gsd_data(self, loader):
        # Loads the gsd data
        gsd_path = self.gsd_picklepath
        if gsd_path is not None:
            self.gsd_data = loader.load_pickle(gsd_path, add_path=False)

    
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

    def add_gsd_data(self, gsd_pickledir, gsd_data):
        # Passes the gsd_data to each period for extraction
        failed_list = []
        for period_data in self.periods.values():
            failed = period_data.add_gsd_data(gsd_pickledir,gsd_data)
            if failed:
                failed_list.append(failed)
        return failed_list


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
        self.apply_period_function(self._accumulate_Qs_data, kwargs)
        
        # Must sort data or many things will not make sense
        self.accumulated_data.sort_values(by='exp_time_hrs', inplace=True)

    def _accumulate_Qs_data(self, period_data, kwargs):
        accumulate_Qs_kwargs = kwargs['accumulate_Qs_kwargs']
        check_ignore = accumulate_Qs_kwargs['check_ignored_fu']
        if check_ignore(period_data):
            return

        if 'cols_to_keep' in kwargs:
            columns_to_keep = accumulate_Qs_kwargs['cols_to_keep']
            data = period_data.Qs_data.loc[:, columns_to_keep]
        else:
            data = period_data.Qs_data

        if self.accumulated_data is None:
            self.accumulated_data = data
        else:
            self.accumulated_data = self.accumulated_data.append(data)

    def _accumulate_photo_data(self, period_data, kwargs):
        pass


    # Saving
    def save_data(self, loader):
        # Tell periods to (over)write pickles for the data
        # Likely won't be used unless you are updating the pickles
        for period_data in self.periods.values():
            period_data.save_data(loader)

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


