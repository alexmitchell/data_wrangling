#!/usr/bin/env python3

# From Helpyr
from helpyr_misc import nsplit

class PeriodData:
    # The PeriodData class is meant as a simple container to store all the 
    # information associated with each period

    """ PeriodData is a quasi-container class representing the data associated with one period. Functions here should be more general purpose. PeriodData pickles are not intended to contain the actual data. However, they will have paths to the data pickles and functions to reload that data. """

    period_ranking = {"rising" : 0, "falling" : 1,
            "50L" : 00, "62L" : 10, "75L" : 20, "87L" : 30, "100L" : 40,
            "t00-t10" : 0, "t10-t20" : 1, "t00-t20" : 2,
            "t20-t30" : 3, "t30-t40" : 4, "t20-t40" : 5,
            "t40-t50" : 6, "t50-t60" : 7, "t40-t60" : 8, "t00-t60" : 9,
                      }
        
    def __init__(self, picklepath):
        self.primary_picklepath = picklepath
        self.secondary_picklepath = None
        self.pkl_name = nsplit(picklepath, 1)[1].split('.',1)[0]
        _, self.exp_code, self.step, self.period = self.pkl_name.split('_')
        
        # Calculate period ranks
        self.limb, self.discharge = self.step.split('-')
        self.discharge_int = int(self.discharge[:-1])
        ranking = PeriodData.period_ranking
        get_ranking = lambda l, d, p: d + p + 2*l*(40-d)
        self.rank = get_ranking(*[ranking[k] for k in
                                  (self.limb, self.discharge, self.period)])

        self.data = None

    @property
    def picklepath(self):
        # provides the path for the most up to date version of the pickle
        primary = self.primary_picklepath
        secondary = self.secondary_picklepath
        return primary if secondary is None else secondary

    def load_data(self, loader):
        # Loads the most up to date version of the pickle
        self.data = loader.load_pickle(self.pkl_name)

    def produce_period_pickle(self, loader):
        # New pickle created
        path = loader.produce_pickles({self.pkl_name: self.data})[0]
        self.secondary_picklepath = path

    def wipe_data(self):
        # So I can save experiment objects without repickling the data
        self.data = None

    def apply_period_function(self, fu, kwargs):
        fu(self, kwargs)


class Experiment:

    """ Experiment is a quasi-container class representing all the data associated with one feed scenario. Functions here should be more general purpose. Experiment pickles (via PeriodData objects) are not intended to contain the actual data. However, they will have paths to the data pickles and functions to reload that data. """

    def __init__(self, experiment_code):
        self.code = experiment_code
        self.periods = {} # { rank : PeriodData}
        self.sorted_ranks = []
        self.accumulated_data = None

    def save_period_data(self, period_data):
        # Store a new PeriodData object in the self.periods dict
        # Complains if an object already exists
        assert(period_data.rank not in self.periods)
        self.periods[period_data.rank] = period_data

    def sort_ranks(self):
        # Sort the ranks. Likely won't need to call this more than once, but 
        # just in case
        self.sorted_ranks = [ r for r in self.periods.keys()]
        self.sorted_ranks.sort()

    def load_data(self, loader):
        # Load data from pickles. PeriodData objects should already have 
        # individual picklepaths to load.
        for period_data in self.periods.values():
            period_data.load_data(loader)

    def accumulate_data(self, kwargs):
        # Accumulate the data for all the periods in this experiment.  
        # Simplifies subsequent functions (like rolling averages).
        self.apply_period_function(self._accumulate_data, kwargs)
        
        # Must sort data or many things will not make sense
        self.accumulated_data.sort_values(by='exp_time_hrs', inplace=True)

    def _accumulate_data(self, period_data, kwargs):
        accumulate_kwargs = kwargs['accumulate_kwargs']
        check_ignore = accumulate_kwargs['check_ignored_fu']
        if check_ignore(period_data):
            return

        if 'cols_to_keep' in kwargs:
            columns_to_keep = accumulate_kwargs['cols_to_keep']
            data = period_data.data.loc[:, columns_to_keep]
        else:
            data = period_data.data

        if self.accumulated_data is None:
            self.accumulated_data = data
        else:
            self.accumulated_data = self.accumulated_data.append(data)

    def produce_pickles(self, loader):
        # Tell periods to (over)write pickles for the data
        # Likely won't be used unless you are updating the pickles
        for period_data in self.periods.values():
            period_data.produce_period_pickle(loader)

    def wipe_data(self):
        # So I can pickle experiment objects without repickling the data
        # Likely won't be used unless you are updating the experiment pickle
        for period_data in self.periods.values():
            period_data.wipe_data()

    def apply_period_function(self, fu, kwargs):
        # Pass a function on to the period data.
        # period data will call the function and provide itself as an argument 
        # then non-expanded kwargs.
        for period_data in self.periods.values():
            period_data.apply_period_function(fu, kwargs)

