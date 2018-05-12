#!/usr/bin/env python3

import os

from tokens import PeriodData, Experiment
from data_loading import DataLoader
import global_settings as settings

# This class is meant to manage the creation and updating of the omnipickle.  
# The omnipickle contains an experiment - period data tree of metadata 
# surrounding the data. The pickled version has paths to the data, but not the 
# actual data.  The data can be loaded from the saved paths
#
# In other words, the omnipickle knows where all the data is and can extract 
# whatever data is needed for the current task.
#
# Originally, the Experiment tree was developed for the Qs processor and Qs 
# data, but I'm trying to extend it to hand all the data by writing the 
# OmnipickleManager.  Could probably use some cleaning up. In particular, the 
# user must run the secondary processor to create the omnipickle. I would like 
# for the omnipickle frame setup to be independent from the secondary 
# processor.
#
# The omnipickle is above the root of a data type directory (eg. manual, cart, 
# or lighttable) and therefore defines it's own path in the constructor.

class OmnipickleManager:
    def __init__(self, logger):
        self.logger = logger
        self.omnipickle_name = "omnipickle"
        self.omnipickle_path = f"/home/alex/feed-timing/data/{self.omnipickle_name}.pkl"
        self.experiments = {} # {exp_code : Experiment}
        self.omniloader = DataLoader(settings.root_dir, logger=logger)

    # Data storage functions
    def store(self):
        # Preparing to save the omnipickle
        # Save and clear out data from the omnipickle.
        # Save the experiment/period tree.
        self.logger.write("Saving data")
        for experiment in self.experiments.values():
            experiment.save_data(self.omniloader)
            experiment.wipe_data()

        # Save the experiment tree.
        # Cheating a little on the saving of the omnipickle. Don't have to save 
        # self or the logger or the loader.
        self.logger.write("Saving omnipickle")
        self.omniloader.produce_pickles({self.omnipickle_path : self.experiments},
                add_path=False)

    def restore(self):
        # Reload the experiment tree containing metadata. Does NOT load the 
        # actual data.
        self.experiments = self.omniloader.load_pickle(self.omnipickle_path,
                                              add_path = False)

    def update_tree_definitions(self):
        # if you change the tokens.py file, you must recreate the tree for the 
        # changes to take effect
        #
        for exp_code, old_experiment in self.experiments.items():
            self.experiments[exp_code] = Experiment.from_existing(old_experiment, self.logger)

        self.store() # save the new tree

    def reload_Qs_data(self, kwargs={}):
        # reload Qs data
        for experiment in self.experiments.values():
            experiment.reload_Qs_data(self.omniloader)
            experiment.accumulate_Qs_data(kwargs)

    def reload_gsd_data(self):
        # reload gsd data
        for experiment in self.experiments.values():
            experiment.reload_gsd_data(self.omniloader)

    def reload_depth_data(self):
        # reload depth data
        for experiment in self.experiments.values():
            experiment.reload_depth_data(self.omniloader)


    # Used by the Qs secondary processor.
    def Qs_build_experiment_tree(self, qs_picklepaths):
        # Use the Qs pickle filepaths from the secondary light table processor 
        # to build the experiment tree
        for picklepath in qs_picklepaths:
            # Create PeriodData objects and hand off to Experiment objects
            period_data = PeriodData(picklepath)

            exp_code = period_data.exp_code
            if exp_code not in self.experiments:
                self.experiments[exp_code] = Experiment(exp_code)

            self.experiments[exp_code].add_period_data(period_data)

        for experiment in self.experiments.values():
            experiment.sort_ranks()

    def Qs_accumulate_time(self, accumulate_fu):
        # Create a sequential timestamp for each period based on lighttable 
        # data
        for experiment in self.experiments.values():
            accumulate_fu(experiment)

    def Qs_finish_secondary_pickles(self, pickle_destination):
        # Creates the first version of the omnipickle
        self.logger.write("Updating secondary Qs data")
        for experiment in self.experiments.values():
            experiment.apply_period_function(self._update_Qs_picklepath,
                     kwargs={'pkl_dest' : pickle_destination})

        self.store()

    def _update_Qs_picklepath(self, period, kwargs):
        # Saves the second processes Qs data and updates the 
        # secondary_picklepath for each period
        destination = kwargs['pkl_dest']
        picklepath = os.path.join(destination, period.Qs_pkl_name + '.pkl')
        path = self.omniloader.produce_pickles({picklepath : period.Qs_data},
                add_path=False, verbose=False)[0]
        period.Qs_secondary_picklepath = path

    
    # Used by the Qs grapher (will be superseded by universal grapher)
    def accumulate_Qs_data(self, accumulate_kwargs):
        for experiment in self.experiments.values():
            experiment.accumulate_Qs_data(
                    {'accumulate_Qs_kwargs':accumulate_kwargs})

    def apply_to_periods(self, fu, kwargs={}):
        # Pass a function on to the period data.
        # period data will call the function and provide itself as an argument 
        # then non-expanded kwargs.
        for experiment in self.experiments.values():
            experiment.apply_period_function(fu, kwargs)


    # Used by gsd_processor
    def add_gsd_data(self, gsd_pickledir, gsd_data):
        # Add the gsd data to each experiment for extraction.
        # gsd_data uses a multiindex to separate data
        write = self.logger.write
        for experiment in self.experiments.values():
            not_found = experiment.add_gsd_data(gsd_pickledir, gsd_data)
            write(f"Experiment {experiment.code} could not find gsd data for:")
            write(not_found, local_indent=1)

    
    # Used by manual_processor
    def add_depth_data(self, depth_pickledir, depth_data):
        # Add the gsd data to each experiment for extraction.
        # depth_data uses a multiindex to separate data
        write = self.logger.write
        for experiment in self.experiments.values():
            not_found = experiment.add_depth_data(depth_pickledir, depth_data)
            write(f"Experiment {experiment.code} could not find depth data for:")
            write(not_found, local_indent=1)

    
    # Attributes
    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, new_val):
        self._experiments = new_val

