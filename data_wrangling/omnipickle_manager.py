#!/usr/bin/env python3

import os

from helpyr.data_loading import DataLoader

from tokens import PeriodData, Experiment
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
        self.omnipickle_name = settings.omnipickle_name
        self.omnipickle_path = settings.omnipickle_path
        self.experiments = {} # {exp_code : Experiment}
        self.omniloader = DataLoader(settings.root_dir, logger=logger)

    def get_exp_codes(self):
        exp_codes = list(self.experiments.keys())
        exp_codes.sort()
        return exp_codes

    # Data storage functions
    def store(self, overwrite={}):
        # Preparing to save the omnipickle
        # Save and clear out data from the omnipickle.
        # Save the experiment/period tree.
        # overwrite = {'dataset_name' : bool=False}
        self.logger.write("Storing Omnipickle Manager")
        self.logger.increase_global_indent()

        self.logger.write("Saving data")
        for experiment in self.experiments.values():
            experiment.save_data(self.omniloader, overwrite=overwrite)
            experiment.wipe_data()

        # Save the experiment tree.
        # Cheating a little on the saving of the omnipickle. Don't have to save 
        # self or the logger or the loader.
        self.logger.write("Saving omnipickle")
        self.omniloader.produce_pickles({self.omnipickle_path : self.experiments},
                add_path=False, overwrite=True)

        self.logger.decrease_global_indent()

    def restore(self):
        # Reload the experiment tree containing metadata. Does NOT load the 
        # actual data.
        self.logger.write("Restoring Omnipickle Manager")
        self.experiments = self.omniloader.load_pickle(self.omnipickle_path,
                                              add_path = False)

    def update_tree_definitions(self):
        # if you change the tokens.py file, you must recreate the tree for the 
        # changes to take effect
        self.logger.write("Updating data tree definitions")
        self.logger.increase_global_indent()

        for exp_code, old_experiment in self.experiments.items():
            self.experiments[exp_code] = Experiment.from_existing(old_experiment, self.logger)

        self.store() # save the new tree
        self.logger.decrease_global_indent()

    def _remove_datasets(self, names):
        # Remove datasets with the given names
        self.logger.write(["Removing datasets:"] + names)
        for experiment in self.experiments.values():
            experiment._remove_datasets(names)


    def reload_Qs_data(self, **acc_kwargs):
        acc_kwargs = {} if acc_kwargs is None else acc_kwargs
        # reload Qs data
        for experiment in self.experiments.values():
            experiment.reload_Qs_data(self.omniloader)
            try:
                experiment.accumulate_Qs_data(acc_kwargs)
            except KeyError:
                continue

    def reload_gsd_data(self):
        # reload gsd data
        for experiment in self.experiments.values():
            experiment.reload_gsd_data(self.omniloader)

    def reload_depth_data(self):
        # reload depth data
        for experiment in self.experiments.values():
            experiment.reload_depth_data(self.omniloader)

    def reload_dem_data(self):
        # reload dem data
        for experiment in self.experiments.values():
            experiment.reload_dem_data(self.omniloader)

    def reload_masses_data(self):
        # reload masses data
        for experiment in self.experiments.values():
            experiment.reload_masses_data(self.omniloader)

    def reload_sieve_data(self):
        # reload sieve data
        for experiment in self.experiments.values():
            experiment.reload_sieve_data(self.omniloader)

    def reload_feed_data(self):
        # reload feed data
        for experiment in self.experiments.values():
            experiment.reload_feed_data(self.omniloader)


    # Used by the Qs secondary processor.
    def manually_add_period(self, exp_code, limb, discharge, period_range, sort=True):
        # args like '3B', 'falling', '62L', 't20-t40'
        # Needed for periods that don't have Qs data.
        period_data = PeriodData.make_empty(
                exp_code, limb, discharge, period_range)

        if exp_code not in self.experiments:
            self.experiments[exp_code] = Experiment(exp_code)
        self.experiments[exp_code].add_period_data(period_data)
        if sort:
            self.experiments[exp_code].sort_ranks()

    def Qs_build_experiment_tree(self, qs_picklepaths):
        # Use the Qs pickle filepaths from the secondary light table processor 
        # to build the experiment tree
        for picklepath in qs_picklepaths:
            # Create PeriodData objects and hand off to Experiment objects
            period_data = PeriodData.from_Qs_picklepath(picklepath)

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
        raise NotImplementedError
        #self.logger.write("Updating secondary Qs data")
        #for experiment in self.experiments.values():
        #    experiment.apply_period_function(self._update_Qs_picklepath,
        #             kwargs={'pkl_dest' : pickle_destination})
        #self.store()

    def _update_Qs_picklepath(self, period, kwargs):
        raise NotImplementedError
        ## Saves the second processes Qs data and updates the 
        ## secondary_picklepath for each period
        #destination = kwargs['pkl_dest']
        #picklepath = os.path.join(destination, 
        #        period._Qs_dataset.misc['Qs_pkl_name'] + '.pkl')
        #path = self.omniloader.produce_pickles({picklepath : period.Qs_data},
        #        add_path=False, verbose=False)[0]
        #period.add_Qs_secondary_data(path)

    def generate_Qs_secondary_picklepath(self, period, pickle_destination):
        return os.path.join(pickle_destination, 
                period._Qs_dataset.misc['Qs_pkl_name'] + '.pkl')

    
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


    # Used by gsd, manual, or dem processors
    def add_generic_data(self, add_fu, pickledir, data, name=''):
        # Add arbitrary data to each experiment for extraction.
        # add_fu should be one of the add_*_data functions from Experiment
        # data should be all of the data for this type type (probably a 
        # dataframe with multiindex to separate it)
        write = self.logger.write
        name = name + ' ' if name else ''
        for experiment in self.experiments.values():
            not_found = add_fu(experiment, pickledir, data)
            write(f"Experiment {experiment.code} could not find {name}data for:")
            write(not_found, local_indent=1)

    def add_gsd_data(self, gsd_pickledir, gsd_data):
        # Add the gsd data to each experiment for extraction.
        # gsd_data uses a multiindex to separate data
        write = self.logger.write
        for experiment in self.experiments.values():
            not_found = experiment.add_gsd_data(gsd_pickledir, gsd_data)
            write(f"Experiment {experiment.code} could not find gsd data for:")
            write(not_found, local_indent=1)

    def add_depth_data(self, depth_pickledir, depth_data):
        # Add the gsd data to each experiment for extraction.
        # depth_data uses a multiindex to separate data
        write = self.logger.write
        for experiment in self.experiments.values():
            not_found = experiment.add_depth_data(depth_pickledir, depth_data)
            write(f"Experiment {experiment.code} could not find depth data for:")
            write(not_found, local_indent=1)

    def add_dem_data(self, dem_pickledir, dem_data):
        # Add the dem data to each experiment for extraction.
        # dem_data is an np array. one array per hour at end of step
        add_dem_fu = Experiment.add_dem_data
        name = 'dem'
        self.add_generic_data(add_dem_fu, dem_pickledir, dem_data, name)

    def add_masses_data(self, masses_pickledir, masses_data):
        # Add the masses data to each experiment for extraction.
        add_masses_fu = Experiment.add_masses_data
        name = 'masses'
        self.add_generic_data(add_masses_fu, masses_pickledir, masses_data, name)

    def add_sieve_data(self, sieve_pickledir, sieve_data):
        # Add the sieve data to each experiment for extraction.
        add_sieve_fu = Experiment.add_sieve_data
        name = 'sieve'
        self.add_generic_data(add_sieve_fu, sieve_pickledir, sieve_data, name)

    def add_feed_data(self, feed_pickledir, feed_data):
        # Add the feed data to each experiment for extraction.
        add_feed_fu = Experiment.add_feed_data
        name = 'feed'
        self.add_generic_data(add_feed_fu, feed_pickledir, feed_data, name)

    
    # Attributes
    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, new_val):
        self._experiments = new_val

