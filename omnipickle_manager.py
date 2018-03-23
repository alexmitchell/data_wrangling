#!/usr/bin/env python3

from tokens import PeriodData, Experiment

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

class OmnipickleManager:
    def __init__(self):
        self.omnipickle_name = "omnipickle"
        self.omnipickle_path = f"/home/alex/feed-timing/data/{self.omnipickle_name}.pkl"
        self.experiments = {} # {exp_code : Experiment}

    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, new_val):
        self._experiments = new_val


    # Used by the Qs secondary processor.
    def Qs_load_pickle_info(self, qs_picklepaths):
        # Use the Qs pickle filepaths from the secondary light table processor 
        # to flesh out the experiment tree
        for picklepath in qs_picklepaths:
            # Create PeriodData objects and hand off to Experiment objects
            period_data = PeriodData(picklepath)

            exp_code = period_data.exp_code
            if exp_code not in self.experiments:
                self.experiments[exp_code] = Experiment(exp_code)

            self.experiments[exp_code].save_period_data(period_data)

        for experiment in self.experiments.values():
            experiment.sort_ranks()

    def Qs_load_data(self, loader):
        # load Qs data into the tree
        for experiment in self.experiments.values():
            experiment.load_Qs_data(loader)

    def Qs_accumulate_time(self, accumulate_fu):
        # Add the different time chunks together
        for experiment in self.experiments.values():
            accumulate_fu(experiment)

    def Qs_produce_pickles(self, loader):
        # Creates the first version of the omnipickle
        for experiment in self.experiments.values():
            # Save the data into pickles
            experiment.produce_pickles(loader)

            # Clear out the data from the omnipickle for saving
            experiment.wipe_data()

        # Save the omnipickle
        picklepaths = loader.produce_pickles(
                {self.omnipickle_path: self.experiments}, add_path=False)

    
    # Used by the Qs grapher (will be superseded by universal grapher)
    def reload_omnipickle(self, loader):
        self.experiments = loader.load_pickle(self.omnipickle_path,
                                              add_path = False)

    def reload_Qs(self, loader, accumulate_kwargs):
        for experiment in self.experiments.values():
            experiment.load_Qs_data(loader)
            experiment.accumulate_Qs_data(
                    {'accumulate_Qs_kwargs':accumulate_kwargs})

    def wipe_data(self):
        # Clear out the data from the omnipickle, keep the meta info though
        for experiment in self.experiments.values():
            experiment.wipe_data()
        self.experiments = {}

