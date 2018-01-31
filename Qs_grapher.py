#!/usr/bin/env python3

from os.path import join as ospath_join
from time import asctime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# From Helpyr
from helpyr_misc import *
from logger import Logger
from data_loading import DataLoader

# To do:
# - ignore 50L step

# For reference
Qs_column_names = [
        # Timing and meta data
        #'elapsed-time sec', <- Calculate this column later
        'timestamp', 'missing ratio', 'vel', 'sd vel', 'number vel',
        'exp_time',
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


class QsGrapher:

    def __init__(self, root_dir):
        # File locations
        self.root_dir = root_dir
        self.pickle_source = f"{self.root_dir}/secondary-processed-pickles"
        self.log_filepath = "./log-files/Qs_grapher.txt"
        self.figure_destination = f"{self.root_dir}/prelim-figures"
        self.exp_metapickle = "experiments_metapickle"
        self.experiments = {}
        
        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("Qs Grapher")

        # Start up loader
        self.loader = DataLoader(self.pickle_source, logger=self.logger)

        ensure_dir_exists(self.figure_destination)

    def make_experiment_plots(self):
        self.logger.write(["Making experiment plots..."])

        self.experiments = {}
        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_full_exp,
                before_msg="Plotting full experiment data",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def load_data(self):
        self.experiments = self.loader.load_pickle(self.exp_metapickle)
        for experiment in self.experiments.values():
            experiment.load_data(self.loader)

    def plot_full_exp(self):
        columns_to_plot = ['Bedload all']

        #fig = plt.figure()
        #ax = plt.axes()
        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))

        figure_name = "bedload_all_hydrograph_20min.png"
        plot_kwargs = {
                'x'    : 'exp_time_hrs',
                'y'    : 'Bedload all',
                'kind' : 'scatter',
                'logy' : True,
                'xlim' : (0.5, 8.5),
                'ylim' : (0.001, 5000),
                }
        rolling_kwargs = {
                'window'      : 60*20, # seconds
                'min_periods' : 40,
                #'on'          : plot_kwargs['x'],
                }
        kwargs = {'plot_kwargs'    : plot_kwargs,
                  'rolling_kwargs' : rolling_kwargs,
                  }

        exp_codes = list(self.experiments.keys())
        exp_codes.sort()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            plot_kwargs['ax'] = ax
            twax = ax.twinx()
            experiment = self.experiments[exp_code]

            self.rolling_av = None
            self.hydrograph = None

            self.logger.write(f"Plotting experiment {exp_code}")
            experiment.apply_period_function(self._plot_dataframe, kwargs)
            experiment.apply_period_function(self._plot_hydrograph, kwargs)

            self.hydrograph.sort_index(inplace=True)
            self.hydrograph.plot(ax=twax, style='g', ylim=(0,400))

            experiment.apply_period_function(self._plot_rolling, kwargs)
            series_plot_kwargs = {k : plot_kwargs[k] for k in plot_kwargs
                                        if k not in ['x', 'y', 'kind']}
            #series_plot_kwargs['xlim'] = ax.get_xlim()
            series_plot_kwargs['style'] = 'r'
            self.rolling_av.plot(**series_plot_kwargs)

            ax.set_title(f"Experiment {experiment.code}")
            #twax.set_ylabel('Discharge (L/s)')

        yrange = plot_kwargs['ylim']
        plt.suptitle(f"Total bedload output with logy scale {yrange} ({asctime()})")

        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')
        plt.show()

    def _plot_hydrograph(self, period_data, kwargs={}):
        if not self._check_ignore_period(period_data):
            plot_kwargs = kwargs['plot_kwargs']
            x = plot_kwargs['x']
            y = 'discharge'

            data = period_data.data
            x0 = data.loc[:,x].iloc[0]
            x1 = data.loc[:,x].iloc[-1]
            y0 = data.loc[:,y].iloc[0]
            y1 = data.loc[:,y].iloc[-1]

            discharge = pd.Series([y0,y1], index=[x0,x1])
            if self.hydrograph is None:
                self.hydrograph = discharge
            else:
                self.hydrograph = self.hydrograph.append(discharge, verify_integrity=True)

    def _plot_dataframe(self, period_data, kwargs={}):
        if not self._check_ignore_period(period_data):
            all_data = period_data.data
            all_data.plot(**kwargs['plot_kwargs'])

            # Too slow...
            #plot_kwargs = kwargs['plot_kwargs']
            #discharge_kwargs = {k : plot_kwargs[k] for k in plot_kwargs}
            #discharge_kwargs['y'] = 'discharge'
            #discharge_kwargs['kind'] = 'line'
            #discharge_kwargs['style'] = 'g'
            #discharge_kwargs['ax'] = plot_kwargs['ax'].twinx()
            #all_data.plot(**discharge_kwargs)

    def _plot_rolling(self, period_data, kwargs={}):
        if self._check_ignore_period(period_data):
            return

        all_data = period_data.data

        x = all_data.loc[:, kwargs['plot_kwargs']['x']]
        y = all_data.loc[:, kwargs['plot_kwargs']['y']]
        series = pd.Series(data=y.values, index=x)

        rolled = series.rolling(**kwargs['rolling_kwargs'])
        average = rolled.mean()

        if self.rolling_av is None:
            self.rolling_av = average
        else:
            self.rolling_av = self.rolling_av.append(average, verify_integrity=True)

        #plot_kwargs = kwargs['plot_kwargs']
        #kind = plot_kwargs['kind']
        #plot_kwargs['kind'] = 'line'
        #average.plot(**plot_kwargs)
        #plot_kwargs['kind'] = kind

    def _check_ignore_period(self, period_data):
        # Return True if period should be ignored
        return period_data.step in self.ignore_steps


if __name__ == "__main__":
    # Run the script
    root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
    grapher = QsGrapher(root_dir)
    grapher.make_experiment_plots()
