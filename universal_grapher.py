#!/usr/bin/env python3

from os.path import join as ospath_join
from time import asctime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# From Helpyr
from helpyr_misc import *
from logger import Logger
from data_loading import DataLoader

from omnipickle_manager import OmnipickleManager
import global_settings as settings

# To do:
# - Remove outliers

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

gsd_column_names = [
        # Stats
        'Sigmag', 'Dg', 'La', 'D90', 'D50', 'D10', 'Fsx',
        # Grain Size Fractions (counts)
        '0.5', '0.71', '1', '1.4', '2', '2.8', '4',
        '5.6', '8', '11.3', '16', '22.6', '32',
        # Scan name (ex: 3B-f75L-t60-8m )
        'scan_name',
        ]


class UniversalGrapher:
    # Based on the Qs_grapher, but extended to graph other data as well.

    # General functions
    def __init__(self):
        # File locations
        self.log_filepath = f"{settings.log_dir}/universal_grapher.txt"

        # Start up logger
        self.logger = Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("Universal Grapher")

        # Start up loader
        #self.loader = DataLoader(self.Qs_pickle_source, logger=self.logger)

        # omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()

        self.figure_destination = settings.figure_destination
        ensure_dir_exists(self.figure_destination)

        self.ignore_steps = []

    # General use functions
    def load_Qs_data(self):
        #self.experiments = self.loader.load_pickle(self.exp_omnipickle)
        accumulate_kwargs = {
                'check_ignored_fu' : self._check_ignore_period,
                }
        self.omnimanager.reload_Qs(self.loader, accumulate_kwargs)

    def roll_data(self, data, kwargs={}):
        # Make rolling averages on the provided data.

        time = data.loc[:, 'exp_time_hrs']
        y = data.loc[:, kwargs['plot_kwargs']['y']]
        series = pd.Series(data=y.values, index=time)

        rolled = series.rolling(**kwargs['rolling_kwargs'])

        #rolled = data.rolling(**kwargs['rolling_kwargs'])
        average = rolled.mean()

        if self.rolling_av is None:
            self.rolling_av = average
        else:
            self.rolling_av = self.rolling_av.append(average, verify_integrity=True)

    def create_experiment_subplots(self):
        # So that I can create a standardized grid for the 8 experiments
        fig, axs = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(16,10))
        return fig, axs

    def generate_rb_colors(self, n_colors):
        # Generate color red to blue color sequence
        n_colors = 8
        half_n = n_colors//2
        colors = np.ones((n_colors, 3))
        s = np.linspace(0,1,num=half_n)
        colors[-half_n : , 0] = 1 - s # r
        colors[     :    , 1] = 0     # g
        colors[ : half_n , 2] = s     # b
        return colors

    def generate_rb_color_fu(self, max_n):
        half_n = max_n // 2
        def rgb_fu(n):
            # red high, blue ramps up then blue high, red ramps down
            r = 1 if n < half_n else (max_n - n) / half_n
            g = 0
            b = 1 if n > half_n else n / half_n
            return (r, g, b)

        return rgb_fu


    # Functions to plot only gsd data
    def make_mean_gsd_time_plots(self, y_name='D50'):
        self.logger.write([f"Making station-averaged gsd time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_gsd_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_mean_gsd_time,
                kwargs={'y_name':y_name},
                before_msg=f"Plotting station mean gsd vs time",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_mean_gsd_time(self, y_name='D50'):
        x_name = 'exp_time'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        gsd_gather_kwargs = {
                'columns'   : y_name if isinstance(y_name, list) else [y_name],
                'new_index' : ['exp_time', 'sta_str'],
                }
        kwargs = {'plot_kwargs'    : plot_kwargs,
                  }

        # Make a color generating function
        get_color = lambda n: (0, 0, 0)

        gsd_data = self.gather_gsd_data(gsd_gather_kwargs)

        # Get subplots
        fig, axs = self.create_experiment_subplots()

        # Make one plot per experiment
        exp_codes = list(gsd_data.keys())
        exp_codes.sort()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            ax.set_title(f"Experiment {exp_code} {experiment.name}")

            # Get the data and group it by the line category
            exp_data = gsd_data[exp_code]
            avg_data = exp_data.groupby(level=x_name).mean()

            # Plot each group as a line
            #plot_kwargs['color'] = get_color(0)
            self.plot_group(avg_data, plot_kwargs)

        self.format_mean_gsd_figure(fig, axs, plot_kwargs)

        # Generate a figure name and save the figure
        filename_x = x_name.replace('_', '-').lower()
        filename_y = '*'.join(y_name).replace('_', '-').replace('*', '_').lower()
        figure_name = f"gsd_{filename_y}_v_{filename_x}.png"
        self.save_figure(figure_name)
        plt.show()

    def format_mean_gsd_figure(self, fig, axs, plot_kwargs):
        x = plot_kwargs['x']
        y = plot_kwargs['y']

        # Set the spacing and area of the subplots
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, left=0.05, bottom=0.075, right=0.90)

        fontsize = 16
        latex_str = {'D10':r'$D_{10}$', 'D50':r'$D_{50}$', 'D90':r'$D_{90}$', 
                '_':' ',
                }

        if len(y) > 1:
            # Format the common legend if there is more than one y
            plot_labels = y
            ax0_lines = axs[0,0].get_lines()
            fig.legend(handles=ax0_lines, labels=plot_labels,
                    loc='center right')

        # Set common x label
        t = r"Experiment time (hours)"
        s = r"Station (mm)"
        xlabel = t if x == 'exp_time' else s if x == 'sta_str' else x
        fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                fontsize=fontsize)

        # Set common y label
        
        ylabel = r"Grain size (mm)"
        fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                fontsize=fontsize, rotation='vertical')
        
        # Make a title
        title_y = y.copy()
        if len(y) > 1:
            title_y[-1] = f"and {title_y[-1]}"
        title_y = ', '.join(title_y)
        for code, ltx in latex_str.items():
            title_y = title_y.replace(code, ltx)
        title_str = rf"Change in the flume-mean {title_y} over time"
        title_str += rf" ({asctime()})"
        print(title_str)
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)


    def make_gsd_plots(self, x_name='exp_time', y_name='D50'):
        name = 'time' if x_name =='exp_time' else 'station' if x_name == 'sta_str' else x_name
        self.logger.write([f"Making gsd {name} plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_gsd_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_gsd,
                kwargs={'x_name':x_name, 'y_name':y_name},
                before_msg=f"Plotting gsd vs {name}",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_gsd(self, y_name='D50', x_name='exp_time'):
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        gsd_gather_kwargs = {
                'columns'   : y_name if isinstance(y_name, list) else [y_name],
                'new_index' : ['exp_time', 'sta_str'],
                }
        kwargs = {'plot_kwargs'    : plot_kwargs,
                  }

        # Get the name of the lines (exp_time or sta_str) automatically
        # ie pick the other item in a two item list
        line_options = gsd_gather_kwargs['new_index'].copy()
        line_options.remove(x_name)
        lines_category = line_options[0]

        # Make a color generating function
        get_color = self.generate_rb_color_fu(8)

        gsd_data = self.gather_gsd_data(gsd_gather_kwargs)

        # Get subplots
        fig, axs = self.create_experiment_subplots()

        # Make one plot per experiment
        exp_codes = list(gsd_data.keys())
        exp_codes.sort()
        self.plot_labels = []
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            ax.set_title(f"Experiment {exp_code} {experiment.name}")

            # Get the data and group it by the line category
            exp_data = gsd_data[exp_code]
            grouped = exp_data.groupby(level=lines_category)

            # Plot each group as a line
            for i, group in enumerate(grouped):
                plot_kwargs['color'] = get_color(i)
                self.plot_group(group, plot_kwargs)

        self.format_gsd_figure(fig, axs, plot_kwargs)

        # Generate a figure name and save the figure
        filename_y = y_name.replace('_', '-').lower()
        filename_x = x_name.replace('_', '-').lower()
        figure_name = f"gsd_{filename_y}_v_{filename_x}.png"
        self.save_figure(figure_name)
        plt.show()

    def save_figure(self, figure_name):
        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')
        
    def format_gsd_figure(self, fig, axs, plot_kwargs):
        x = plot_kwargs['x']
        y = plot_kwargs['y']

        # Set the spacing and area of the subplots
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, left=0.05, bottom=0.075, right=0.90)

        # Format the common legend
        if x == 'sta_str':
            # Then lines are times, make nicer time labels for the legend
            hour_fu = lambda n: 'hour' if n == 1 else 'hours'
            label_fu = np.vectorize(
                    lambda m: f"{m//60} {hour_fu(m//60)} {m%60} mins")
            self.plot_labels = label_fu(self.plot_labels)
        elif x == 'exp_time':
            # Then lines are stations, make nicer station labels for the legend

            get_meter = lambda sta_str: float(sta_str[-4:])/1000
            label_fu = np.vectorize(
                    lambda sta_str: f"Station {get_meter(sta_str):.1f}m")
            self.plot_labels = label_fu(self.plot_labels)

        ax0_lines = axs[0,0].get_lines()
        fig.legend(handles=ax0_lines, labels=self.plot_labels, loc='center right')

        fontsize = 16

        # Set common x label
        t = r"Experiment time (hours)"
        s = r"Station (mm)"
        xlabel = t if x == 'exp_time' else s if x == 'sta_str' else x
        fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                fontsize=fontsize)

        # Set common y label
        d50 = r"$D_{50} (mm)$"
        ylabel = d50 if y == 'D50' else y
        fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                fontsize=fontsize, rotation='vertical')
        
        # Make a title
        title_y = r"$D_{50}$" if y == 'D50' else y
        title_str = rf"Change in {title_y}"
        if x == 'exp_time':
            title_str += r" over time for each station"
        elif x == 'sta_str':
            title_str += r" at a station for each time"
        title_str += rf" ({asctime()})"
        print(title_str)
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)

    def plot_group(self, group, plot_kwargs):
        # Plot each group as a line
        try:
            name, group_data = group
            if name not in self.plot_labels:
                self.plot_labels.append(name)
        except (ValueError, AttributeError):
            group_data = group
        time_series = group_data.reset_index()
        time_series.sort_values(plot_kwargs['x'], inplace=True)
        if 'exp_time' in time_series.columns:
            time_series['exp_time'] = time_series['exp_time'] / 60
        time_series.plot(**plot_kwargs)
        plot_kwargs['ax'].get_xaxis().get_label().set_visible(False)
        plot_kwargs['ax'].tick_params(bottom=True, top=True, left=True, right=True)

    def gather_gsd_data(self, kwargs):
        # Gather all the gsd data into a dict of dataframes separated by 
        # exp_code
        self.gsd_data_frames = {}
        self.omnimanager.apply_to_periods(self._gather_gsd_time_data, kwargs)

        # Combines the separate frames into one dataframe per experiment
        gsd_data = {}
        for exp_code, frames in self.gsd_data_frames.items():
            gsd_data[exp_code] = pd.concat(frames)
        return gsd_data

    def _gather_gsd_time_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall gsd 
        # dataframe dict
        cols = kwargs['columns']
        new_index = kwargs['new_index']
        exp_code = period.exp_code

        data = period.gsd_data
        if data is not None:
            data.reset_index(inplace=True)
            data.set_index(new_index, inplace=True)
            if exp_code not in self.gsd_data_frames:
                self.gsd_data_frames[exp_code] = []
            self.gsd_data_frames[exp_code].append(data.loc[:, cols])


    # Functions to plot only Qs data
    def make_experiment_plots(self):
        self.logger.write(["Making experiment plots..."])

        #self.experiments = {}
        self.omnimanager.wipe_data()
        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_full_exp,
                before_msg="Plotting full experiment data",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_full_exp(self):
        # Meant for plotting one column against time
        x_column = 'exp_time_hrs'
        #y_column = 'Bedload all'
        y_column = 'D16'
        roll_window = 5 #minutes

        plot_kwargs = {
                'x'    : 'exp_time_hrs',
                'y'    : y_column,
                'kind' : 'scatter',
                'logy' : True,
                'xlim' : (0.5, 8.5),
                #'ylim' : (0.001, 5000), # for use with logy
                #'ylim' : (0.001, 500),
                'ylim' : (0, 40),
                }
        rolling_kwargs = {
                'window'      : roll_window*60, # seconds
                'min_periods' : 20,
                'center'      : True,
                #'on'          : plot_kwargs['x'],
                }
        kwargs = {'plot_kwargs'    : plot_kwargs,
                  'rolling_kwargs' : rolling_kwargs,
                  }

        columns_to_plot = [y_column]

        filename_y_col = y_column.replace(' ', '-').lower()
        logy_str = '_logy' if 'logy' in plot_kwargs and plot_kwargs['logy'] else ''
        figure_name = f"{filename_y_col}_roll-{roll_window}min{logy_str}.png"

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))
        twin_axes = []

        # Make one plot per experiment
        exp_codes = list(self.omnimanager.experiments.keys())
        exp_codes.sort()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            twax = ax.twinx()
            twin_axes.append(twax)
            experiment = self.omnimanager.experiments[exp_code]
            accumulated_data = experiment.accumulated_data

            self.rolling_av = None
            self.hydrograph = None

            # Plot the data points
            accumulated_data.plot(**plot_kwargs)

            # Generate and plot the hydrograph
            experiment.apply_period_function(self._generate_hydrograph, kwargs)
            self.hydrograph.sort_index(inplace=True)
            self.hydrograph.plot(ax=twax, style='g', ylim=(50,800))
            twax.tick_params('y', labelright='off')
            #if exp_code in ['2B', '5A']:
            #    twax.set_ylabel('Discharge (L/s)')

            # Generate and plot rolled averages
            self.roll_data(accumulated_data, kwargs)
            series_plot_kwargs = {k : plot_kwargs[k] for k in plot_kwargs
                                        if k not in ['x', 'y', 'kind']}
            #series_plot_kwargs['xlim'] = ax.get_xlim()
            series_plot_kwargs['style'] = 'r'
            self.rolling_av.plot(**series_plot_kwargs)

            ax.set_title(f"Experiment {experiment.code} {experiment.name}")

        # Can't figure out how to share the twinned y axes
        #for ax_row in 0,1:
        #    ax = axs[ax_row, 3]
        #twin_axes[0].get_shared_y_axis().join(*twin_axes)
        #twin_axes[0].set_ylabel('Discharge (L/s)')
        #yrange = plot_kwargs['ylim']
        plt.suptitle(f"{y_column} output ({roll_window} min" +
                f"roll window, {asctime()})")

        # Save the figure
        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')
        plt.show()

    def make_hysteresis_plots(self):
        self.logger.write(["Making hysteresis plots..."])

        #self.experiments = {}
        self.omnimanager.wipe_data()
        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_hysteresis,
                before_msg="Plotting hysteresis trends",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_hysteresis(self):
        #sns.set_style('ticks')
        t_column = 'exp_time_hrs'
        x_column = 'discharge'
        y_column = 'Bedload all'
        columns_to_plot = [t_column, x_column, y_column]

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))

        roll_window = 5 #minutes
        figure_name = f"hysteresis_{roll_window}min-mean_logy.png"
        plot_kwargs = {
                'x'      : x_column,
                'y'      : y_column,
                'kind'   : 'scatter',
                'logy'   : True,
                'xlim'   : (50, 125),
                #'ylim'   : (0.001, 5000),
                #'ylim'   : (0, 200),
                }
        rolling_kwargs = {
                'window'      : roll_window*60, # seconds
                'min_periods' : 40,
                #'on'          : plot_kwargs['x'],
                }
        kwargs = {'plot_kwargs'       : plot_kwargs,
                  'rolling_kwargs'    : rolling_kwargs,
                  }

        # Make one plot per experiment
        exp_codes = list(self.omnimanager.experiments.keys())
        exp_codes.sort()
        max_val = 0
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            accumulated_data = experiment.accumulated_data

            self.rolling_av = None

            # Generate rolled averages
            self.roll_data(accumulated_data, kwargs)
            max_val = max(max_val, self.rolling_av.max())

            # Select the hysteresis data
            data = pd.DataFrame(accumulated_data.loc[:,[x_column, t_column]])
            data.set_index(t_column, inplace=True)
            data[y_column] = self.rolling_av

            # Subsample data
            subsample_step = rolling_kwargs['window']
            data = data.iloc[::subsample_step, :]
            
            # Generate colors and markers
            n_rows = data.index.size
            half_n = n_rows//2
            colors = np.ones((n_rows, 3))
            s = np.linspace(0,1,num=half_n)
            colors[-half_n : , 0] = 1 - s # r
            colors[     :    , 1] = 0     # g
            colors[ : half_n , 2] = s     # b
            plot_kwargs['marker'] = 'o'
            plot_kwargs['c'] = 'none' # set default color to transparent
            plot_kwargs['facecolors'] = 'none' # set fill to transparent
            plot_kwargs['edgecolors'] = colors

            # Plot it!
            data.plot(**plot_kwargs)

            ax.set_title(f"Experiment {experiment.code} {experiment.name}")

        max_y = int(max_val*1.1)
        for ax in axs.flatten():
            #ax.set_ylim((0, max_y))
            # There are some outliers that should be removed. Ignore them for 
            # now.
            ax.set_ylim((0, 200))

        plt.suptitle(f"Hysteresis trends between total bedload output "+
                f"and discharge ({roll_window} min roll window, {asctime()})")

        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')
        plt.show()

    def make_cumulative_plots(self):
        self.logger.write(["Making cumulative plots..."])

        #self.experiments = {}
        self.omnimanager.wipe_data()
        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_cumulative,
                before_msg="Plotting cumulative transport",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_cumulative(self):
        #sns.set_style('ticks')
        x_column = 'exp_time_hrs'
        y_column = 'Bedload all'
        cumsum_column = 'cumulative'

        figure_name = f"cumulative.png"
        plot_kwargs = {
                'y'      : cumsum_column,
                'kind'   : 'line',
                'xlim'   : (0.5, 8.5),
                #'ylim'   : (0.001, 5000),
                #'ylim'   : (0, 200),
                }
        kwargs = {'plot_kwargs'       : plot_kwargs,
                  }

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))
        twin_axes = []

        # Make one plot per experiment
        exp_codes = list(self.omnimanager.experiments.keys())
        exp_codes.sort()
        max_val = 0
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            accumulated_data = experiment.accumulated_data

            # Generate and plot the hydrograph
            plot_kwargs['x'] = 'exp_time_hrs'
            self.hydrograph = None
            twax = ax.twinx()
            twin_axes.append(twax)
            experiment.apply_period_function(self._generate_hydrograph, kwargs)
            self.hydrograph.sort_index(inplace=True)
            self.hydrograph.plot(ax=twax, style='g', ylim=(50,800))
            del plot_kwargs['x']
            twax.tick_params('y', labelright='off')

            # Select the target data
            data = pd.DataFrame(accumulated_data.loc[:,[x_column, y_column]])
            data.set_index(x_column, inplace=True)

            # Generate cumulative
            cumsum = data.cumsum(axis=0)
            data[cumsum_column] = cumsum

            # Subsample data
            subsample_step = 10
            data = data.iloc[::subsample_step, :]
            
            # Plot it!
            data.plot(**plot_kwargs)

            ax.set_title(f"Experiment {experiment.code} {experiment.name}")

        plt.suptitle(f"Cumulative sum of total bedload output ({asctime()})")

        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')
        plt.show()

    def make_transport_histogram_plots(self):
        # To help identify outliers. Plot a histogram of the raw transport rate 
        # (grams/sec). 
        self.logger.write(["Making total transport distribution plots..."])

        #self.experiments = {}
        self.omnimanager.wipe_data()
        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_transport_histograms,
                before_msg="Plotting total transport distribution",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_transport_histograms(self):
        x_column = 'Bedload all'
        t_column = 'exp_time_hrs'

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))

        figure_name = f"bedload_transport_histograms.png"
        plot_kwargs = {
                #'kind' : 'scatter',
                #'by'   : x_column,
                #'y'    : x_column,
                'logy' : True,
                #'logx' : True,
                }
        kwargs = {'plot_kwargs'       : plot_kwargs,
                  }

        # Make one plot per experiment
        exp_codes = list(self.omnimanager.experiments.keys())
        exp_codes.sort()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            accumulated_data = experiment.accumulated_data

            # Select the target data
            data = pd.DataFrame(accumulated_data.loc[:,[x_column, t_column]])
            data.set_index(t_column, inplace=True)
            series = pd.Series(data.loc[:, x_column])
            lg_max = np.log10(series.max())
            lg_min = np.log10(series.min())

            # Make log-spaced bins
            bins = np.power([10], np.linspace(lg_min*0.9, lg_max*1.1, num=100))
            binned = pd.cut(series, bins, labels=False)
            counts = binned.value_counts()

            y = counts.values
            x = counts.index

            # Plot it!
            counts.plot.hist(**plot_kwargs)

            ax.set_title(f"Experiment {experiment.code} {experiment.name}")

        plt.suptitle(f"Histogram of bedload transport rates ({asctime()})")

        #filepath = ospath_join(self.figure_destination, figure_name)
        #self.logger.write(f"Saving figure to {filepath}")
        #plt.savefig(filepath, orientation='landscape')
        plt.show()


    # Functions to be given to PeriodData instances
    def _generate_hydrograph(self, period_data, kwargs={}):
        # Generate a hydrograph series for plotting later
        #
        # This function is called by the PeriodData (via apply_to_period)
        if not self._check_ignore_period(period_data):
            plot_kwargs = kwargs['plot_kwargs']
            x = plot_kwargs['x']
            y = 'discharge'

            data = period_data.Qs_data
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
        # Plot a dataframe using plot kwargs provided in kwargs
        # As in, kwargs['plot_kwargs'] needs to exists. It will be handed off 
        # directly to plt/mpl/pandas plotting function
        #
        # This function is called by the PeriodData (via apply_to_period)
        if not self._check_ignore_period(period_data):
            all_data = period_data.Qs_data
            all_data.plot(**kwargs['plot_kwargs'])

    def _check_ignore_period(self, period_data):
        # Return True if period should be ignored
        return period_data.step in self.ignore_steps



if __name__ == "__main__":
    # Run the script
    grapher = UniversalGrapher()
    grapher.make_mean_gsd_time_plots(y_name=['D50'])#, 'D90'])
    #grapher.make_gsd_plots(x_name='exp_time', y_name='D50')
    #grapher.make_gsd_plots(x_name='sta_str',  y_name='D50')
    #grapher.make_experiment_plots()
    #grapher.make_hysteresis_plots()
    #grapher.make_cumulative_plots()
    #grapher.make_transport_histogram_plots()
