#!/usr/bin/env python3

from os.path import join as ospath_join
from time import asctime
from time import sleep
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import statsmodels.api as sm

# From Helpyr
from helpyr import helpyr_misc as hm
from helpyr import logger
from helpyr import data_loading

from omnipickle_manager import OmnipickleManager
import global_settings as settings
import tokens

# This file is intentionally very repetitive. Each type of plot gets its own 
# set of functions due to the potential for required very specialized code to 
# format the image just right. Though there are opportunities to generalize 
# some...

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

    # Generic functions
    def __init__(self, fig_debug=False):

        self.fig_debug = fig_debug

        # File locations
        self.log_filepath = f"{settings.log_dir}/universal_grapher.txt"

        # Start up logger
        self.logger = logger.Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("Universal Grapher")

        # omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()

        self.figure_destination = settings.figure_destination
        self.figure_subdir = ''
        self.figure_subdir_dict = {
                'dem_stats'      : 'dem_stats',
                'dem_subplots'   : 'dem_subplots',
                'dem_variograms' : 'dem_variograms',
                'depth'          : 'depth-based',
                'lighttable'     : 'lighttable',
                'trap'           : 'trap',
                'feed_sieve'     : 'feed_sieve',
                'gsd'            : 'gsd',
                'synthesis'      : 'synthesis',
                'mass_balance'   : 'mass_balance',
                }
        hm.ensure_dir_exists(self.figure_destination)

        self.export_destination = settings.export_destination
        self.export_subdir = ''
        hm.ensure_dir_exists(self.export_destination)
        self.data_loader = data_loading.DataLoader(
                self.export_destination, logger=self.logger)

        self.ignore_steps = []

        self.for_paper = True
        self.skip_2B = False

    # General use functions
    def generic_make(self, name, load_fu, plot_fu, load_fu_kwargs=None, plot_fu_kwargs=None, fig_subdir=''):
        self.logger.write([f"Making {name} plots..."])
        
        if fig_subdir in self.figure_subdir_dict:
            self.figure_subdir = self.figure_subdir_dict[fig_subdir]

        indent_function = self.logger.run_indented_function

        indent_function(load_fu, kwargs=load_fu_kwargs,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(plot_fu, kwargs=plot_fu_kwargs,
                before_msg=f"Plotting {name}",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def create_experiment_subplots(self, rows=4, cols=2):
        # So that I can create a standardized grid for the 8 experiments
        size = (12, 7.5) if self.for_paper else (16, 10)
        fig, axs = plt.subplots(rows, cols, sharey=True, sharex=True,
                figsize=size)
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
            r = 1 if n <= half_n else (max_n - n) / half_n
            g = 0
            b = 1 if n > half_n else n / half_n
            return (r, g, b)

        return rgb_fu

    def roll_data(self, data, roll_kwargs={}):
        # Roll the data. Returns the generic rolled object for flexibility of 
        # what functions to call on the roll.
        # data is a dataframe with columns for x and y values
        roll_kwargs = roll_kwargs.copy()

        if isinstance(data, pd.Series):
            [roll_kwargs.pop(key) for key in ['x', 'y'] if key in roll_kwargs]
            series = data
        else:
            # Get the function parameters
            x_var = roll_kwargs.pop('x')
            y_var = roll_kwargs.pop('y')

            # Get the data
            x = data.loc[:, x_var]
            y = data.loc[:, y_var]

            # Convert to a series
            series = pd.Series(data=y.values, index=x)

        # Roll it
        rolled_data = series.rolling(**roll_kwargs)

        return rolled_data

    def _calc_retrended_slope(self, data, flume_elevations=None, intercept=None):
        # Assumes data values have same units as data column names (which are 
        # positions)
        # Return ols results and flume elevation (for reuse)

        if flume_elevations is None:
            # Calculate new flume elevations
            positions = data.columns.values
            flume_elevations = positions * settings.flume_slope

        # Calculate the slope
        trended_profile = data + flume_elevations
        ols_out = trended_profile.apply(self._ols, axis=1, intercept=intercept)
        return ols_out, flume_elevations

    def _ols(self, series, intercept=None):
        # Do an OLS linear regression to calculate average slope and intercept.
        # series is a Panda Series where:
        #   series index is the independent var
        #   series values are the dependent var
        # if intercept is None, then ols will calculate an intercept
        # if intercept is not None, then ols will calculate line through 
        # provided intercept.
        #
        # output is a pd.Series:
        #              slope intercept r-sqr
        #  series name  #     #         #
        #
        # Intended for use with DataFrame.apply() function
        profile_name = series.name
        positions = series.index.values

        if pd.isnull(series).all():
            return pd.Series({'r-sqr'     : None,
                              'slope'     : None,
                              'intercept' : None,
                             }, name=profile_name)

        if intercept is None:
            # No fixed intercept provided, allow freedom
            independent = sm.add_constant(positions.astype(np.float64))
        else:
            # sm.OLS will force fit through zero by default
            # offset the data by the intercept to force fit through intercept
            series = series - intercept
            independent = positions.astype(np.float64)
        dependent = series.values.astype(np.float64)

        results = sm.OLS(dependent, independent, missing='drop').fit()
        p = results.params
        out = {'r-sqr' : results.rsquared}
        if intercept is None:
            out['slope'] = p[1]
            out['intercept'] = p[0]
        else:
            out['slope'] = p[0]
            out['intercept'] = intercept

        return pd.Series(out, name=profile_name)

    def save_figure(self, figure_name):
        filepath = ospath_join(self.figure_destination, self.figure_subdir, figure_name)
        if self.fig_debug:
            self.logger.write(f"DEBUG: NOT saving figure to {filepath}")
        else:
            self.logger.write(f"Saving figure to {filepath}")
            plt.savefig(filepath, orientation='landscape')

    def export_data(self, data, filename):
        dirpath = ospath_join( self.export_destination, self.export_subdir)
        hm.ensure_dir_exists(dirpath)

        filepath = ospath_join( dirpath, filename)

        self.logger.write(f"Exporting data to {filepath}")
        self.data_loader.save_txt(data, filepath,
                kwargs={'sep':',', 'index':True}, is_path=True)

    
    def plot_group(self, group, plot_kwargs):
        # Plot each group as a line
        # Really, this would be more appropriate named as plot_dataframe as it 
        # is most often used to plot a dataframe. However, it was originally 
        # written for groupby objects, which I am having trouble using 
        # properly.
        groups = []
        if isinstance(group, pd.core.frame.DataFrame):
            groups = [group]
        elif isinstance(group, pd.core.series.Series):
            groups = [group.to_frame()]
        elif isinstance(group, pd.core.groupby.DataFrameGroupBy):
            # Set of groups in groupby object
            names, groups = zip(*[iter_val for iter_val in group])
            try:
                for name in names:
                    if name not in self.plot_labels:
                        self.plot_labels.append(name)
            except AttributeError:
                pass
        elif isinstance(group, tuple) and len(group) == 2:
            assert isinstance(group[1], pd.core.groupby.DataFrame)
            # Came from a grouby object
            self.plot_labels.append(group[0])
            groups = [group[1]]
        else:
            print("Unknown argument")
            print(type(group))
            print(group)
            assert(False)
        for group_data in groups:
            self._time_plot_prep(group_data, plot_kwargs)

        plot_kwargs['ax'].set_ylabel('')

    def _time_plot_prep(self, data, plot_kwargs, auto_plot=True):
        # For plotting data where 'exp_time' is in the index as minutes
        df = data.reset_index()
        df.sort_values(plot_kwargs['x'], inplace=True)
        if 'exp_time' in df.columns:
            df['exp_time'] = df['exp_time'] / 60
        if auto_plot:
            # Plot using some default settings
            self._generic_df_plot(df, plot_kwargs)
        else:
            # Return the time formatted df for special plotting
            return df

    def _generic_df_plot(self, df, plot_kwargs):
        # Plot dataframe with some default formatting
        try:
            df.plot(**plot_kwargs)
        except TypeError:
            # No values to plot, skip it
            pass
        plot_kwargs['ax'].get_xaxis().get_label().set_visible(False)
        plot_kwargs['ax'].tick_params(
                bottom=True, top=True, left=True, right=True)

    def generic_plot_experiments(self, plot_fu, post_plot_fu, data, plot_kwargs, figure_name, subplot_shape=(4,2), save=True):
        # Generic framework for plotting for the 8 experiments
        # Data is recommended to be a dict of experiments, but doesn't have to 
        # be

        # Get subplots
        fig, axs = self.create_experiment_subplots(*subplot_shape)

        # Make one plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        legend_ax = None
        legend_exp_code = plot_kwargs.pop('legend_exp_code') \
                if 'legend_exp_code' in plot_kwargs else '2B'
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            if exp_code == legend_exp_code:
                legend_ax = ax

            if self.for_paper and exp_code == '2B' and self.skip_2B:
                    self.logger.write(f">>Skipping<< experiment {exp_code}")
                    ax.set_axis_off()
                    continue

            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            ax.set_title(f"{exp_code} {experiment.name}")

            plot_fu(exp_code, data, plot_kwargs)

            if not self.for_paper:
                self.plot_2B_X(exp_code, ax)

        if self.for_paper:
            self.add_common_label(axs[0,0], legend_ax, has_quartiles=False)

        post_plot_fu(fig, axs, plot_kwargs)

        # Generate a figure name and save the figure
        if save:
            self.save_figure(figure_name)
        plt.show()

    def format_generic_figure(self, fig, axs, plot_kwargs, fig_kwargs):
        # Must add xlabel, ylabel, and title to fig_kwargs
        xlabel = fig_kwargs['xlabel'] # Common xlabel
        ylabel = fig_kwargs['ylabel'] # Common ylabel
        title_str = fig_kwargs['title'] # Common title
        plot_labels = [] if 'legend_labels' not in fig_kwargs \
                else fig_kwargs['legend_labels'] # Common legend labels

        # Set the spacing and area of the subplots
        fig.tight_layout()
        if self.for_paper:
            # Minimize whitespace
            # leave out title and legend
            # Larger font so it is readable in pdf

            fig.subplots_adjust(top=0.95, left=0.075, bottom=0.075, right=0.99)
            fontsize = 25
            
            # Set common x label
            fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                    fontsize=fontsize)

            # Set common y label
            rotation = fig_kwargs['ylabel_rotation'] if 'ylabel_rotation' in fig_kwargs else 'vertical'
            fig.text(0.0005, 0.5, ylabel, va='center', usetex=True,
                    fontsize=fontsize, rotation=rotation)
        else:
            fig.subplots_adjust(top=0.9, left=0.10, bottom=0.075, right=0.90)
            fontsize = 16

            if len(plot_labels) > 1:
                # Format the common legend if there is more than one y
                ax0_lines = axs[0,0].get_lines()
                fig.legend(handles=ax0_lines, labels=plot_labels,
                        loc='center right')

            # Set common x label
            fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                    fontsize=fontsize)

            # Set common y label
            rotation = fig_kwargs['ylabel_rotation'] if 'ylabel_rotation' in fig_kwargs else 'vertical'
            fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                    fontsize=fontsize, rotation=rotation)
            
            # Make a title
            plt.suptitle(title_str, fontsize=fontsize, usetex=True)

    def generic_set_grid(self, ax, **kwargs):
        # Use a consistent format for the grids
        # This makes the major grid a light grey and only shows the minor ticks 
        # on the x axis
        if 'xticks_minor' in kwargs and kwargs['xticks_minor']:
            minor_locator = mpl.ticker.AutoMinorLocator(2)
            ax.xaxis.set_minor_locator(minor_locator)
            ax.tick_params(axis='x', which='minor', top=True, bottom=True)
        if 'yticks_minor' in kwargs and kwargs['yticks_minor']:
            minor_locator = mpl.ticker.AutoMinorLocator(2)
            ax.yaxis.set_minor_locator(minor_locator)
            ax.tick_params(axis='y', which='minor', left=True, right=True)
        ax.grid(True, which='major', color='#d6d6d6')
        #plot_kwargs['ax'].grid(True, which='minor', color='#f2f2f2', axis='x')
        ax.set_axisbelow(True)

    def draw_feed_Di(self, ax, Di, zorder=1, multiple=1.0, text=None):
        # Di like 'D50'
        # multiple is meant for multiples of the armor ratio
        # text is the in-plot line label 
        assert(Di in settings.sum_feed_Di)
        kwargs = {'c' : 'k', 'linestyle' : '--',
                'label' : self.get_feed_Di_label(Di), # -> Feed D_50
            }
        if zorder is not None:
            kwargs['zorder'] = zorder
        feed_Di = settings.sum_feed_Di[Di] * multiple
        ax.axhline(feed_Di, **kwargs)

        if text is not None:
            ax.annotate(text, xy=(1.25, feed_Di*1.01), xycoords='data')

    def get_feed_Di_label(self, Di):
        return rf"Feed $D_{{{Di[1:]}}}$"

    def plot_distribution(self, distribution, is_frac=True, ax=None, **kwargs):
        if not is_frac:
            distribution = distribution.cumsum() / distribution.sum()
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        distribution.plot(ax=ax, logx=True)
        index = distribution.index.values
        if 'hlines' in kwargs:
            [ax.axhline(y) for y in kwargs['hlines']]

        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(index))
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(index))
        ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(np.arange(11) / 10))
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(11) / 10))
        ax.minorticks_off()
        ax.set_ylim((0,1.01))
        ax.set_ylabel(f"Fraction less than")
        self.generic_set_grid(ax)

        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

    def plot_2B_X(self, exp_code, ax):
        # Puts a big X over the 2B plot so people aren't confused
        if exp_code != '2B':
            return
        for i in [0,1]:
            #line = mpl.lines.Line2D([0, 1], [i, abs(i-1)],
            #        lw=2, color='k', alpha=0.75)
            #line.set_clip_on(False)
            #ax.add_line(line)
            ax.plot([0, 1], [i, 1-i], transform=ax.transAxes,
                    lw=2, color='k')


    # Plotting functions that cross genres
    def make_pseudo_hysteresis_plots(self, y_name='D50', plot_2m=True):
        #reload_kwargs = {
        #    'check_ignored_fu' : \
        #        lambda period_data: period_data.step == 'rising-50L',
        #}
        if y_name in ['Bedload all', 'D50', 'D84']:
            reload_fu = self.omnimanager.reload_Qs_data
            fig_subdir = 'lighttable'
        elif y_name in ['depth', 'slope']:
            reload_fu = self.omnimanager.reload_depth_data
            fig_subdir = 'depth'
        elif y_name in ['bed-D50', 'bed-D84']:
            reload_fu = self.omnimanager.reload_gsd_data
            fig_subdir = 'gsd'
        else:
            raise NotImplementedError

        self.generic_make(f"pseudo hysteresis",
                reload_fu, self.plot_pseudo_hysteresis,
                #load_fu_kwargs=reload_kwargs,
                plot_fu_kwargs={'y_name':y_name, 'plot_2m':plot_2m},
                fig_subdir=fig_subdir)

    def plot_pseudo_hysteresis(self, y_name='D50', plot_2m=True):
        # Do stuff before plot loop
        x_name = 'pseudo discharge'
        roll_window = 10 #minutes
        #
        # note: plot_2m Only works for data with stationing (eg. depth data)

        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'line',
                #'legend' : True,
                'legend' : False,
                #'xlim' : (-0.25, 8.25),
                #'ylim' : (0, settings.lighttable_bedload_cutoff), # for use without logy
                #'logy' : True,
                #'ylim' : (0.001, settings.lighttable_bedload_cutoff), # for use with logy
                }

        rolling_kwargs = {
                'x'           : 'exp_time_hrs',
                'y'           : y_name,
                'window'      : roll_window*60, # seconds
                'min_periods' : 20,
                'center'      : True,
                #'on'          : plot_kwargs['x'],
                }
        # Add to plot_kwargs as a hacky way to get info to _plot function
        plot_kwargs['rolling_kwargs'] = rolling_kwargs
        plot_kwargs['plot_2m'] = plot_2m

        if y_name in ['Bedload all', 'D50', 'D84']:
            gather_kwargs = {
                    #'ignore_steps' : ['rising-50L']
                    }
            all_data = self.gather_Qs_data(gather_kwargs)
            if y_name in ['D50', 'D84']:
                # Gather the sieve data Di for plotting
                all_sieve_data = self.gather_sieve_data({})
                plot_kwargs['sieve_data'] = all_sieve_data
        elif y_name in ['depth', 'slope']:
            gather_kwargs = {
                    'new_index' : ['exp_time', 'location'],
                    }
            all_data = self.gather_depth_data(gather_kwargs)
        elif y_name in ['bed-D50', 'bed-D84']:
            gather_kwargs = {
                'columns'   : [y_name[-3:]],
                'new_index' : ['exp_time', 'sta_str'],
                }
            all_data = self.gather_gsd_data(gather_kwargs)
        else:
            raise NotImplementedError

        filename_y_col = y_name.replace(' ', '-').lower()
        logy_str = '_logy' if 'logy' in plot_kwargs and plot_kwargs['logy'] else ''
        plot_2m_str = '_2m' if plot_2m else ''
        figure_name = '_'.join([
            f"pseudo_hysteresis{plot_2m_str}",
            f"{filename_y_col}",
            f"roll-{roll_window}min{logy_str}.png",
            ])

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_pseudo_hysteresis, self._format_pseudo_hysteresis, 
            all_data, plot_kwargs, figure_name, subplot_shape=(2,4))

    def _plot_pseudo_hysteresis(self, exp_code, all_data, plot_kwargs):
        # Pull out some kwargs variables
        y_name = plot_kwargs['y']
        x_name = plot_kwargs['x']
        ax = plot_kwargs['ax']
        plot_kwargs = plot_kwargs.copy()
        rolling_kwargs = plot_kwargs.pop('rolling_kwargs')
        plot_2m = plot_kwargs.pop('plot_2m')

        exp_data = all_data[exp_code]
        if y_name in ['Bedload all', 'D50', 'D84']:
            # Based on Qs data
            data = exp_data[exp_data['discharge'] != 50]
            try:
                sieve_data = plot_kwargs.pop('sieve_data')[exp_code]
            except KeyError:
                sieve_data = None

            # Roll it
            roll_y_var = rolling_kwargs['y']
            rolled =  self.roll_data(data, rolling_kwargs)
            if y_name == 'Bedload all':
                data = data.assign(roll_median=rolled.median().values)
                plot_kwargs['y'] = 'roll_median'
            else:
                data = data.assign(roll_mean=rolled.mean().values)
                plot_kwargs['y'] = 'roll_mean'

        elif y_name in ['depth', 'slope']:
            # Based on depth data
            sta_min, sta_max = (4.5, 6.5) if plot_2m else (0, 8)
            sta_keep = [s for s in exp_data.columns if sta_min <= s <= sta_max]
            if y_name == 'depth':
                data = exp_data.xs(y_name, level='location').loc[:, sta_keep]
                avg_depths = data.mean(axis=1)
                data.loc[:, y_name] = avg_depths
            elif y_name == 'slope':
                # Assume water surface slope
                cm2m = 1/100
                surface_data = exp_data.xs('surface', level='location') *cm2m
                keep_data = surface_data.loc[:, sta_keep]
                notnull = keep_data.notnull()
                keep_data[notnull.sum(axis=1) <=3] = np.nan # require > 3 pts
                ols_out, flume_elevations = self._calc_retrended_slope(
                        keep_data, None, None)
                data = ols_out
        elif y_name in ['bed-D50', 'bed-D84']:
            # Trim stations
            sta_min, sta_max = (4.5, 6.5) if plot_2m else (0, 8)
            sta_keep = [(t,s) for t,s in exp_data.index \
                    if sta_min <= float(s.split('-')[1])/1000 <= sta_max]
            raw_bed_data = exp_data.loc[sta_keep, :]
            
            # Get geom mean
            log2_bed_data = np.log2(raw_bed_data)
            log2_mean = log2_bed_data.groupby('exp_time').mean()
            data = np.exp2(log2_mean)

            # Make a geometric mean point at 270min (4.5hrs) so lines look 
            # cleaner
            index_vals = data.index.values
            insert_index = index_vals.searchsorted(270)
            before_index = index_vals[insert_index - 1]
            after_index = index_vals[insert_index]
            
            yn = y_name[-3:]
            log2_before = np.log2(data.loc[before_index, yn])
            log2_after = np.log2(data.loc[after_index, yn])
            data.loc[270, yn] = np.exp2((log2_before + log2_after)/2)
            data.sort_index(inplace=True)

            plot_kwargs['y'] = y_name[-3:]
        else:
            raise NotImplementedError

        # Fold time in half around the peak using 'exp_time_hrs'
        peak_time = 4.5 # hrs
        exp_time_name = 'exp_time'
        exp_time_hrs_name = 'exp_time_hrs'
        if exp_time_hrs_name in data.columns:
            pass
        elif exp_time_name == data.index.name:
            # Make new column based on index
            data.loc[:, exp_time_hrs_name] = data.index / 60
        else:
            raise NotImplementedError

        exp_time_hrs = data[exp_time_hrs_name]
        #assert(exp_time_hrs.iloc[-1] <= 8.0) # Otherwise not peak_time invalid
        data['pseudo discharge'] = peak_time - np.fabs(exp_time_hrs - peak_time)

        # Split data into limbs
        rising = data[data[exp_time_hrs_name] <= peak_time]
        falling = data[data[exp_time_hrs_name] >= peak_time]

        # Print some mass stats if applicable
        if y_name == 'Bedload all':
            rising_sum = rising[roll_y_var].sum() / 1000 # kg
            falling_sum = falling[roll_y_var].sum() / 1000 # kg
            limb_ratio = rising_sum / falling_sum

            self.logger.write([f"Rising sum = {rising_sum:3.0f} kg",
                f"Falling sum = {falling_sum:3.0f} kg",
                f"Rising/falling = {limb_ratio:3.2f}"],
                local_indent=1)

            # print latex table row
            print(f" {exp_code} & {rising_sum:3.0f} & {falling_sum:3.0f} & " + \
                  f"{(rising_sum+falling_sum):3.0f} & {limb_ratio:3.2f} \\\\")

        # Grab the default pyplot colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        rising_color, falling_color = colors[0:2]

        # Not actually grouped, but can still use self.plot_group
        plot_kwargs['label'] = 'Rising Limb'
        plot_kwargs['color'] = rising_color
        self.plot_group(rising, plot_kwargs)

        plot_kwargs['label'] = 'Falling Limb'
        plot_kwargs['color'] = falling_color
        self.plot_group(falling, plot_kwargs)
        
        # Draw the feed Di if applicable
        if y_name in ['D50', 'D84']:
            self.draw_feed_Di(ax, y_name)
            
            # Plot the sieve data Di if applicable
            sieve_data[y_name] = self.calc_Di(sieve_data, target_Di=y_name)

            sieve_data.reset_index(inplace=True)
            sieve_exp_time_hrs = sieve_data['exp_time'] / 60
            assert(sieve_exp_time_hrs.iloc[-1] == 8.0)
            sieve_folded_hrs = peak_time - np.fabs(sieve_exp_time_hrs - peak_time)
            sieve_data['pseudo discharge'] = sieve_folded_hrs

            sieve_plot_kwargs = {
                    'x'      : 'exp_time' if x_name == 'exp_time_hrs' else x_name,
                    'y'      : y_name,
                    'label'  : rf"Trap {y_name}",
                    'ax'     : ax,
                    'legend' : False,
                    'marker' : '*',
                    'markersize': 7,
                    'linestyle' : 'None',
                    }
            sieve_rising = sieve_data[sieve_exp_time_hrs <= peak_time]
            sieve_falling = sieve_data[sieve_exp_time_hrs > peak_time]

            sieve_plot_kwargs['color'] = rising_color
            self.plot_group(sieve_rising, sieve_plot_kwargs)
            sieve_plot_kwargs['color'] = falling_color
            self.plot_group(sieve_falling, sieve_plot_kwargs)

        elif y_name == 'bed-D84':
            self.draw_feed_Di(ax, y_name[-3:])

        elif y_name == 'bed-D50':
            yn = y_name[-3:]
            self.draw_feed_Di(ax, yn, multiple=1.0, text='Armor 1.0')
            self.draw_feed_Di(ax, yn, multiple=1.5, text='Armor 1.5')

        # Turn on the grid
        self.generic_set_grid(ax, yticks_minor=True)

        # Set the tick marks
        ax.set_xlim((1, peak_time))
        ticker = mpl.ticker
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(5)))
        ax.tick_params(axis='x', which='major', top=True, bottom=True)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

        # Turn off minor axis ticks
        ax.tick_params(axis='x', which='minor', top=False, bottom=False)

        #text_locations = np.concatenate([np.arange(4) + 0.5, [4.25]])

        ##for xloc, label in zip(text_locations, text_labels):
        ##    ax.text(xloc, 0, label)
        #text_labels = mpl.ticker.FixedFormatter([rf'${d}L/s$' for d in [50, 62, 75, 87, 100]])
        #ax.xaxis.set_major_formatter(text_labels)
        #ax.xaxis.OFFSETTEXTPAD = 100
        ##ax.tick_params(pad=15)

        #ax.grid(True, which='major', color='#d6d6d6')
        #plot_kwargs['ax'].grid(True, which='minor', color='#f2f2f2', axis='x')
        #ax.set_axisbelow(True)

    def _format_pseudo_hysteresis(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Discharge",
                'legend_labels' : [r"Rising", "Falling"],
                }

        y_name = plot_kwargs['y']
        if y_name == 'Bedload all': 
            fig_kwargs['ylabel'] = r"Bedload transport (g/s)"
            fig_kwargs['title'] = r"Pseudo hysteresis of bedload transport"
        elif y_name in ['D50', 'D84']: 
            Di = plot_kwargs['y']
            fig_kwargs['ylabel'] = rf"$D_{{ {Di[1:]} }}$ (mm)"
            fig_kwargs['title'] = rf"Pseudo hysteresis of {Di}"
            fig_kwargs['legend_labels'].append(self.get_feed_Di_label(Di))
        elif y_name == 'depth':
            fig_kwargs['ylabel'] = rf"Depth (cm)"
            fig_kwargs['title'] = rf"Pseudo hysteresis of depth"
        elif y_name == 'slope':
            fig_kwargs['ylabel'] = rf"Slope (m/m)"
            fig_kwargs['title'] = rf"Pseudo hysteresis of slope"
        elif y_name in ['bed-D50', 'bed-D84']: 
            Di = plot_kwargs['y'][-3:]
            fig_kwargs['ylabel'] = rf"$D_{{ {Di[1:]} }}$ (mm)"
            fig_kwargs['title'] = rf"Pseudo hysteresis of bed surface {Di}"
            fig_kwargs['legend_labels'].append(self.get_feed_Di_label(Di))

            if y_name == 'bed-D50':
                # Hacky hardcoded way to make sure armor ratio text fits on 
                # graph
                global_ymin, global_ymax = (7, 12.5) # min range
                for ax in axs.flatten():
                    ymin, ymax = ax.get_ylim()
                    global_ymin = min(global_ymin, ymin)
                    global_ymax = max(global_ymax, ymax)
                    ax.set_ylim((global_ymin, global_ymax))

        else:
            raise NotImplementedError
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)

        ticker = mpl.ticker
        for ax in axs[-1]:
            # Set the major tick length
            default_length = ax.xaxis.get_major_ticks()[0].get_tick_padding()
            ax.tick_params(axis='x', which='major', bottom=True, top=False,
                    length=default_length * 3)
            
            # Set the minor tick locations
            text_locations = np.concatenate([np.arange(4) + 0.5, [4.25]])
            text_locator = ticker.FixedLocator(text_locations)
            ax.xaxis.set_minor_locator(text_locator)

            # Set the minor tick labels
            text_formats = [rf'${d}L/s$' for d in [50, 62, 75, 87, 100]]
            text_formatter = ticker.FixedFormatter(text_formats)
            ax.xaxis.set_minor_formatter(text_formatter)


    # Functions to plot only dem data
    def make_dem_subplots(self, plot_2m=True):
        self.generic_make("dem subplots",
                self.omnimanager.reload_dem_data,
                self.plot_dem_subplots, plot_fu_kwargs={'plot_2m' : plot_2m},
                fig_subdir='dem_subplots')

    def plot_dem_subplots(self, plot_2m=True):
        # Can't use the generic_plot_experiments function because of the way I 
        # want to display and save them.
        plot_kwargs = {
                }
        dem_gather_kwargs = {
                'wall_trim'  : settings.dem_wall_trim,
                }

        if plot_2m:
            dem_gather_kwargs['sta_lim'] = settings.stationing_2m
        dem_data = self.gather_dem_data(dem_gather_kwargs)

        # Calculate period ranks
        ranking = {"rising" : 0, "falling" : 1,
                "50L" : 0, "62L" : 1, "75L" : 2, "87L" : 3, "100L" : 4}
        get_rank = lambda l, d: d + 2*l*(4-d)
        get_ax_index = lambda l, d: get_rank(*[ranking[k] for k in (l, d)])

        # Generate a base file name
        length_str = '2m' if plot_2m else '8m'
        filename_base = f"{length_str}_dems_{{}}.png"

        # Start plotting; 1 plot per experiment
        color_min, color_max = settings.dem_color_limits
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code in exp_codes:
            self.logger.write(f"Creating plots for {exp_code}")
            self.logger.increase_global_indent()
            
            # Create the subplot for this experiment
            if plot_2m:
                # make 3x3 grid
                fig, axs = (plt.subplots(3, 3, sharey=True, sharex=True,
                        figsize=(18,10)))
            else:
                # make 8x1 grid
                fig, axs = plt.subplots(8, 1, sharey=True, sharex=True,
                        figsize=(10,12))
            first_image=None
            axs = axs.flatten()

            # Plot the dems for this experiment
            for period_key, period_dem in dem_data[exp_code].items():
                limb, discharge, time, exp_time = period_key
                self.logger.write(f"Plotting {limb} {discharge} {time}")
                assert(time == 't60')
                ax = axs[get_ax_index(limb, discharge)]

                title_hr = exp_time//60
                title_hr_str = f"{title_hr} {'hour' if title_hr == 1 else 'hours'}"
                ax.set_title(f"{limb.capitalize()} {discharge} {title_hr_str}")

                px_y, px_x = period_dem.shape
                ax.set_ybound(0, px_y)
                img = ax.imshow(period_dem, vmin=color_min, vmax=color_max)
                if first_image is None:
                    first_image = img

                # Convert axes from px to stationing
                dem_res = settings.dem_resolution
                ticker = mpl.ticker

                # Convert x tick labels
                dem_offset = settings.stationing_2m[0] if plot_2m else settings.dem_long_offset
                dem_offset_mod = dem_offset%500#settings.dem_long_offset % 1000
                long_fu = lambda x, p: f"{(x * dem_res + dem_offset) / 1000:1.1f}"
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(long_fu))

                # Convert x tick locations
                # Make the resulting tick locations at either 0.5 or 1 m 
                # intervals (250px or 500px).
                # Use MultipleLocator to easily generate tick locations then 
                # correct for the dem offset
                auto_locator = ticker.MultipleLocator(250 if plot_2m else 500)
                auto_ticks = auto_locator.tick_values(0, px_x)[1:-1]
                offset_ticks = auto_ticks - dem_offset_mod / dem_res
                ax.xaxis.set_major_locator(ticker.FixedLocator(offset_ticks))

                # Convert y tick labels
                dem_trim = settings.dem_wall_trim
                trav_fu = lambda x, p: f"{(x * dem_res + dem_trim):4.0f}"
                trav_formatter = ticker.FuncFormatter(trav_fu)
                ax.yaxis.set_major_formatter(trav_formatter)

                y_locator = ticker.MultipleLocator(200)
                ax.yaxis.set_major_locator(y_locator)
                
            self.logger.decrease_global_indent()

            # Format the figure
            xlabel = rf"Longitudinal Station (m)"
            ylabel = rf"Transverse Station (mm)"

            min_tick = np.amin(offset_ticks)
            max_tick = np.amax(offset_ticks)
            plt.xlim((min(0, min_tick), max(px_x, max_tick)))
            plt.ylim((0, px_y))

            plt.tight_layout()
            if plot_2m:
                title_str = rf"{exp_code} DEM 2m subsections with wall trim"
                if self.for_paper:
                    fig.subplots_adjust(top=0.9, left=0.05,
                                        bottom=0.075, right=0.95)
                    fontsize = 16
                    axs[-1].set_visible(False)
                    plt.colorbar(first_image, ax=axs[-1], use_gridspec=True)
                else:
                    fig.subplots_adjust(top=0.9, left=0.05,
                                        bottom=0.075, right=0.95)
                    fontsize = 16
            else:
                title_str = rf"{exp_code} DEM 8m with wall trim"
                if self.for_paper:
                    fig.subplots_adjust(top=0.97, left=0.1,
                                        bottom=0.05, right=0.95)
                    fontsize = 16
                    plt.colorbar(first_image, ax=list(axs), use_gridspec=True,
                            aspect=35, pad=0.01, anchor=(0.9, 0.5))
                    fig.subplots_adjust(right=0.9)
                else:
                    fig.subplots_adjust(top=0.9, left=0.1,
                                        bottom=0.1, right=0.95)
                    fontsize = 16

            # Make a title
            if not self.for_paper:
                plt.suptitle(title_str, fontsize=fontsize, usetex=True)

            # Set common x label
            fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                    fontsize=fontsize)

            # Set common y label
            fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                    fontsize=fontsize, rotation='vertical')

            # Save the figure
            filename = filename_base.format(exp_code)
            self.logger.write(f"Saving {filename}")
            self.save_figure(filename)

            #plt.show()
            #assert(False)


    def make_dem_semivariogram_plots(self):
        self.generic_make("dem semivariogram",
                self.omnimanager.reload_dem_data,
                self.plot_dem_semivariograms, figure_subdir='dem_variograms')

    def plot_dem_semivariograms(self):
        # Semivariogram are based on Curran Waters 2014
        # Can't use the generic_plot_experiments function because of the way I 
        # want to display and save them.
        max_xlag = 300 #px
        max_ylag = 50 #px
        plot_kwargs = {
                }
        dem_gather_kwargs = {
                'sta_lim'    : settings.stationing_2m,
                'wall_trim'  : settings.dem_wall_trim,
                }

        dem_data = self.gather_dem_data(dem_gather_kwargs)

        # Calculate period ranks
        ranking = {"rising" : 0, "falling" : 1,
                "50L" : 0, "62L" : 1, "75L" : 2, "87L" : 3, "100L" : 4}
        get_rank = lambda l, d: d + 2*l*(4-d)
        get_ax_index = lambda l, d: get_rank(*[ranking[k] for k in (l, d)])

        # Generate a base file name
        filename_base = f"dem_semivariograms_{{}}_{max_xlag}xlag-{max_ylag}ylag.png"

        # Start plotting; 1 plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code in exp_codes:
            self.logger.write(f"Creating semivariograms for {exp_code}")
            self.logger.increase_global_indent()
            
            # Make buffer room for labels in fig window
            btop    = 0.8
            bleft   = 0.04
            bbottom = 0.1
            bright  = 0.985
            figsize = self._get_figsize(max_xlag, max_ylag, 
                    xbuffer=bleft + 1 - bright, ybuffer=bbottom + 1 - btop)

            # Create the subplots for this experiment
            fig, axs = plt.subplots(3, 3, sharey=True, sharex=True,
                    figsize=figsize)
            axs = axs.flatten()

            # other parameters
            last_image=None
            levels_min, levels_max = (0,1) # contour levels min, max values
            xlabel = rf"l_x (px)"
            ylabel = rf"l_y (px)"
            title_str = rf"Semivariograms for experiment {exp_code} normalized by variance ($\sigma_z^2$)"
            fontsize = 16

            # Calculate and plot the semivariograms for this experiment
            for period_key, period_dem in dem_data[exp_code].items():
                limb, discharge, time, exp_time = period_key
                self.logger.write(f"Plotting {limb} {discharge} {time}")
                assert(time == 't60')
                ax = axs[get_ax_index(limb, discharge)]
                ax.axis('equal')
                ax.set_title(f"{limb} {discharge} {time}")

                # Calculate semivariogram
                x_coor, y_coor, semivariogram = self._calc_semivariogram(
                        period_dem, max_xlag, max_ylag)
                ## Debug code (fast data)
                #x_coor, y_coor = np.meshgrid(np.arange(-max_xlag, max_xlag+1),
                #                             np.arange(-max_ylag, max_ylag+1))

                ## Plot semivariogram
                levels = np.linspace(levels_min, levels_max, 40)
                last_image = ax.contourf(x_coor, y_coor, semivariogram,
                        levels=levels, cmap='Greys_r')
                ## Debug code (fast data)
                #last_image = ax.contourf(x_coor, y_coor, 
                #        np.abs(x_coor*y_coor)/np.max(x_coor*y_coor),
                #        levels=levels, cmap='Greys_r')

            self.logger.decrease_global_indent()

            # Format the figure
            plt.tight_layout()
            fig.subplots_adjust(top=btop, left=bleft,
                    bottom=bbottom, right=bright)
            if max_xlag >= max_ylag:
                plt.ylim((-max_ylag, max_ylag))
                plt.colorbar(last_image, orientation='horizontal',
                        ticks=np.linspace(levels_min, levels_max, 6))
            else:
                plt.xlim((-max_xlag, max_xlag))
                plt.colorbar(last_image, orientation='vertical',
                        ticks=np.linspace(levels_min, levels_max, 6))
            
            # Make a title
            plt.suptitle(title_str, fontsize=fontsize, usetex=True)

            # Set common x label
            fig.text(0.5, 0.01, xlabel, ha='center', usetex=True,
                    fontsize=fontsize)

            # Set common y label
            fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                    fontsize=fontsize, rotation='vertical')
            axs[-1].set_visible(False)

            # Save the figure
            filename = filename_base.format(exp_code)
            self.logger.write(f"Saving {filename}")
            self.save_figure(filename)

            break

    def _calc_semivariogram(self, dem, x_pxlag_max, y_pxlag_max, normalize=True):
        # Create a semivariogram from the dem
        # x_pxlag_max and y_pxlag_max are the largest pixel lags to calculate
        # Will calculate from -lag_max to +lag_max in both dimensions
        # x is parallel to flow, y is perpendicular to flow
        
        nx = x_pxlag_max
        ny = y_pxlag_max

        x, y = np.meshgrid(np.arange(-nx, nx+1), np.arange(-ny, ny+1))
        # Subset the right half (quadrants I and IV)
        x_half = x[:, nx:]
        y_half = y[:, nx:]
        half_svg = np.empty_like(x_half, dtype=np.float)

        iter = np.nditer([x_half, y_half, half_svg],
                flags = ['buffered', 'multi_index'],
                op_flags = [['readonly'], ['readonly'],
                            ['writeonly', 'allocate', 'no_broadcast']])
        
        for lx, ly, v in iter:
            # lx, ly are the lag coordinates (in pixels) of a dem element.
            # Calculate for quadrants I and IV -ny < y < ny and 0 <= x < nx
            # Quadrants II and III are symmetric
            semivariance = self._calc_semivariance(dem, lx, ly)
            v[...] = semivariance

        if normalize:
            variance = np.var(dem)
            half_svg /= variance

        # Rotate and trim the column on the y axis (x=0)
        rot_half_svg = half_svg[::-1, :0:-1]
        # Concatenate with the original half to create the full image
        semivariogram = np.concatenate((rot_half_svg, half_svg), axis=1)

        return x, y, semivariogram

        #plt.figure()
        #plt.hist(half_svg.flatten(), bins=40)
        #plt.figure()
        #plt.imshow(dem)
        #plt.figure()
        #levels = np.linspace(0,1,40)
        #plt.contourf(x, y, semivariogram, levels=levels, cmap='Greys_r')
        #plt.colorbar()
        ##plt.imshow(half_svg)
        #plt.show()
        #assert(False)

    def _calc_semivariance(self, dem, x_pxlag=0, y_pxlag=0):
        # Calculate the semivariance for the given 2D lag distances
        # dem should be an mxn numpy array. It should be oriented so upstream 
        # is positive x.
        # x_pxlag is the pixel lag distance parallel to the flume
        # y_pxlag is the pixel lag distance perpendicular to the flume
        # returns an np array of [x_pxlag, y_pxlag, semivar]
        #
        # x_pxlag and y_pxlag can be negative, but the semivariance function is 
        # rotationally symmetric so it might be wasted calculations
        #
        # Based on equation from Curran and Waters 2014
        # semivar (lagx, lagy) = sum[i=1 -> N-n; j=1 -> M-m] (
        #  z(xi + lagx, yj + lagy) - z(xi, yj))**2 / (2(N-n)(M-m))
        #
        # Converted to:
        # semivar (lagx, lagy) = sum(
        # (dem[lagx, lagy subset] - dem[origin subset])**2 ) / (2(N-n)(M-m))

        # Get size of dem
        M, N = dem.shape # M = rows, N = cols
        nx = abs(x_pxlag)
        ny = abs(y_pxlag)
        if ny >= M or nx >= N:
            self.logger.write("Lag out of bounds")
            self.logger.write([f"# rows, # columns = {M, N}",
                f"y_pxlag, x_pxlag = {y_pxlag, x_pxlag}"], local_indent=1)
            assert(False)
            #return np.nan

        ## Get index coordinates for offset dem subsets
        # Can handle positive and negative lag values
        #
        # Calculate start and end offsets for subarray A
        s_fu = lambda l: int((abs(l) - l) / 2)
        e_fu = lambda l: int((abs(l) + l) / 2)
        # Start corner of subarray A
        sx = s_fu(x_pxlag)
        sy = s_fu(y_pxlag)
        # End corner of subarray A
        ex = e_fu(x_pxlag)
        ey = e_fu(y_pxlag)

        # Get the offset dem subsets
        # Remember, slicing is [rows, cols]
        dem_A = dem[sy : M-ey,  sx : N-ex] # starts at origin if both lags >= 0
        dem_B = dem[ey : M-sy,  ex : N-sx]
        
        # Calculate the deviation squared.
        deviations = (dem_B - dem_A)**2

        # Calculate the semivariance
        denominator = 2 * (N - nx) * (M - ny)
        semivar = np.sum(deviations) / denominator

        #print(f"x_lag, y_lag = {x_pxlag, y_pxlag}")
        #print(f"dem_A = dem[{sy} : {M - ey},  {sx} : {N - ex}]")
        #print(f"dem_B = dem[{ey} : {M - sy},  {ex} : {N - sx}]")
        #print(f"denominator = 2 * ({N - nx}) * ({M - ny}) = {denominator}")
        #print(f"semivar = {semivar}")
        #print()
        #assert(False)

        return semivar

    def _get_figsize(self, xmax, ymax, xbuffer=0, ybuffer=0):
        aspect_ratio = xmax / ymax
        xbuff_ratio = (1 - xbuffer)
        ybuff_ratio = (1 - ybuffer)
        # largest allowed for my screen accounting for buffer
        imgx = 19.0 * xbuff_ratio
        imgy = 10.0 * ybuff_ratio
        
        # aspect corrected size
        ax = imgy * aspect_ratio
        ay = imgx / aspect_ratio

        # pick the one that fits and correct back to full size
        figx = (imgx if ax > imgx else ax) / xbuff_ratio
        figy = (imgy if ay > imgy else ay) / ybuff_ratio

        return (figx, figy)


    def make_dem_stats_plots(self):
        self.generic_make("dem stats time",
                self.omnimanager.reload_dem_data,
                self.plot_dem_stats, figure_subdir='dem_stats')

    def plot_dem_stats(self):
        # Stats are based on Curran Waters 2014
        # Do stuff before plot loop
        x_name = 'exp_time'
        # y_name options:
        y_names = ['mean', 'stddev', 'skewness', 'kurtosis']
        plot_kwargs = {
                'x'      : x_name,
                #'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        dem_gather_kwargs = {
                'sta_lim'    : settings.stationing_2m,
                'wall_trim'  : settings.dem_wall_trim,
                }

        dem_data = self.gather_dem_data(dem_gather_kwargs)

        # Calculate stats
        dem_stats = {}
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code in exp_codes:
            self.logger.write(f"Calculating stats for {exp_code}")
            dem_stats[exp_code] = self._calc_dem_stats(exp_code, dem_data)

        # Generate a base file name
        filename_x = x_name.replace('_', '-').lower()
        if 'sta_lim' in dem_gather_kwargs:
            sta_min, sta_max = dem_gather_kwargs['sta_lim']
            subset_str = f"_sta-{sta_min}-{sta_max}"
        else:
            subset_str = ''
        figure_name = f"dem_{{}}_v_{filename_x}{subset_str}.png"
        figure_name = ospath_join("dem_stats", figure_name)

        # Plot the 4 different stats
        for y_name in y_names:
            self.logger.write(f"Plotting {y_name}")
            self.logger.increase_global_indent()
            plot_kwargs['y'] = y_name
            filename_y = y_name.replace('_', '-').lower()

            # Start plot loop
            self.generic_plot_experiments(
                self._plot_dem_stats, self._format_dem_stats, 
                dem_stats, plot_kwargs, figure_name.format(filename_y))
            self.logger.decrease_global_indent()

    def _plot_dem_stats(self, exp_code, all_stats_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment

        # Not actually grouped, but can still use self.plot_group
        stats_data = all_stats_data[exp_code]
        self.plot_group(stats_data, plot_kwargs)

    def _format_dem_stats(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        y_name = plot_kwargs['y']
        y_labels = {
                'mean'     : ('Mean', '(mm)'),
                'stddev'   : ('Standard Deviation', '(mm)'),
                'skewness' : ('Skewness', ''),
                'kurtosis' : ('Kurtosis', ''),
                }
        name, units = y_labels[y_name]
        y_label = rf"{name}" + " {units}" if units else rf"{units}"
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : rf"Bed Elevation {y_label}",
                'title'         : rf"{name} of the detrended bed surface elevations for the 2m subsection",
                'legend_labels' : [rf"Elevation {name}"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)

    def _calc_dem_stats(self, exp_code, dem_data, kwargs={}):
        # Calculate the four different statistics from Curran Waters 2014

        # Keep ordered lists of the keys and resulting data
        # Will be converted later to pandas dataframe with multiindex
        key_limb = []
        key_discharge = []
        key_time = []
        exp_times = []

        dem_means = []
        dem_stddevs = []
        dem_skewnesses = []
        dem_kurtoses = []
        for period_key, period_dem in dem_data[exp_code].items():
            limb, discharge, time, exp_time = period_key
            key_limb.append(limb)
            key_discharge.append(discharge)
            key_time.append(time)
            exp_times.append(exp_time)

            # Calculate overall mean elevation
            mean = np.mean(period_dem)

            # Calculate bed elevation variance (std dev squared)
            # Can I simply use overall mean?? Eq uses mean value for that 
            # location?
            deviation = period_dem - mean
            variance = np.mean(deviation**2)
            stddev = np.sqrt(variance)
            #stddev2 = np.nanstd(period_dem) # Same as above

            # Calculate skewness
            n_points = period_dem.size
            skewness = np.sum(deviation**3) / (n_points * stddev**3)

            # Calculate kurtosis
            kurtosis = np.sum(deviation**4) / (n_points * stddev**4) - 3

            ## Some debugging code
            #print(exp_code, period_key)
            ##print(period_dem[::50, ::100])
            #print(f"Mean = {mean}")
            #print(f"Std dev = {stddev}")
            #print(f"Skewness = {skewness}")
            #print(f"Kurtosis = {kurtosis}")
            #plt.figure(10)
            #plt.imshow(period_dem)
            #plt.figure(20)
            #plt.hist(period_dem.flatten(), 50, normed=True)
            #plt.show(block=False)
            #sleep(0.5)
            #plt.close('all')

            # Add it to the lists
            dem_means.append(mean)
            dem_stddevs.append(stddev)
            dem_skewnesses.append(skewness)
            dem_kurtoses.append(kurtosis)

        # Create the dataframe
        mindex = pd.MultiIndex.from_arrays([key_limb, key_discharge, key_time],
                names=['limb', 'discharge', 'time'])
        stats_df = pd.DataFrame({'exp_time' : exp_times,
                                'mean'      : dem_means,
                                'stddev'    : dem_stddevs,
                                'skewness'  : dem_skewnesses,
                                'kurtosis'  : dem_kurtoses,
                                },
                index=mindex)
        return stats_df


    def make_dem_roughness_plots(self):
        self.logger.write([f"Making dem roughness time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_dem_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_dem_roughness,
                before_msg=f"Plotting dem roughness",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_dem_roughness(self):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'stddev'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        dem_gather_kwargs = {
                }

        dem_data = self.gather_dem_data(dem_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        figure_name = f"dem_{filename_y}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_dem_roughness, self._format_dem_roughness, 
            dem_data, plot_kwargs, figure_name)

    def _plot_dem_roughness(self, exp_code, dem_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        key_limb = []
        key_discharge = []
        key_time = []
        stddev = []
        exp_times = []
        for period_key, period_dem in dem_data[exp_code].items():
            stddev.append(np.nanstd(period_dem))
            limb, discharge, time, exp_time = period_key
            key_limb.append(limb)
            key_discharge.append(discharge)
            key_time.append(time)
            exp_times.append(exp_time)

        mindex = pd.MultiIndex.from_arrays([key_limb, key_discharge, key_time],
                names=['limb', 'discharge', 'time'])
        stats_df = pd.DataFrame({'stddev' : stddev, 'exp_time' : exp_times},
                index=mindex)

        # Not actually grouped, but can still use self.plot_group
        self.plot_group(stats_df, plot_kwargs)

    def _format_dem_roughness(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Bed Elevation Standard Deviation (mm)",
                'title'         : r"Standard deviations of the detrended bed surface elevations",
                'legend_labels' : [r"Elevation Std dev"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def gather_dem_data(self, kwargs):
        # Gather all the dem data into a dict of data dicts separated by 
        # exp_code = { exp_code : {(limb, flow, time) : data}}
        self.dem_data_all = {}
        self.omnimanager.apply_to_periods(self._gather_dem_data, kwargs)
        #plt.show()
        #assert(False)

        return self.dem_data_all

    def _gather_dem_data(self, period, kwargs):
        # Have periods add themselves to the overall dem dict
        exp_code = period.exp_code

        data = period.dem_data
        if data is not None:
            dem_res = settings.dem_resolution # mm/px
            #data_copy = data.copy()
            if 'sta_lim' in kwargs:
                # Throw away data outside the stationing limits
                # sta_lim = stationing for a subsection of the dem
                dem_offset = settings.dem_long_offset
                index_lim = [(x - dem_offset) // dem_res for x in kwargs['sta_lim']]
                idx_min, idx_max = index_lim
                data = data[:, idx_min:idx_max]

            if 'wall_trim' in kwargs:
                # Throw away data too close to the wall
                trim = kwargs['wall_trim']
                n_trim_rows = trim // dem_res
                data = data[n_trim_rows:-n_trim_rows, :]

            key = (period.limb, period.discharge, period.period_end,
                    period.exp_time_end)
            
            # Debugging code
            #if exp_code == '1A':# and period.limb == 'rising' and period.discharge == '87L':
            #    print(exp_code, key)
            #    #f1 = plt.figure(1)
            #    #plt.imshow(data_copy)
            #    fig = plt.figure()
            #    #plt.imshow(data)
            #    plt.hist(data.flatten(), 50, normed=True)
            #    plt.title(f"{exp_code} {key}")
            #    plt.xlim((120, 220))
            #    #plt.show()

            if exp_code in self.dem_data_all:
                self.dem_data_all[exp_code][key] = data
            else:
                self.dem_data_all[exp_code] = {key : data}


    # Functions to plot only manual data
    def make_mobility_plots(self, t_interval='period'):
        # Plot mobility 
        assert(t_interval in ['period', 'step'])
        plot_fu_kwargs = {
                't_interval' : t_interval
                }
        def reload_fu():
            self.omnimanager.reload_gsd_data()
            self.omnimanager.reload_sieve_data()
        self.generic_make("mobility",
                reload_fu,
                self.plot_mobility, plot_fu_kwargs=plot_fu_kwargs,
                fig_subdir='synthesis')

    def plot_mobility(self, t_interval = 'D50'):
        # Do stuff before plot loop
        sizes = [0.5, 0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8, 11.2, 16, 22.3, 32]
        x_name = 'Dgm'
        plot_kwargs = {
                'x'      : x_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        # Add to plot_kwargs as a hacky way to get info to _plot function
        plot_kwargs['t_interval'] = t_interval

        # Collect the data
        gsd_gather_kwargs = {
                'columns'   : sizes,
                'new_index' : ['exp_time', 'sta_str'],
                }
        data = {
                'gsd' : self.gather_gsd_data(gsd_gather_kwargs),
                'sieve' : self.gather_sieve_data({'columns' : sizes}),
                }

        # Make a filename
        filename_x = x_name.replace('_', '-').lower()
        figure_name = f"mobility_{t_interval}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_mobility, self._format_mobility, 
            data, plot_kwargs, figure_name)

    def _plot_mobility(self, exp_code, data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        ax = plot_kwargs['ax']
        plot_kwargs = plot_kwargs.copy()
        t_interval = plot_kwargs.pop('t_interval')

        # Get the data
        surf_gsd_data = data['gsd'][exp_code]
        bedload_gsd_data = data['sieve'][exp_code]

        # Collapse the surf_gsd data to just time
        surf_gsd_data = surf_gsd_data.groupby(level='exp_time').sum()

        # Partially collapse time if desired
        dt = 60 if t_interval == 'step' else 20
        bins = np.arange(60, 8*60+dt, dt)
        def sum_interval(data):
            intervals = np.digitize(data.index, bins, right=True)
            summed = data.groupby(intervals).sum()
            bins_used = [bins[i] for i in summed.index.values]
            summed.index = pd.Index(bins_used, name='exp_time')
            return summed
        surf_gsd_data = sum_interval(surf_gsd_data)
        bedload_gsd_data = sum_interval(bedload_gsd_data)

        # Make Dgm
        surf_col_values = surf_gsd_data.columns.values
        bedload_col_values = bedload_gsd_data.columns.values
        assert(np.all(surf_col_values == bedload_col_values))
        upper = surf_col_values
        lower = np.roll(upper, 1)
        lower[0] = 0.001
        Dgm = np.exp2((np.log2(upper) + np.log2(lower))/2)

        # surf is by count, bedload is by mass.
        # approximate mass per rock based on geometric mean size
        # assume sphere?
        ### How to convert them properly???
        Dg_mass_np = settings.sediment_density * 4000/3 * np.pi *(Dgm/2000)**3
        #[print(f"{m:10.5f} -> {d:10.2f} mm") for m, d in zip(Dg_mass, Dgm)]
        Dg_mass = pd.Series(Dg_mass_np, index=surf_gsd_data.columns)
        surf_gsd_mass = surf_gsd_data.multiply(Dg_mass)

        # Convert both to fractional values
        def make_fractional(data):
            return data.div(data.sum(axis=1), axis=0)
        surf_fractional = make_fractional(surf_gsd_mass)
        bedload_fractional = make_fractional(bedload_gsd_data)

        # Calculate mobility
        mobility = bedload_fractional / surf_fractional
        mobility[surf_fractional == 0] = np.nan

        # Change index and columns
        exp_times = mobility.index.values
        mobility.index = pd.Index(exp_times/60, name='exp_time_hrs')
        mobility.columns = pd.Index(Dgm, name='Dgm')

        # Plot it
        plot_kwargs['y'] = bins/60
        if t_interval == 'step':
            # only 8 lines, give each line a color
            plot_kwargs['marker'] = 'x'
            if exp_code == '5A':
                # Bad form.... but watevs
                plot_kwargs['legend'] = True
            color_idx = np.linspace(0, 1, num=len(exp_times))
            colorset = [plt.cm.cool_r(i) for i in color_idx]
            plot_kwargs['color'] = colorset
        else:
            # Too many lines, plot points instead
            plot_kwargs['linestyle'] = 'None'
            plot_kwargs['marker'] = '.'
            plot_kwargs['color'] = 'k'

        mobility = (mobility.T).reset_index()
        mobility.plot(**plot_kwargs)
        #ax.set_ylim((0, 4))
        ax.set_xlabel('')
        self.generic_set_grid(ax)
        #self.plot_group(mobility.T, plot_kwargs)

    def _format_mobility(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Geometric grain size (mm)",
                'ylabel'        : r"$\frac{P_i}{f_i}$",
                'title'         : r"Mobility of each grain size class",
                'legend_labels' : [r"mobility"],
                #'legend_labels' : [r"Shear from surface"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_loc_shear_plots(self):
        self.logger.write([f"Making shear time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_depth_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_loc_shear,
                before_msg=f"Plotting shear vs time",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_loc_shear(self):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'shear'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        depth_gather_kwargs = {
                'new_index' : ['exp_time', 'location'],
                }

        depth_data = self.gather_depth_data(depth_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        figure_name = f"loc-mean_surf_shear_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_loc_shear, self._format_loc_shear, 
            depth_data, plot_kwargs, figure_name)

    def _plot_loc_shear(self, exp_code, depth_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        ax = plot_kwargs['ax']

        draw_box = False

        # Get the data
        exp_data = depth_data[exp_code] * cm2m

        # Remove null columns
        null_cols = exp_data.isnull().all(axis=0)
        exp_data = exp_data.loc[:, ~null_cols]

        surf = exp_data.xs('surface', level='location')
        bed = exp_data.xs('bed', level='location')
        depth = exp_data.xs('depth', level='location')

        # Calc flume-averaged depth
        #flume_mean_depth = depth.mean(axis=1)

        # Use adjacent point slopes to calc avg slopes at each position
        stations = np.sort(depth.columns.values)
        head_sta = stations[1:]
        tail_sta = stations[0:-1]
        dx = head_sta - tail_sta
        avg_positions = (head_sta + tail_sta) / 2
        avg_depths = (depth.loc[:, head_sta].values +\
                depth.loc[:, tail_sta].values)/2
        
        # T = rho * g * R * S
        rho = settings.water_density
        g = settings.gravity
        w = settings.flume_width

        #R = w * flume_mean_depth / (w + 2 * flume_mean_depth)
        R = w * avg_depths / (w + 2 * avg_depths)

        for loc_data in surf,:#, bed:
            times = loc_data.index
            # Use flume-averaged slope
            #ols_stats, flume_elevations = self._calc_retrended_slope(loc_data)
            #S = ols_stats['slope']
            
            # Use position avg slope
            head = loc_data.loc[:, head_sta].values
            tail = loc_data.loc[:, tail_sta].values
            dy = head - tail

            slope = dy / dx + settings.flume_slope
            #S = pd.DataFrame(slope, index=loc_data.index, columns=avg_positions)
            S = slope

            #shear_series = rho * g * R * S
            #shear = shear_series.to_frame(name='shear')
            shear_np = rho * g * R * S
            shear = pd.DataFrame(shear_np, index=times, columns=avg_positions)

            time_corrected = self._time_plot_prep(shear, plot_kwargs, auto_plot=False)
            final_data = time_corrected.set_index('exp_time').astype(np.float64)

            medians = final_data.median(axis=1)
            means = final_data.mean(axis=1)
            stat_line = means
            if draw_box:
                # Box plot w lines
                final_data.T.boxplot(ax=ax)#, showfliers=False)

                # Plot a line through the box plots
                x = np.arange(len(stat_line.index)) + 1
                y = stat_line.values
                ax.plot(x, y)
            else:
                # Just lines
                stat_line.plot(ax=ax)

        if draw_box:
            xticks_locs = np.arange(8)*3 + 1.5
            xticks_labels = np.arange(8) + 1
            ax.set_xticks(xticks_locs)
            ax.set_xticklabels(xticks_labels)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.set_ylim([-15,35])
        ax.get_xaxis().get_label().set_visible(False)

    def _format_loc_shear(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Shear stress ($N/m^2$)",
                'title'         : r"Shear stress based on surface slope at each measurement location",
                #'legend_labels' : [r"Shear from surface", r"Shear from bed"],
                'legend_labels' : [r"Shear from surface"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)
        #ax0_lines = axs[0,0].get_lines()
        #fig.legend(handles=ax0_lines, labels=['Mean shear'],
        #            loc='center right')


    def make_shields_plots(self, plot_2m=True, Di_target='D50'):
        # Plot shields 
        plot_fu_kwargs = {
                'plot_2m' : plot_2m,
                'Di_target' : Di_target
                }
        def reload_fu():
            self.omnimanager.reload_depth_data()
            self.omnimanager.reload_sieve_data()
        self.generic_make("shields",
                reload_fu,
                self.plot_shields, plot_fu_kwargs=plot_fu_kwargs,
                fig_subdir='synthesis')

    def plot_shields(self, plot_2m=True, Di_target = 'D50'):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'shields'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        # Add to plot_kwargs as a hacky way to get info to _plot function
        plot_kwargs['plot_2m'] = plot_2m
        plot_kwargs['Di_target'] = Di_target

        # Collect the data
        depth_gather_kwargs = {
                'new_index' : ['exp_time', 'location'],
                }
        data = {
                'depth' : self.gather_depth_data(depth_gather_kwargs),
                'sieve' : self.gather_sieve_data({}),
                }

        # Make a filename
        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        plot_2m_str = '2m_' if plot_2m else ''
        figure_name = f"{plot_2m_str}{filename_y}_{Di_target}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_shields, self._format_shields, 
            data, plot_kwargs, figure_name)

    def _plot_shields(self, exp_code, data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        ax = plot_kwargs['ax']
        plot_kwargs = plot_kwargs.copy()
        plot_2m = plot_kwargs.pop('plot_2m')
        Di_target = plot_kwargs.pop('Di_target')

        # Get the data
        cm2m = 1/100
        depth_data = data['depth'][exp_code] * cm2m
        sieve_data = data['sieve'][exp_code]

        # surface elevations for flume avg slope
        surf = depth_data.xs('surface', level='location')

        # avg depth values in 2m subset.
        sta_min, sta_max = (4.5, 6.5) if plot_2m else (0, 8)
        sta_keep = [s for s in depth_data.columns if sta_min <= s <= sta_max]
        depth = depth_data.xs('depth', level='location').loc[:, sta_keep]
        avg_depths = depth.mean(axis=1)

        # T* = T / [(rhos - rhow) * g * D50
        # T = rhow * g * R * S
        # T* = [rhow * R * S] / [(rhos - rhow) * D]
        rhow = settings.water_density
        rhos = settings.sediment_density
        w = settings.flume_width

        R = w * avg_depths / (w + 2 * avg_depths)

        ols_stats, flume_elevations = self._calc_retrended_slope(surf)
        S = ols_stats['slope']

        Di = self.calc_Di(sieve_data, target_Di=Di_target) / 1000 # mm2m
        
        RS = R * S
        RS.index = RS.index + 10
        shields_series = (rhow * RS) / ((rhos - rhow) * Di)
        shields = shields_series.to_frame(name='shields')

        self.plot_group(shields, plot_kwargs)

    def _format_shields(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        Di_target = plot_kwargs['Di_target']
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"$\tau^*$",
                'title'         : r"Shields stress from bedload {Di_target}",
                'legend_labels' : [r"Shields stress"],
                #'legend_labels' : [r"Shear from surface"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_flume_shear_plots(self):
        self.logger.write([f"Making flume-averaged shear time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_depth_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_flume_shear,
                before_msg=f"Plotting shear vs time",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_flume_shear(self):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'shear'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        depth_gather_kwargs = {
                'new_index' : ['exp_time', 'location'],
                }

        depth_data = self.gather_depth_data(depth_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        figure_name = f"comp_flume-avg_shear_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_flume_shear, self._format_flume_shear, 
            depth_data, plot_kwargs, figure_name)

    def _plot_flume_shear(self, exp_code, depth_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        ax = plot_kwargs['ax']
        cm2m = 1/100

        # Get the data
        exp_data = depth_data[exp_code] * cm2m

        surf = exp_data.xs('surface', level='location')
        bed = exp_data.xs('bed', level='location')
        depth = exp_data.xs('depth', level='location')

        # Calc flume-averaged depth
        avg_depths = depth.mean(axis=1)

        # T = rho * g * R * S
        rho = settings.water_density
        g = settings.gravity
        w = settings.flume_width

        #R = w * flume_mean_depth / (w + 2 * flume_mean_depth)
        R = w * avg_depths / (w + 2 * avg_depths)

        for loc_data in surf, bed:
            # Use flume-averaged slope
            ols_stats, flume_elevations = self._calc_retrended_slope(loc_data)
            S = ols_stats['slope']
            
            shear_series = rho * g * R * S
            shear = shear_series.to_frame(name='shear')
            
            # Not actually grouped, but can still use self.plot_group
            self.plot_group(shear, plot_kwargs)

    def _format_flume_shear(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Shear stress ($N/m^2$)",
                'title'         : r"Flume-averaged shear stress based on water surface and bed surface slopes",
                'legend_labels' : [r"Shear from water slope", r"Shear from bed slope"],
                #'legend_labels' : [r"Shear from surface"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_avg_slope_plots(self, plot_2m=True, plot_trends='both'):
        plot_fu_kwargs = {
                'plot_2m' : plot_2m,
                'plot_trends' : plot_trends,
                }
        self.generic_make("slope",
                self.omnimanager.reload_depth_data,
                self.plot_avg_slope, plot_fu_kwargs=plot_fu_kwargs,
                fig_subdir='depth')

    def plot_avg_slope(self, plot_2m=True, plot_trends='both'):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'slope'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }

        # Add to plot_kwargs as a hacky way to get info to _plot function
        plot_kwargs['plot_2m'] = plot_2m
        plot_kwargs['plot_trends'] = plot_trends

        depth_gather_kwargs = {
                'new_index' : ['exp_time', 'location'],
                }

        depth_data = self.gather_depth_data(depth_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        plot_2m_str = '2m_' if plot_2m else ''
        figure_name = f"{plot_2m_str}avg-slope_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_avg_slope, self._format_avg_slope, 
            depth_data, plot_kwargs, figure_name)

    def _plot_avg_slope(self, exp_code, depth_data, plot_kwargs):
        ax = plot_kwargs['ax']
        plot_kwargs = plot_kwargs.copy()
        plot_2m = plot_kwargs.pop('plot_2m')
        plot_trends = plot_kwargs.pop('plot_trends')

        cm2m = 1/100

        # Get the data
        exp_data = depth_data[exp_code] * cm2m
        #stats_data_frames = {}

        sta_min, sta_max = (4.5, 6.5) if plot_2m else (0, 8)
        sta_keep = [s for s in exp_data.columns if sta_min <= s <= sta_max]

        init_bed_height = settings.init_bed_height * cm2m
        flume_elevations = None

        #slice_all = slice(None)
        plot_locations = ['surface', 'bed'] if plot_trends == 'both' else [plot_trends]
        for location in plot_locations:
            loc_data = exp_data.xs(location, level='location')
            data = loc_data.loc[:, sta_keep]
            notnull = data.notnull()
            data[notnull.sum(axis=1) <=3] = np.nan # require > 3 pts

            if plot_2m:
                intercept = None
            elif location == 'bed':
                # Fix intercept to flume lip
                intercept = init_bed_height
            elif location == 'surface':
                # Fix intercept to flume lip + water depth at sta 2 m
                #raise NotImplementedError
                #depth = exp_data.xs('depth', level='location')
                #last_depth = depth.loc[:, 2.0]
                #print(last_depth)
                #assert(False)
                #intercept = init_bed_height + last_depth
                intercept = None
            else:
                assert(False)

            #stats_data_frames[location] = ols_out
            
            ols_out, flume_elevations = self._calc_retrended_slope(data, flume_elevations, intercept)
            ### NOTE 2m slope is very chaotic.
            ### 8m slope seems to be more reliable.
            #if exp_code != '1A':
            #    ols_out2, flume_elevations2 = self._calc_retrended_slope(loc_data, None, intercept)
            #    t = 270
            #    for t in ols_out.index.values:
            #        plt.figure()
            #        ax = plt.gca()

            #        # plot all z
            #        t_z = (loc_data.loc[t, :])
            #        true_z = t_z + flume_elevations2
            #        plt.plot(true_z.index.values, true_z.values, 'o',
            #                linestyle='None')

            #        # plot overall s
            #        all_s = (ols_out2.loc[t, 'slope'])
            #        all_b = (ols_out2.loc[t, 'intercept'])
            #        all_cols = loc_data.columns.values
            #        all_x = [all_cols[0], all_cols[-1]]
            #        all_y  = [all_s*x + all_b for x in all_x]
            #        plt.plot(all_x, all_y)
            #        
            #        # plot local s
            #        local_s = (ols_out.loc[t, 'slope'])
            #        local_b = (ols_out.loc[t, 'intercept'])
            #        local_cols = data.columns.values
            #        local_x = [local_cols[0], local_cols[-1]]
            #        local_y  = [local_s*x + local_b for x in local_x]
            #        plt.plot(local_x, local_y)

            #        ax.set_xlim((2,8))
            #        ax.set_ylim((0.3, 0.45))
            #        print(local_s, all_s)
            #        plt.show()
            #    #assert(False)
            # Not actually grouped, but can still use self.plot_group
            self.plot_group(ols_out, plot_kwargs)

        ax.locator_params(axis='y', steps=[4, ], min_n_ticks=3)
        self.generic_set_grid(ax)
        #stats_data = pd.concat(stats_data_frames,
        #        names=['location', 'exp_time'])
        #grouped = stats_data.groupby(level='location')
        #self.plot_group(grouped, plot_kwargs)

    def _format_avg_slope(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Slope (m/m)",
                'title'         : r"Flume-averaged water surface and bed surface slopes",
                'legend_labels' : [r"Water", r"Bed"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_avg_depth_plots(self, plot_2m=True):
        self.generic_make("depth",
                self.omnimanager.reload_depth_data,
                self.plot_avg_depth, plot_fu_kwargs={'plot_2m' : plot_2m},
                fig_subdir='depth')

    def plot_avg_depth(self, plot_2m=True):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'depth'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }

        # Add to plot_kwargs as a hacky way to get info to _plot function
        plot_kwargs['plot_2m'] = plot_2m

        depth_gather_kwargs = {
                'new_index' : ['exp_time', 'location'],
                }

        depth_data = self.gather_depth_data(depth_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        #figure_name = f"box_flume-depth_v_{filename_x}.png"
        plot_2m_str = '2m_' if plot_2m else ''
        figure_name = f"{plot_2m_str}flume-depth_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_avg_depth, self._format_avg_depth, 
            depth_data, plot_kwargs, figure_name)

    def _plot_avg_depth(self, exp_code, depth_data, plot_kwargs):
        # Do stuff during plot loop
        ax = plot_kwargs['ax']
        plot_kwargs = plot_kwargs.copy()
        plot_2m = plot_kwargs.pop('plot_2m')

        draw_boxplot = False
        sta_min, sta_max = (4.5, 6.5) if plot_2m else (0, 8)

        # Get the data
        exp_data = depth_data[exp_code]
        sta_keep = [s for s in exp_data.columns if sta_min <= s <= sta_max]
        all_depths = exp_data.xs('depth', level='location')
        depths = all_depths.loc[:, sta_keep]

        if draw_boxplot:
            depths.T.boxplot(ax=plot_kwargs['ax'], showfliers=False)
        else:
            avg_depths = depths.mean(axis=1)
            avg_depths.rename('depth', inplace=True)

            # Plot depth values in background
            d_kwargs = {'x' : plot_kwargs['x'], 'ax' : ax,
                    'linestyle' : 'None', 'legend' : False,
                    'marker' : '.', 'color' : 'silver'}
            self.plot_group(depths, d_kwargs)

            # Plot avg depth
            self.plot_group(avg_depths, plot_kwargs)
            #ax.set_xlim((0,8))

            ### NOTE 2m depths are reasonably close the average depth
            #if exp_code != '1A':
            #    for t in depths.index.values:
            #        plt.figure()
            #        ax = plt.gca()

            #        local_tz = depths.loc[t, :]
            #        all_tz = all_depths.loc[t, :]

            #        # plot all z
            #        plt.plot(all_tz.index.values, all_tz.values, 'o',
            #                linestyle='None')

            #        # plot all avg z
            #        all_avg = all_tz.mean()
            #        ax.axhline(all_avg)

            #        # plot local avg
            #        local_avg = local_tz.mean()
            #        local_cols = depths.columns.values
            #        local_x = [local_cols[0], local_cols[-1]]
            #        local_y  = [local_avg, local_avg]
            #        plt.plot(local_x, local_y)

            #        print(f"{t}: {all_avg:0.2f}, {local_avg:0.2f}")
            #        ax.set_xlim((2, 8))
            #        ax.set_ylim((3, 15))
            #        plt.show()

            self.generic_set_grid(ax)

    def _format_avg_depth(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Depth (cm)",
                'title'         : r"Flume-averaged water depth",
                'legend_labels' : [r"Water depth"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_simple_feed_plots(self):
        self.generic_make("feed",
                self.omnimanager.reload_feed_data,
                self.plot_simple_feed_plots, fig_subdir='feed_sieve')

    def plot_simple_feed_plots(self):
        # Can't use the generic_plot_experiments function
        plot_kwargs = {
                }

        feed_data_dict = {}
        for exp_code in self.omnimanager.experiments.keys():
            experiment = self.omnimanager.experiments[exp_code]

            data = experiment.feed_data
            if data is None:
                continue
            for sample in data.index:
                feed_data_dict[(exp_code, sample)] = data.loc[sample, :].copy()
        feed_data = pd.DataFrame.from_dict(feed_data_dict)
        
        cumsum = feed_data.cumsum()
        frac = cumsum / feed_data.sum()

        D50 = self.calc_Di(feed_data.T, target_Di=50)
        D84 = self.calc_Di(feed_data.T, target_Di=84)

        print(frac)
        print(D50)
        print(f"Mean {D50.mean()} mm")

        sum_data = feed_data.sum(axis=1)
        sum_frac = sum_data.cumsum() / sum_data.sum()
        sum_D16 = self.calc_Di(pd.DataFrame({'sum' : sum_data}).T, target_Di=16)
        sum_D50 = self.calc_Di(pd.DataFrame({'sum' : sum_data}).T, target_Di=50)
        sum_D84 = self.calc_Di(pd.DataFrame({'sum' : sum_data}).T, target_Di=84)

        print(sum_frac)
        print(f"D16 of samples sum {sum_D16} mm")
        print(f"D50 of samples sum {sum_D50} mm")
        print(f"D84 of samples sum {sum_D84} mm")

        n = feed_data.shape[1]
        title = f"Overall feed distribution ({n} samples)"
        #self.plot_distribution(frac, hlines=[0.5, 0.84], title=title)
        self.plot_distribution(sum_frac, title=title)

        # Save the figure
        filename = f"simple_feed_sum_distributions_logx.png"
        #filename = f"simple_feed_sum_distributions.png"
        self.save_figure(filename)

        plt.show()


    def make_simple_masses_plots(self):
        self.generic_make("total mass vs time",
                self.omnimanager.reload_masses_data,
                self.plot_simple_masses, fig_subdir='trap')

    def plot_simple_masses(self):
        x_name = 'exp_time'
        y_name = 'total dry (kg)'
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'scatter',
                'legend' : False,
                }
        masses_gather_kwargs = {
                'columns'   : plot_kwargs['y']
                }

        masses_data = self.gather_masses_data(masses_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace(' ', '-').lower()
        filename_y = filename_y.replace('-(kg)', '')
        figure_name = f"simple_masses_{filename_y}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_simple_masses, self._format_simple_masses, 
            masses_data, plot_kwargs, figure_name)

    def _plot_simple_masses(self, exp_code, masses_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment

        # Not actually grouped, but can still use self.plot_group
        self.plot_group(masses_data[exp_code], plot_kwargs)
        self.generic_set_grid(plot_kwargs['ax'])

    def _format_simple_masses(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Total dry mass (kg)",
                'title'         : r"Total transported bedload mass at end of period",
                'legend_labels' : [r"Total dry mass (kg)"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_simple_sieve_plots(self):
        self.generic_make(" 11.2mm vs time",
                self.omnimanager.reload_sieve_data,
                self.plot_simple_sieve, fig_subdir='trap')

    def plot_simple_sieve(self):
        x_name = 'exp_time'
        y_name = 11.2
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        sieve_gather_kwargs = {
                'columns'   : plot_kwargs['y']
                }

        sieve_data = self.gather_sieve_data(sieve_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = f"{y_name}".replace('.', '-').lower()
        figure_name = f"simple_sieve_{filename_y}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_simple_sieve, self._format_simple_sieve, 
            sieve_data, plot_kwargs, figure_name)

    def _plot_simple_sieve(self, exp_code, sieve_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment

        # Not actually grouped, but can still use self.plot_group
        self.plot_group(sieve_data[exp_code], plot_kwargs)

    def _format_simple_sieve(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Mass (g)",
                'title'         : r"Total bedload of 11.2mm particles per period",
                'legend_labels' : [r"Mass (g)"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_sieve_di_plots(self, Di=50):
        self.generic_make(f" D{Di} vs time",
                self.omnimanager.reload_sieve_data,
                self.plot_sieve_di, plot_fu_kwargs={'Di' : Di}, fig_subdir='trap')

    def plot_sieve_di(self, **kwargs):
        x_name = 'exp_time'
        y_name = f"D{kwargs['Di']}"
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                #'kind'   : 'scatter',
                'legend' : False,
                }
        sieve_gather_kwargs = {
                }

        sieve_data = self.gather_sieve_data(sieve_gather_kwargs)

        filename_x = x_name.replace('_', '-').lower()
        filename_y = f"{y_name}".replace('.', '-').lower()
        figure_name = f"sieve_di_{filename_y}_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_sieve_di, self._format_sieve_di, 
            sieve_data, plot_kwargs, figure_name)

    def _plot_sieve_di(self, exp_code, sieve_data, plot_kwargs):
        # Do stuff during plot loop
        # Plot an experiment
        
        Di_name = plot_kwargs['y']
        target_Di = int(Di_name[1:])

        data = sieve_data[exp_code]
        data[Di_name] = self.calc_Di(data, target_Di)

        # Not actually grouped, but can still use self.plot_group
        self.plot_group(sieve_data[exp_code], plot_kwargs)
    
    def _format_sieve_di(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        y_name = plot_kwargs['y']
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Size (mm)",
                'title'         : rf"{y_name} of the bedload collected in the sediment trap",
                'legend_labels' : [r"Mass (g)"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def gather_depth_data(self, kwargs):
        # Gather all the depth data into a dict of dataframes separated by 
        # exp_code
        self.depth_data_frames = {}
        self.omnimanager.apply_to_periods(self._gather_depth_data, kwargs)

        # Combines the separate frames into one dataframe per experiment
        depth_data = {}
        for exp_code, frames in self.depth_data_frames.items():
            combined_data = pd.concat(frames)
            if 'new_index' in kwargs:
                sort_index = kwargs['new_index'][0]
                depth_data[exp_code] = combined_data.sort_index(level=sort_index)
            else:
                depth_data[exp_code] = combined_data
        return depth_data

    def _gather_depth_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall depth 
        # dataframe dict
        #cols = kwargs['columns']
        new_index = kwargs['new_index'] if 'new_index' in kwargs else None
        exp_code = period.exp_code

        data = period.depth_data
        if data is not None:
            index_names = list(data.index.names)
            if new_index is not None:
                drop_list = list(set(index_names) - set(new_index))
                data.reset_index(inplace=True)
                data.set_index(new_index, inplace=True)
                data.drop(drop_list, axis=1, inplace=True)
            if exp_code not in self.depth_data_frames:
                self.depth_data_frames[exp_code] = []
            self.depth_data_frames[exp_code].append(data.loc[:, :])

    def gather_masses_data(self, kwargs):
        # Gather all the masses data into a dict of dataframes separated by 
        # exp_code
        self.masses_data_frames = {}
        self.omnimanager.apply_to_periods(self._gather_masses_data, kwargs)

        # Combines the separate frames into one dataframe per experiment
        masses_data = {}
        for exp_code, frames in self.masses_data_frames.items():
            combined_data = pd.concat(frames)
            masses_data[exp_code] = combined_data.sort_index()
        return masses_data

    def _gather_masses_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall masses 
        # dataframe dict
        cols = kwargs['columns'] if 'columns' in kwargs else 'all'
        add_cols = kwargs['add_cols'] if 'add_cols' in kwargs else None
        exp_code = period.exp_code

        data = period.masses_data
        if data is not None:
            if exp_code not in self.masses_data_frames:
                self.masses_data_frames[exp_code] = []

            if add_cols is not None:
                skipped = []
                for col in add_cols:
                    if col == 'limb' and 'limb' not in data.columns:
                        data['limb'] = period.limb
                    elif col == 'discharge' and 'discharge' not in data.columns:
                        data['discharge'] = period.discharge_int
                    elif col == 'period_range' and 'period_range' not in data.columns:
                        data['period_range'] = period.period_range
                    elif col == 'exp_time_hrs' and 'exp_time_hrs' not in data.columns:
                        assert(data.index.name == 'exp_time')
                        data['exp_time_hrs'] = data.index.values / 60
                    else:
                        skipped.append(col)
                if len(skipped) > 0:
                    self.logger.write(f"Skipping columns {skipped} for gathering masses data")

            if cols == 'all':
                self.masses_data_frames[exp_code].append(data)
            else:
                self.masses_data_frames[exp_code].append(data.loc[:, cols])

    def gather_sieve_data(self, kwargs):
        # Gather all the sieve data into a dict of dataframes separated by 
        # exp_code
        cols = kwargs['columns'] if 'columns' in kwargs else None
        self.sieve_data_frames = {}
        self.omnimanager.apply_to_periods(self._gather_sieve_data, kwargs)

        # Combines the separate frames into one dataframe per experiment
        sieve_data = {}
        for exp_code, frames in self.sieve_data_frames.items():
            combined_data = pd.concat(frames)
            exp_data = combined_data.sort_index()
            if cols is not None:
                exp_data = exp_data.loc[:, cols]
            sieve_data[exp_code] = exp_data
        return sieve_data

    def _gather_sieve_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall sieve 
        # dataframe dict
        exp_code = period.exp_code
        add_cols = kwargs['add_cols'] if 'add_cols' in kwargs else None

        data = period.sieve_data
        if data is not None:
            if exp_code not in self.sieve_data_frames:
                self.sieve_data_frames[exp_code] = []

            if add_cols is not None:
                skipped = []
                for col in add_cols:
                    if col == 'limb' and 'limb' not in data.columns:
                        data['limb'] = period.limb
                    elif col == 'discharge' and 'discharge' not in data.columns:
                        data['discharge'] = period.discharge_int
                    elif col == 'period_range' and 'period_range' not in data.columns:
                        data['period_range'] = period.period_range
                    elif col == 'exp_time_hrs' and 'exp_time_hrs' not in data.columns:
                        assert(data.index.name == 'exp_time')
                        data['exp_time_hrs'] = data.index.values / 60
                    else:
                        skipped.append(col)
                if len(skipped) > 0:
                    self.logger.write(f"Skipping columns {skipped} for gathering masses data")

            self.sieve_data_frames[exp_code].append(data.loc[:, ])

    def calc_Di(self, data, target_Di=50):
        # Calculate the Di values for a dataframe of sieve masses
        # data should be a dataframe of raw masses per size class (size sorted 
        # smallest to largest)
        # target_Di is an integer between 0 and 100
        # 
        # returns series

        if isinstance(target_Di, str):
            target_Di = int(target_Di[1:])

        assert(0 < target_Di < 100)
        target = target_Di/100
        name = f"D{target_Di}"

        notnull_idx = data.notnull().all(axis=1)
        raw_data = data
        data = data.loc[notnull_idx, :]
        # Calculate cumulative curve and normalize
        cumsum = data.cumsum(axis=1)
        notnull_fractional = cumsum.divide(cumsum.iloc[:, -1], axis=0)

        # Make sure the target falls between datapoints on the distribution
        fines_okay = (notnull_fractional < target).any(axis=1)
        coarse_okay = (notnull_fractional > target).any(axis=1)
        okay_rows = fines_okay & coarse_okay
        fractional = notnull_fractional.loc[okay_rows, :]

        # interpolate the percentile
        # I CANNOT find a cleaner way to do this... Definitely not in Pandas.
        np_frac = fractional.values
        np_cols = fractional.columns.values

        np_equal = np_frac == target

        np_lesser = np_frac < target
        np_rlesser = np.roll(np_lesser, -1, axis=1)
        np_lower = np_lesser & ~np_rlesser # find True w/ False to right

        np_greater = np_frac > target
        np_rgreater = np.roll(np_greater, 1, axis=1)
        np_upper = np_greater & ~np_rgreater # find True w/ False to left

        lower_frac = np_frac[np_lower]
        upper_frac = np_frac[np_upper]
        lower = np_cols[np.argmax(np_lower, axis=1)] # lower size classes
        upper = np_cols[np.argmax(np_upper, axis=1)] # upper size classes
        equal = np_cols[np.argmax(np_equal, axis=1)] # equal size class
        equal_rows = np_equal.any(axis=1)

        lower_psi = np.log2(lower)
        upper_psi = np.log2(upper)

        Di_psi = lower_psi + (target - lower_frac) * (upper_psi - lower_psi) /\
                            (upper_frac - lower_frac)
        Di = 2**Di_psi
        Di[equal_rows] = equal[equal_rows]

        # Add the null values back in to make the array the same size
        notnull_fractional.loc[okay_rows, name] = Di
        out = pd.Series(index=raw_data.index, name=name)
        out.loc[notnull_idx] = notnull_fractional[name]

        return out


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
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            ax.set_title(f"{exp_code} {experiment.name}")

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
        figure_name = f"gsd_mean_{filename_y}_v_{filename_x}.png"
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
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)


    def make_box_gsd_plots(self, x_name='exp_time', y_name='D50'):
        name = 'time' if x_name =='exp_time' else 'station' if x_name == 'sta_str' else x_name
        self.logger.write([f"Making gsd {name} plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_gsd_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_box_gsd,
                kwargs={'x_name':x_name, 'y_name':y_name},
                before_msg=f"Plotting gsd vs {name}",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_box_gsd(self, y_name='D50', x_name='exp_time'):
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
        exp_codes = self.omnimanager.get_exp_codes()
        self.plot_labels = []
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            #if exp_code != '2B':
            #    ### Hacking stuff together
            #    ### Skip anything but 2B
            #    ### Because that's the only one with 2m data
            #    continue
            self.logger.write(f"Plotting experiment {exp_code}")

            #plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]

            # Get the data and group it by the line category
            exp_data = gsd_data[exp_code]
            idx = pd.IndexSlice
            exp_data = exp_data.loc[idx[:, ('sta-3000', 'sta-5000')], :]
            print(exp_data)
            assert(False)
            exp_data.reset_index(inplace=True)
            exp_data['exp_time'] = exp_data['exp_time'] / 60
            exp_data.set_index(gsd_gather_kwargs['new_index'], inplace=True)
            #grouped = exp_data.groupby(level=lines_category)
            exp_data.boxplot(column=y_name, by=x_name, ax=ax)
            ax.get_xaxis().get_label().set_visible(False)
            xticks_locs = ax.get_xticks()
            xticks_labels = ax.get_xticklabels()
            ax.set_xticks(xticks_locs[::3])
            ax.set_xticklabels(xticks_labels[::3])
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.set_title(f"{exp_code} {experiment.name}")

        self.format_box_gsd_figure(fig, axs, plot_kwargs)

        # Generate a figure name and save the figure
        filename_y = y_name.replace('_', '-').lower()
        filename_x = x_name.replace('_', '-').lower()
        ### Hacking together the 2m part
        figure_name = f"gsd_box_{filename_y}_v_{filename_x}.png"
        self.save_figure(figure_name)
        plt.show()

    def format_box_gsd_figure(self, fig, axs, plot_kwargs):
        x = plot_kwargs['x']
        y = plot_kwargs['y']

        # Set the spacing and area of the subplots
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, left=0.05, bottom=0.075, right=0.90)

        # Format the common legend

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
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)


    def make_gsd_plots(self, x_name='exp_time', y_name='D50'):
        self.generic_make("2m gsd",
                self.omnimanager.reload_gsd_data,
                self.plot_gsd,
                plot_fu_kwargs={'x_name':x_name, 'y_name':y_name},
                fig_subdir='gsd')

    def plot_gsd(self, y_name='D50', x_name='exp_time'):
        sizes = ['0.5', '0.71', '1', '1.4', '2', '2.8', '4', '5.6', '8',
                '11.3', '16', '22.6', '32',]
        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'scatter',
                'legend' : False,
                #'stations' : [4500, 5168, 5836],
                'stations' : [4500, 5000, 5168, 5836, 6000],
                }
        gsd_gather_kwargs = {
                #'columns'   : y_name if isinstance(y_name, list) else [y_name],
                'columns'   : [y_name],# + sizes,
                'new_index' : ['exp_time', 'sta_str'],
                }

        # Get the name of the lines (exp_time or sta_str) automatically
        # ie pick the other item in a two item list
        line_options = gsd_gather_kwargs['new_index'].copy()
        line_options.remove(x_name)
        plot_kwargs['lines_category'] = line_options[0]

        gsd_data = self.gather_gsd_data(gsd_gather_kwargs)

        # Comparison of D50 calculation methods
        #row = gsd_data['1A'].iloc[0:5, :]
        #sizes = [0.5, 0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8, 11.2, 16, 22.3, 32]
        #curr_D50 = row.loc[:, 'D50']
        #print(curr_D50)
        #rsizes = row.loc[:, sizes]
        #new_D50 = self.calc_Di(rsizes)
        #print()
        #print(new_D50)
        #print()
        #print(curr_D50 / new_D50)
        #assert(False)

        # Generate a figure name and save the figure
        filename_y = y_name.replace('_', '-').lower()
        filename_x = x_name.replace('_', '-').lower()
        figure_name = f"2m_gsd_{filename_y}_v_{filename_x}.png"

        self.plot_labels = []
        self.generic_plot_experiments(
            self._plot_gsd, self._format_gsd_figure, 
            gsd_data, plot_kwargs, figure_name)

    def _plot_gsd(self, exp_code, all_data, plot_kwargs):
        y_name = plot_kwargs['y']
        lines_category = plot_kwargs.pop('lines_category')
        stations = plot_kwargs.pop('stations')
        sta_str = [f"sta-{station}" for station in stations]

        # Get the data and group it by the line category
        exp_data = all_data[exp_code]
        idx = pd.IndexSlice
        data = exp_data.loc[idx[:, sta_str], :]
        if data.empty:
            #print(f"EMPTY {exp_code}")
            pass
        else:
            log2_Di_name = f"{y_name}_log2"
            data.loc[:, log2_Di_name] = (np.log2(data.loc[:, y_name]))
            
            # Group the data
            sta_grouped = data.groupby(level=lines_category)
            time_grouped = data.groupby(level=plot_kwargs['x'])

            # Make a color generating function
            get_color = self.generate_rb_color_fu(len(stations))

            # Plot each sta group 
            for i, group in enumerate(sta_grouped):
                plot_kwargs['color'] = 'silver'#get_color(i)
                self.plot_group(group, plot_kwargs)

            # Plot the time group geometric mean
            time_mean = time_grouped.mean()
            Di_gm_name = f"{y_name}_gm"
            Di_gm = np.power(2, time_mean.loc[:, log2_Di_name])
            time_mean.loc[:, Di_gm_name] = Di_gm

            mean_kwargs = plot_kwargs.copy()
            del mean_kwargs['color']# = 'k'
            mean_kwargs['kind'] = 'line'
            mean_kwargs['y'] = Di_gm_name

            self.plot_group(time_mean, mean_kwargs)

        ax = plot_kwargs['ax']
        if y_name == 'D84':
            self.draw_feed_Di(ax, plot_kwargs['y'])
        elif y_name == 'D50':
            yn = y_name[-3:]
            self.draw_feed_Di(ax, yn, multiple=1.0, text='Armor 1.0')
            self.draw_feed_Di(ax, yn, multiple=1.5, text='Armor 1.5')
        ax.locator_params(axis='y', steps=[1,2,5, 10], min_n_ticks=3)
        self.generic_set_grid(ax)

        plot_kwargs['lines_category'] = lines_category
        plot_kwargs['stations'] = stations

    def _format_gsd_figure(self, fig, axs, plot_kwargs):
        x = plot_kwargs['x']
        y = plot_kwargs['y']
        Di = rf"$D_{{ {y[1:]} }}$"

        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Size (mm)",
                'title'         : rf"Bed surface {Di} between Stations 4.5 m and 6.5 m",
                'legend_labels' : [r"Mean {Di}", r"Feed {Di}"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def gather_gsd_data(self, kwargs):
        kwargs['gsd_frames'] = {}
        self.omnimanager.apply_to_periods(
                self._gather_gsd_time_data, kwargs=kwargs)
        gsd_data = {}
        for exp_code, frames in kwargs['gsd_frames'].items():
            gsd_data[exp_code] = pd.concat(frames)
            if 'new_index' in kwargs:
                gsd_data[exp_code].sort_index(level=kwargs['new_index'][0],
                        inplace=True)
            else:
                gsd_data[exp_code].sort_index()

        return gsd_data

    def _gather_gsd_time_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall gsd 
        # dataframe dict
        cols = kwargs['columns'] if 'columns' in kwargs else None
        new_index = kwargs['new_index'] if 'new_index' in kwargs else None
        add_cols = kwargs['add_cols'] if 'add_cols' in kwargs else None

        exp_code = period.exp_code
        limb = period.limb
        discharge = period.discharge_int

        data = period.gsd_data
        if data is not None:

            if add_cols is not None:
                skipped = []
                for col in add_cols:
                    if col == 'limb' and 'limb' not in data.columns:
                        data['limb'] = period.limb
                    elif col == 'discharge' and 'discharge' not in data.columns:
                        data['discharge'] = period.discharge_int
                    elif col == 'period_range' and 'period_range' not in data.columns:
                        data['period_range'] = period.period_range
                    elif col == 'exp_time_hrs' and 'exp_time_hrs' not in data.columns:
                        if data.index.name == 'exp_time':
                            data['exp_time_hrs'] = data.index.values / 60
                        else:
                            assert('exp_time' in data.columns)
                            data['exp_time_hrs'] = data['exp_time'] / 60
                    else:
                        skipped.append(col)
                if len(skipped) > 0:
                    self.logger.write(f"Skipping columns {skipped} for gathering gsd data")

            if new_index is not None:
                data.reset_index(inplace=True)
                if 'limb' in new_index:
                    data['limb'] = limb
                if 'discharge' in new_index:
                    data['discharge'] = discharge
                data.set_index(new_index, inplace=True)

            if exp_code not in kwargs['gsd_frames']:
                kwargs['gsd_frames'][exp_code] = []

            if cols is None:
                kwargs['gsd_frames'][exp_code].append(data.loc[:, :])
            else:
                kwargs['gsd_frames'][exp_code].append(data.loc[:, cols])


    # Functions to plot only Qs data
    def make_manual_Di_plots(self):
        self.logger.write([f"Making manual Di time plots..."])

        self.figure_subdir = self.figure_subdir_dict['lighttable']
        self.ignore_steps = []#['rising-50L']

        indent_function = self.logger.run_indented_function

        reload_functions = [self.load_Qs_data,
                self.omnimanager.reload_sieve_data]
        indent_function(lambda : [x() for x in reload_functions],
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_manual_Di_data,
                before_msg=f"Plotting Qs data",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_manual_Di_data(self):
        # Do stuff before plot loop
        x_name = 'exp_time_hrs'
        #y_name = 'D84'
        y_name = 'D50'
        roll_window = 10 #minutes

        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'scatter',
                #'legend' : True,
                'legend' : False,
                'xlim' : (-0.25, 8.25),
                #'ylim' : (0, settings.lighttable_bedload_cutoff), # for use without logy, for 'Bedload all'
                #'logy' : True,
                #'ylim' : (0.001, settings.lighttable_bedload_cutoff), # for use with logy, for 'Bedload all'
                }

        rolling_kwargs = {
                'x'           : x_name,
                'y'           : y_name,
                'window'      : roll_window*60, # seconds
                'min_periods' : 20,
                'center'      : True,
                #'on'          : plot_kwargs['x'],
                }
        gather_kwargs = {
                }

        qs_data = self.gather_Qs_data(gather_kwargs)
        all_sieve_data = self.gather_sieve_data({})

        figure_name = f"manual_{y_name}_check.png"

        fig, axs = self.create_experiment_subplots(rows=2, cols=4)
        
        # Make one plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]

            accumulated_data = qs_data[exp_code]
            sieve_data = all_sieve_data[exp_code]

            bedload_masses = accumulated_data.loc[:, 'Bedload 0.5':'Bedload 45'].copy()
            bedload_masses.columns = sieve_data.columns
            Di = self.calc_Di(bedload_masses, target_Di=y_name)
            labview_Di = accumulated_data.loc[:, y_name]

            ax.scatter(labview_Di.values, Di.values)
            ax.plot([0,30], [0,30], c='r')

        plt.show()


    def make_Qs_plots(self, y_name='Bedload all'):
        self.logger.write([f"Making Qs time plots..."])

        self.figure_subdir = self.figure_subdir_dict['lighttable']
        self.ignore_steps = []#['rising-50L']

        indent_function = self.logger.run_indented_function

        reload_functions = [self.load_Qs_data,
                self.omnimanager.reload_sieve_data]
        indent_function(lambda : [x() for x in reload_functions],
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_Qs_data,
                kwargs = {'y_name' : y_name},
                before_msg=f"Plotting Qs data",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_Qs_data(self, y_name='Bedload all'):
        # Do stuff before plot loop
        x_name = 'exp_time_hrs'
        #y_name = 'Bedload all'
        #y_name = 'D50'
        #y_name = 'D84'
        roll_window = 10 #minutes

        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'scatter',
                #'legend' : True,
                #'legend' : False,
                'xlim' : (-0.25, 8.25),
                #'ylim' : (0, settings.lighttable_bedload_cutoff), # for use without logy, for 'Bedload all'
                #'logy' : True,
                #'ylim' : (0.001, settings.lighttable_bedload_cutoff), # for use with logy, for 'Bedload all'
                }

        rolling_kwargs = {
                'x'           : x_name,
                'y'           : y_name,
                'window'      : roll_window*60, # seconds
                'min_periods' : 20,
                'center'      : True,
                #'on'          : plot_kwargs['x'],
                }
        gather_kwargs = {
                }

        qs_data = self.gather_Qs_data(gather_kwargs)
        if y_name in ['D50', 'D84']:
            # Gather the sieve data Di for plotting
            all_sieve_data = self.gather_sieve_data({})

        filename_y_col = y_name.replace(' ', '-').lower()
        logy_str = '_logy' if 'logy' in plot_kwargs and plot_kwargs['logy'] else ''
        figure_name = f"scaled_{filename_y_col}_roll-{roll_window}min{logy_str}.png"

        fig, axs = self.create_experiment_subplots(rows=2, cols=4)
        
        # Make one plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        ax_2B = None
        lines_plotted = {}
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            if exp_code == '2B':
                ax_2B = ax
                if self.skip_2B:
                    self.logger.write(f">>Skipping<< experiment {exp_code}")
                    ax.set_axis_off()
                    continue

            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]

            accumulated_data = qs_data[exp_code]

            # Plot the data points
            points_label = 'Bedload rates' if y_name == 'Bedload all' else f"Bedload {y_name}"
            accumulated_data.plot(**plot_kwargs, marker='.', c='silver',
                    label=points_label)

            # Generate and plot rolled data
            rolled_data = self.roll_data(accumulated_data,
                    roll_kwargs=rolling_kwargs)

            rolling_median = rolled_data.median()
            rolling_75p = rolled_data.quantile(quantile=0.75)
            rolling_25p = rolled_data.quantile(quantile=0.25)
            rolling_mean = rolled_data.mean()
            rolling_stddev = rolled_data.std()

            #def plot_stddev(mean, stddev, label, series_plot_kwargs):
            #    #range = pd.concat([mean - stddev, mean +stddev])
            #    range = mean + stddev
            #    range.plot(label=label, **series_plot_kwargs)
                
            # Plot the rolls
            series_plot_kwargs = {k : plot_kwargs[k] for k in plot_kwargs
                                        if k not in ['x', 'y', 'kind']}
            #series_plot_kwargs['xlim'] = ax.get_xlim()
            #series_plot_kwargs['style'] = 'r'

            rolling_75p.plot(label=r"75^{th} Percentile", 
                    style='c', **series_plot_kwargs)
            rolling_25p.plot(label=r"25^{th} Percentile",
                    style='c', **series_plot_kwargs)
            rolling_median.plot(label='Median',
                    style='b', **series_plot_kwargs)

            #rolling_mean.plot(label='Mean', style='b', **series_plot_kwargs)
            #(rolling_mean + rolling_stddev).plot(label="Mean + Stddev",
            #        style='c', **series_plot_kwargs)
            #plot_stddev(rolling_mean, rolling_stddev, "Mean + Stddev", series_plot_kwargs)
            #plot_stddev(rolling_mean, rolling_stddev*5, "5x Stddev envelope", series_plot_kwargs)

            if y_name in ['D50', 'D84']:
                self.draw_feed_Di(ax, y_name)

                # Plot the sieve data Di if applicable
                sieve_data = all_sieve_data[exp_code]
                sieve_data[y_name] = self.calc_Di(sieve_data,
                        target_Di=int(y_name[1:]))
                sieve_plot_kwargs = {
                        'x'      : 'exp_time' if x_name == 'exp_time_hrs' else x_name,
                        'y'      : y_name,
                        'label'  : rf"Trap {y_name}",
                        'ax'     : ax,
                        'legend' : False,
                        'marker' : '*',
                        'markersize': 7,
                        'linestyle' : 'None',
                        'color'  : 'r',
                        }
                self.plot_group(sieve_data, sieve_plot_kwargs)

            if not self.for_paper:
                self.plot_2B_X(exp_code, ax)

            ax.set_title(f"{experiment.code} {experiment.name}")
            ax.set_ylabel('')
            ax.set_xlabel('')
            self.generic_set_grid(ax, yticks_minor=True)

            ax.get_legend().remove()

        if y_name == 'Bedload all':
            fig_kwargs = {
                    'xlabel' : r"Experiment time (hours)",
                    'ylabel' : r"Bedload (g/s)",
                    'title'  : r"Light table bedload data with standard deviation envelope",
                    #'legend_labels' : [r"Mean", r"Mean + Std Dev"],
                    }
            #axs[0,0].set_ylim((-50, 400))

        elif y_name in ['D50', 'D84']:
            fig_kwargs = {
                    'xlabel' : r"Experiment time (hours)",
                    'ylabel' : rf"$D_{{ {y_name[1:]} }}$ (mm)",
                    'title'  : rf"Bedload {y_name} with standard deviation envelope",
                    #'legend_labels' : [r"Mean", r"Mean + Std Dev"],
                    }
            ylim = (-3, 40)if y_name == 'D50' else (-5, 70)
            axs[0,0].set_ylim(ylim)
        if self.for_paper:
            self.add_common_label(axs[0,0], ax_2B)
            
        if not self.for_paper:
            plt.legend()
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)

        # Save the figure
        self.save_figure(figure_name)
        plt.show()

    def add_common_label(self, source_ax, display_ax, has_quartiles=True):
        if display_ax is None:
            return

        # Get the labels and line handles from one plot
        handles, labels = source_ax.get_legend_handles_labels()

        if has_quartiles:
            # Remove one of the quartiles and rename the other
            r75_idx = labels.index(r"75^{th} Percentile")
            handles.pop(r75_idx)
            labels.pop(r75_idx)
            r25_idx = labels.index(r"25^{th} Percentile")
            labels[r25_idx] = rf"$25^{{th}}$ and $75^{{th}}$ Percentiles"

        legend = display_ax.legend(handles, labels)

        return legend

    def make_Di_ratio_plots(self):
        self.generic_make(f"Qs Di ratio",
                self.omnimanager.reload_Qs_data, self.plot_Di_ratio,
                fig_subdir='lighttable')

    def plot_Di_ratio(self):
        # Do stuff before plot loop
        x_name = 'exp_time_hrs'
        #y_name = 'Bedload all'
        Dnumerator = 'D84'
        Dnominator = 'D50'
        y_name = f'{Dnumerator}_{Dnominator}'
        roll_window = 10 #minutes

        plot_kwargs = {
                'x'      : x_name,
                'y'      : y_name,
                'kind'   : 'scatter',
                #'legend' : True,
                'legend' : False,
                'xlim' : (-0.25, 8.25),
                #'ylim' : (0, settings.lighttable_bedload_cutoff), # for use without logy, for 'Bedload all'
                #'logy' : True,
                #'ylim' : (0.001, settings.lighttable_bedload_cutoff), # for use with logy, for 'Bedload all'
                }

        rolling_kwargs = {
                'x'           : x_name,
                'window'      : roll_window*60, # seconds
                'min_periods' : 20,
                'center'      : True,
                #'on'          : plot_kwargs['x'],
                }
        gather_kwargs = {
                }

        qs_data = self.gather_Qs_data(gather_kwargs)

        filename_y_col = y_name.replace(' ', '-').lower()
        logy_str = '_logy' if 'logy' in plot_kwargs and plot_kwargs['logy'] else ''
        figure_name = f"scaled_{filename_y_col}_roll-{roll_window}min{logy_str}.png"

        fig, axs = self.create_experiment_subplots(rows=4, cols=2)
        
        # Make one plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]

            accumulated_data = qs_data[exp_code]

            ## Plot the data points
            #accumulated_data.plot(**plot_kwargs, marker='.', c='silver')

            numerator = accumulated_data.loc[:, Dnumerator]
            denominator = accumulated_data.loc[:, Dnominator]
            ratio = numerator / denominator
            accumulated_data.loc[:, 'ratio'] = ratio
            rolling_kwargs['y'] = 'ratio'
            rolled_ratio = self.roll_data(
                    accumulated_data, roll_kwargs=rolling_kwargs)
            ratio_mean = rolled_ratio.mean()

            # Do ratio before window mean
            ## Generate and plot rolled data
            #rolling_kwargs['y'] = Dnumerator
            #rolled_numerator = self.roll_data(
            #        accumulated_data, roll_kwargs=rolling_kwargs)

            #rolling_kwargs['y'] = Dnominator
            #rolled_denominator = self.roll_data(
            #        accumulated_data, roll_kwargs=rolling_kwargs)

            #numerator_mean = rolled_numerator.mean()
            #denominator_mean = rolled_denominator.mean()

            #ratio = numerator_mean / denominator_mean
            # Plot the rolls
            series_plot_kwargs = {k : plot_kwargs[k] for k in plot_kwargs
                                        if k not in ['x', 'y', 'kind']}
            #series_plot_kwargs['xlim'] = ax.get_xlim()
            #series_plot_kwargs['style'] = 'r'
            #ratio.plot(**series_plot_kwargs)
            ratio_mean.plot(**series_plot_kwargs)

            ax.set_title(f"{experiment.code} {experiment.name}")
            ax.set_ylabel('')
            ax.set_xlabel('')
            self.generic_set_grid(ax, yticks_minor=True)

        fig_kwargs = {
                'xlabel' : r"Experiment time (hours)",
                'ylabel' : rf"$\frac{{{Dnumerator}}}{{{Dnominator}}}$",
                'title'  : rf"Bedload grain size ratio $\frac{{{Dnumerator}}}{{{Dnominator}}}$",
                #'legend_labels' : [r"Mean", r"Mean + Std Dev"],
                'ylabel_rotation' : 'horizontal',
                }
        #plt.legend()
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)
        
        # Save the figure
        self.save_figure(figure_name)
        plt.show()


    def make_experiment_plots(self):
        self.logger.write(["Making experiment plots..."])

        #self.experiments = {}
        self.ignore_steps = []#['rising-50L']

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
        y_column = 'Bedload all'
        #y_column = 'D50'
        roll_window = 10 #minutes

        plot_kwargs = {
                'x'    : x_column,
                'y'    : y_column,
                'kind' : 'scatter',
                #'logy' : True,
                'xlim' : (-0.25, 8.25),
                #'xlim' : (0.5, 8.5),
                #'ylim' : (0.001, 5000), # for use with logy
                #'ylim' : (0.001, 500), # for use with logy
                #'ylim' : (0, 40),
                }
        rolling_kwargs = {
                'y'           : plot_kwargs['y'],
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
        figure_name = f"cleaned_{filename_y_col}_roll-{roll_window}min{logy_str}.png"

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))
        #twin_axes = []

        # Make one plot per experiment
        exp_codes = self.omnimanager.get_exp_codes()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            #twax = ax.twinx()
            #twin_axes.append(twax)
            experiment = self.omnimanager.experiments[exp_code]
            accumulated_data = experiment.accumulated_data

            self.rolling_av = None
            #self.hydrograph = None

            # Plot the data points
            accumulated_data.plot(**plot_kwargs)

            ## Generate and plot the hydrograph
            #experiment.apply_period_function(self._generate_hydrograph, kwargs)
            #self.hydrograph.sort_index(inplace=True)
            #self.hydrograph.plot(ax=twax, style='g', ylim=(50,800))
            #twax.tick_params('y', labelright='off')
            #if exp_code in ['2B', '5A']:
            #    twax.set_ylabel('Discharge (L/s)')

            # Generate and plot rolled averages
            self.roll_data(accumulated_data, roll_kwargs=rolling_kwargs)
            series_plot_kwargs = {k : plot_kwargs[k] for k in plot_kwargs
                                        if k not in ['x', 'y', 'kind']}
            #series_plot_kwargs['xlim'] = ax.get_xlim()
            series_plot_kwargs['style'] = 'r'
            self.rolling_av.plot(**series_plot_kwargs)

            ax.set_title(f"{experiment.code} {experiment.name}")

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


    def make_bedload_feed_plots(self):
        reload_kwargs = {
                'add_feed' : True,
                }
        self.generic_make(f"bedload feed",
                self.omnimanager.reload_Qs_data, self.plot_bedload_feed,
                load_fu_kwargs=reload_kwargs,
                fig_subdir='lighttable')

    def plot_bedload_feed(self):
        # Do stuff before plot loop
        x_name = 'feed'
        y_name = 'Bedload all'

        #roll_window = 10 #minutes
        #rolling_kwargs = {
        #        'x'           : 'exp_time_hrs',
        #        'y'           : y_name,
        #        'window'      : roll_window*60, # seconds
        #        'min_periods' : 20,
        #        'center'      : True,
        #        #'on'          : plot_kwargs['x'],
        #        }
        gather_kwargs = {
                #'ignore_steps' : ['rising-50L']
                }

        qs_data = self.gather_Qs_data(gather_kwargs)

        # Can't use generic loop
        # Want to plot all of them on one chart
        fig = plt.figure(figsize=(16,10))
        ax = plt.gca()

        # Make one color per experiment
        #colors = list(plt.get_cmap('tab20').colors)
        #colors = [ # from 'Set1' w darker yellow
        #    (0.8941, 0.1019, 0.1098), (0.2156, 0.4941, 0.7215),
        #    (0.3019, 0.6862, 0.2901), (0.5960, 0.3058, 0.6392),
        #    (1.0000, 0.4980, 0.0000), (0.6000, 0.6000, 0.1176),
        #    (0.6509, 0.3372, 0.1568), (0.9686, 0.5058, 0.7490),
        #    (0.6000, 0.6000, 0.6000),
        #    ]
        colors = list(zip([
            # Bright
            [228,  26,  28],
            [ 55, 126, 184],
            [ 77, 175,  74],
            [152,  78, 163],
            [255, 127,   0],
            #[255, 255,  51],
            [231, 214,  43],
            [166,  86,  40],
            #[247, 129, 191],
            [247, 103, 178],
            ], [
            # Faded
            [255,  87,  88],
            [101, 169, 225],
            [113, 204, 110],
            [205, 125, 217],
            [255, 193, 132],
            [224, 202,  99],
            [180, 130, 100],
            [247, 170, 211],
            #[251, 180, 174],
            #[179, 205, 227],
            #[204, 235, 197],
            #[222, 203, 228],
            #[254, 217, 166],
            #[255, 255, 204],
            #[229, 216, 189],
            #[253, 218, 236],
            ]))
        #[print(c) for c in colors]
        exp_codes = self.omnimanager.get_exp_codes()
        exp_codes.sort(reverse=True)
        offsets = {c: -o for c, o in zip(exp_codes, range(len(exp_codes)))}
        assert(len(exp_codes) <= len(colors))
        for exp_code, color_pair in zip(exp_codes, colors[:len(exp_codes)]):
            self.logger.write(f"Plotting experiment {exp_code}")
            experiment = self.omnimanager.experiments[exp_code]

            # Separate data
            peak_time = 4.5 #hrs
            all_data = qs_data[exp_code]
            rising = all_data[all_data['exp_time_hrs'] <= peak_time]
            falling = all_data[all_data['exp_time_hrs'] > peak_time]
            loop_args = zip([rising, falling], ['rising', 'falling'],
                    color_pair)
            for l_data, limb, color in loop_args:
                color = np.array(color)/255
                data = l_data.loc[l_data['discharge'] != 50, (x_name, y_name)]
                groups = data.groupby(by='feed')

                # FYI DataFrameGroupBy.boxplot is sucking for some reason. Cannot 
                # figure out why it plots both feed and discharge or crashes if I 
                # try to select one. If boxplots are necessary, rather than mean & 
                # stddev, separate the data out by hand and plot boxplots one at a 
                # time. Don't waste more time on this!!
                
                # Get stats
                means = groups.mean()
                means.reset_index(inplace=True)
                stds = groups.std()
                stds.reset_index(inplace=True)

                # Make the mean point
                ax.plot(stds['feed'], means['Bedload all'],
                        c=color, linestyle='None',
                        marker=('*' if limb == 'rising' else 'x'), ms=8,
                        label=f"{exp_code} {limb}",
                        )
                # Make the error bars
                limb_offset = 0.0 if limb == 'rising' else 0.5
                ax.errorbar(stds['feed'] + offsets[exp_code] + limb_offset,
                        means['Bedload all'], stds['Bedload all'],
                        c=color, linestyle='None',
                        elinewidth=1, capsize=5,
                        label=None,
                        marker='_',
                        )

        plt.legend()
        ax.set_xlabel(r"Feed rate (kg/hr)")
        ax.set_ylabel(r"Bedload transport (g/s)")
        ax.set_title(r"Mean bedload transport rates and standard deviations compared to sediment supply")

        # Generate a figure name and save the figure
        x_name = 'feed'
        y_name = 'Bedload all'
        filename_y_col = y_name.replace(' ', '-').lower()
        figure_name = f"bedload_feed_limbs.png"
        #figure_name = f"bedload_feed_roll-{roll_window}min.png"

        self.save_figure(figure_name)
        plt.show()


    def make_hysteresis_plots(self):
        self.logger.write(["Making hysteresis plots..."])

        self.ignore_steps = ['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_hysteresis,
                before_msg="Plotting hysteresis trends",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_hysteresis(self):
        t_column = 'exp_time_hrs'
        #x_column = 'discharge'
        x_column = 'pseudo_time'
        #y_column = 'Bedload all'
        y_column = 'rolled_bedload'

        roll_window = 5 #minutes
        figure_name = f"hysteresis_{roll_window}min-mean_logy.png"
        plot_kwargs = {
                'x'        : x_column,
                'y'        : y_column,
                't_column' : t_column,
                'kind'     : 'scatter',
                'logy'     : True,
                #'xlim'     : (50, 125),
                'xlim'     : (1, 8),
                #'ylim'     : (0.001, 5000),
                #'ylim'     : (0, 200),
                'legend'   : False,
                }
        roll_kwargs = {
                'window'      : roll_window*60, # seconds
                'min_periods' : 40,
                'y'           : 'Bedload all',
                #'on'          : plot_kwargs['x'],
                }
        plot_kwargs['roll_kwargs'] = roll_kwargs

        Qs_data = self.gather_Qs_data()

        self.generic_plot_experiments(
            self._plot_hysteresis, self._format_hysteresis, 
            Qs_data, plot_kwargs, figure_name, subplot_shape=(2,4))

    def _plot_hysteresis(self, exp_code, Qs_data, plot_kwargs):
        x_column = plot_kwargs['x']
        y_column = plot_kwargs['y']
        t_column = plot_kwargs.pop('t_column')

        #scale = 1/1000 # covert to kg
        accumulated_data = Qs_data[exp_code]

        roll_kwargs = plot_kwargs.pop('roll_kwargs')

        self.rolling_av = None

        # Generate rolled averages
        self.roll_data(accumulated_data, roll_kwargs.copy(), plot_kwargs)
        #max_val = max(max_val, self.rolling_av.max())

        # Select the hysteresis data
        data = pd.DataFrame(accumulated_data.loc[:,['Bedload all', t_column]])

        # Make pseudo-time
        # reverse at t = 4.5 hours
        time_hrs = data[t_column].copy()
        times_after = time_hrs.index > 4.5
        time_hrs.loc[times_after] = 9 - time_hrs.loc[times_after]
        data['pseudo_time'] = time_hrs

        data.set_index(t_column, inplace=True)
        data[y_column] = self.rolling_av

        # Subsample data
        subsample_step = roll_kwargs['window']
        data = data.iloc[::subsample_step, :]
        
        if plot_kwargs['kind'] == 'scatter':
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
        self._generic_df_plot(data, plot_kwargs)
        plot_kwargs['ax'].grid(True, color='#d8dcd6')

        plot_kwargs['t_column'] = t_column
        plot_kwargs['roll_kwargs'] = roll_kwargs

    def _format_hysteresis(self, fig, axs, plot_kwargs):
        #max_y = int(max_val*1.1)
        for ax in axs.flatten():
            #ax.set_ylim((0, max_y))
            # There are some outliers that should be removed. Ignore them for 
            # now.
            ax.set_ylim((0, 200))

        fig_kwargs = {
                'xlabel'        : r"Discharge",
                'ylabel'        : r"Bedload transport",
                'title'         : r"Hysteresis of bedload transport",
                'legend_labels' : ["Bedload"],
                }
        #plt.suptitle(f"Hysteresis trends between total bedload output "+
        #        f"and discharge ({roll_window} min roll window, {asctime()})")
        
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)

    def ___plot_hysteresis(self):
        pass # Old code
        #t_column = 'exp_time_hrs'
        #x_column = 'discharge'
        #y_column = 'Bedload all'
        #columns_to_plot = [t_column, x_column, y_column]

        #fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))

        #roll_window = 5 #minutes
        #figure_name = f"hysteresis_{roll_window}min-mean_logy.png"
        #plot_kwargs = {
        #        'x'      : x_column,
        #        'y'      : y_column,
        #        'kind'   : 'scatter',
        #        'logy'   : True,
        #        'xlim'   : (50, 125),
        #        #'ylim'   : (0.001, 5000),
        #        #'ylim'   : (0, 200),
        #        }
        #rolling_kwargs = {
        #        'window'      : roll_window*60, # seconds
        #        'min_periods' : 40,
        #        #'on'          : plot_kwargs['x'],
        #        }
        #kwargs = {'plot_kwargs'       : plot_kwargs,
        #          'rolling_kwargs'    : rolling_kwargs,
        #          }

        ## Make one plot per experiment
        #exp_codes = list(qs_data.keys())
        #max_val = 0
        #for exp_code, ax in zip(exp_codes, axs.flatten()):
        #    self.logger.write(f"Plotting experiment {exp_code}")

        #    plot_kwargs['ax'] = ax
        #    experiment = self.omnimanager.experiments[exp_code]
        #    accumulated_data = experiment.accumulated_data

        #    self.rolling_av = None

        #    # Generate rolled averages
        #    self.roll_data(accumulated_data, kwargs)
        #    max_val = max(max_val, self.rolling_av.max())

        #    # Select the hysteresis data
        #    data = pd.DataFrame(accumulated_data.loc[:,[x_column, t_column]])
        #    data.set_index(t_column, inplace=True)
        #    data[y_column] = self.rolling_av

        #    # Subsample data
        #    subsample_step = rolling_kwargs['window']
        #    data = data.iloc[::subsample_step, :]
        #    
        #    # Generate colors and markers
        #    n_rows = data.index.size
        #    half_n = n_rows//2
        #    colors = np.ones((n_rows, 3))
        #    s = np.linspace(0,1,num=half_n)
        #    colors[-half_n : , 0] = 1 - s # r
        #    colors[     :    , 1] = 0     # g
        #    colors[ : half_n , 2] = s     # b
        #    plot_kwargs['marker'] = 'o'
        #    plot_kwargs['c'] = 'none' # set default color to transparent
        #    plot_kwargs['facecolors'] = 'none' # set fill to transparent
        #    plot_kwargs['edgecolors'] = colors

        #    # Plot it!
        #    data.plot(**plot_kwargs)

        #    ax.set_title(f"Experiment {experiment.code} {experiment.name}")

        #max_y = int(max_val*1.1)
        #for ax in axs.flatten():
        #    #ax.set_ylim((0, max_y))
        #    # There are some outliers that should be removed. Ignore them for 
        #    # now.
        #    ax.set_ylim((0, 200))

        #plt.suptitle(f"Hysteresis trends between total bedload output "+
        #        f"and discharge ({roll_window} min roll window, {asctime()})")

        #filepath = ospath_join(self.figure_destination, figure_name)
        #self.logger.write(f"Saving figure to {filepath}")
        #plt.savefig(filepath, orientation='landscape')
        #plt.show()


    def make_cumulative_mass_bal_plots(self):
        self.logger.write(["Making cumulative plots..."])

        self.figure_subdir = self.figure_subdir_dict['lighttable']
        self.ignore_steps = []#['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_cumulative_mass_bal,
                before_msg="Plotting cumulative transport",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_cumulative_mass_bal(self):
        #sns.set_style('ticks')
        x_column = 'exp_time_hrs'
        cumsum_column = 'cumulative'
        massbal_column = 'mass_balance'
        feed_column = 'feed'

        figure_name = f"cleaned_cumulative_mass_balance.png"

        plot_kwargs = {
                'x'      : x_column,
                'y'      : [feed_column, cumsum_column, massbal_column],
                'label'  : ['Cumulative Feed', 'Cumulative Sediment Yield', 'Mass Balance'],
                #'y'      : [cumsum_column],
                'kind'   : 'line',
                'xlim'   : (0, 8),
                #'ylim'   : (0.001, 5000),
                #'ylim'   : (0, 200),
                'legend' : False,
                'legend_exp_code' : '1A',
                'color'  : ['k', 'r', 'g'],
                }
        kwargs = {'plot_kwargs'       : plot_kwargs,
                  }

        Qs_data = self.gather_Qs_data()

        self.generic_plot_experiments(
            self._plot_cumulative_mass_bal, self._format_cumulative_mass_bal, 
            Qs_data, plot_kwargs, figure_name, subplot_shape=(2,4))

    def _plot_cumulative_mass_bal(self, exp_code, Qs_data, plot_kwargs):
        bedload_col = 'Bedload all'
        time_col = plot_kwargs['x']
        plot_cols = plot_kwargs['y']
        cumsum_col = 'cumulative' # hacky

        scale = 1/1000 # covert to kg
        accumulated_data = Qs_data[exp_code]
        accumulated_data[bedload_col] = accumulated_data[bedload_col] * scale

        # Select the target data
        data = pd.DataFrame(accumulated_data.loc[:,[time_col, bedload_col]])
        data.set_index(time_col, inplace=True)
        data.loc[0.0, bedload_col] = 0 # Set first point to zero

        # Generate cumulative
        cumsum = data[bedload_col].cumsum(axis=0)
        data[cumsum_col] = cumsum

        # Subsample data
        subsample_step = 60*1
        data = data[data.notnull().all(axis=1)] # Get rid of nan rows
        data = data.iloc[::subsample_step, :] # reduce num data points

        # Make feed series
        feed_rate = settings.feed_rates[exp_code]
        #n_res = 1
        #feed_matrix = np.tile(feed_rate, (n_res, 1)) / n_res
        #feed_series = feed_matrix.T.flatten()
        #feed_cumsum = np.concatenate(([0], feed_series.cumsum()))
        #feed_times = np.linspace(0, 8, num=len(feed_cumsum))

        def calc_cumsum_feed(time):
            # Calculate the feed cumsum up to time
            # Assumes time is in hours
            if time >= len(feed_rate):
                return feed_rate.sum()
            lower_index = np.floor(time).astype(np.int)
            lower_cumsum = feed_rate[:lower_index].sum()
            hour_rate = feed_rate[lower_index]
            partial_index = time%1
            partial_cumsum = hour_rate * partial_index
            cumsum = lower_cumsum + partial_cumsum
            #print(time, lower_index, partial_index, cumsum)
            return cumsum

        #get_feed = np.vectorize(calc_cumsum_feed)
        time_indices = data.index.to_series()
        feed_cumsum = time_indices.apply(calc_cumsum_feed)
        feed_cumsum.index = time_indices
        data['feed'] = feed_cumsum
        mass_balance = feed_cumsum - data[cumsum_col]
        data['mass_balance'] = mass_balance
        #print(feed_cumsum.astype(np.int))
        #print(feed_cumsum.astype(np.int)[8])
        #print([int(calc_cumsum_feed(t)) for t in [0,1.5,2,2.5,3,8]])
        min_mb = mass_balance.min()
        print(f"Min mass balance for exp {exp_code} = {min_mb}")

        # Plot it!
        self._time_plot_prep(data, plot_kwargs)

        self.generic_set_grid(plot_kwargs['ax'], yticks_minor=True)

    def _format_cumulative_mass_bal(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Mass (kg)",
                'title'         : r"Cumulative sum of bedload output and flume mass balance",
                #'legend_labels' : [r"Shear from surface", r"Shear from bed"],
                'legend_labels' : ["Cumulative\nbedload", "Flume\nmass balance"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)


    def make_cumulative_plots(self):
        self.logger.write(["Making cumulative plots..."])

        #self.experiments = {}
        #self.omnimanager.wipe_data()
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
        exp_codes = list(qs_data.keys())
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

            ax.set_title(f"{experiment.code} {experiment.name}")

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
        exp_codes = list(qs_data.keys())
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

            ax.set_title(f"{experiment.code} {experiment.name}")

        plt.suptitle(f"Histogram of bedload transport rates ({asctime()})")

        #filepath = ospath_join(self.figure_destination, figure_name)
        #self.logger.write(f"Saving figure to {filepath}")
        #plt.savefig(filepath, orientation='landscape')
        plt.show()


    def make_maria_2019_plot(self):
        self.logger.write(["Making Maria 2019 plot..."])

        self.figure_subdir = self.figure_subdir_dict['mass_balance']
        self.ignore_steps = []#['rising-50L']

        indent_function = self.logger.run_indented_function

        indent_function(self.load_Qs_data,
                before_msg="Loading light table data", 
                after_msg="Light table data Loaded!")
        indent_function(self.omnimanager.reload_feed_data,
                before_msg="Loading feed data", after_msg="Feed data Loaded!")

        indent_function(self.plot_maria_2019,
                before_msg="Plotting mass balance by size class",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_maria_2019(self):
        #sns.set_style('ticks')
        x_column = 'exp_time_hrs'

        figure_name = f"cleaned_cumulative_mass_balance.png"

        plot_kwargs = {
                'x'      : x_column,
                'y'      : None, # defined at end of _plot_maria_2019
                'label'  : None, # defined at end of _plot_maria_2019
                #'y'      : [cumsum_column],
                'kind'   : 'line',
                'xlim'   : (0, 8),
                'ylim'   : (-400, 200),
                'legend' : False,
                'legend_exp_code' : '1A',
                #'color'  : ['k', 'r', 'g'],
                }
        kwargs = {'plot_kwargs'       : plot_kwargs,
                  }

        Qs_data = self.gather_Qs_data()
        feed_data = self.get_feed_GSDs().sum(axis=1)
        feed_frac_data = feed_data / feed_data.sum()

        data_dict = {'Qs_data' : Qs_data,
                'feed_frac_data' : feed_frac_data,
                }

        self.generic_plot_experiments(
            self._plot_maria_2019, self._format_maria_2019, 
            data_dict, plot_kwargs, figure_name, subplot_shape=(2,4))

    def _plot_maria_2019(self, exp_code, data_dict, plot_kwargs):
        #if exp_code != '2A':
        #    print(f'Skipping {exp_code} for debugging')
        #    return
        feed_col = 'feed'
        raw_data = data_dict['Qs_data'][exp_code]
        feed_frac_data = data_dict['feed_frac_data']

        # Drop uneccesary columns from Qs data
        bedload_cols = [name for name in raw_data.columns if 'Bedload' in name]
        time_cols = [name for name in raw_data.columns if name in 
                ['exp_time_hrs', 'exp_time']]
        drop_cols = [name for name in raw_data.columns if name not in 
                bedload_cols + time_cols]
        raw_data.drop(columns=drop_cols, inplace=True)
        output_data = raw_data.loc[:, time_cols]

        # Make feed rate column (g/sec)
        hourly_feed_rate = settings.feed_rates[exp_code]
        def calc_feed_rate(time_hrs):
            # Calculate the g per sec feed rate
            if time_hrs >= len(hourly_feed_rate):
                return 0
            lower_index = np.floor(time_hrs).astype(np.int)
            rate = hourly_feed_rate[lower_index] / 3.600
            return rate
        raw_data[feed_col] = raw_data['exp_time_hrs'].apply(calc_feed_rate)

        # Calculate cumsum and convert to kg
        rate_cols = bedload_cols + [feed_col]
        cumsum_data = raw_data[rate_cols].cumsum() / 1000

        # Do consolidation and calculations by GS categories 
        gs_categories = {
                new_col_name: GSCategory(new_col_name, gs_sizes) \
                        for (new_col_name, gs_sizes) in [
                            ('total'   , []),
                            ('lt_1mm'  , [ '0.5' ,  '0.71',  '1'   , ]),
                            ('1_2mm'   , [ '1.4' ,  '2'   , ]),
                            #('lt_2mm'  , [ '0.5' ,  '0.71',  '1'   ,
                            #               '1.4' ,  '2'   , ]),
                            ('2_4mm'   , [ '2.8' ,  '4'   , ]),
                            ('4_8mm'   , [ '5.6' ,  '8'   , ]),
                            #('2_8mm'   , [ '2.8' ,  '4'   ,
                            #               '5.6' ,  '8'   , ]),
                            ('8_16mm'  , ['11.2' , '16'   , ]),
                            ('geq_16mm', ['22'   , '32'   , '45'   ]),
                            ]
                        }
        gs_plot_order = ['total', 'lt_1mm', '1_2mm', '2_4mm', '4_8mm',
                '8_16mm', 'geq_16mm']
        #gs_plot_order = ['total', 'lt_2mm', '2_8mm', '8_16mm', 'geq_16mm']
        gs_legend_names = []
        for new_col in gs_plot_order:
        #for (new_col, gs_category) in gs_categories.items():
            gs_category = gs_categories[new_col]
            #gs_plot_order.append(new_col)
            legend_name = gs_category.legend_name
            gs_legend_names.append(legend_name)
            cat_bedload_cols = gs_category.bedload_cols
            cat_sieve_cols = gs_category.sieve_cols

            # Calculate bedload mass for this GS category
            cat_bedload_mass = cumsum_data.loc[:, cat_bedload_cols].sum(axis=1)

            # Must manually set nan rows to zero.
            # For some reason sum is not doing it correctly
            nan_row = cumsum_data.isna().any(axis=1)
            cat_bedload_mass[nan_row] = np.nan

            # Calculate fraction of feed in this GS class
            cat_feed_frac = feed_frac_data[cat_sieve_cols].sum()
            cat_feed_mass = cumsum_data.loc[:, feed_col] * cat_feed_frac

            # Calculate mass balance for this GS category
            cat_mass_balance = cat_feed_mass - cat_bedload_mass

            # Add back into the cumsum dataframe
            output_data[new_col] = cat_mass_balance

        # Set index to exp_time
        output_data.set_index('exp_time', drop=True, inplace=True)

        ## Subsample data
        #subsample_step = 60*5 # 5 min
        #subsample_index = (output_data.index.values % subsample_step) == 0
        #output_data = output_data.loc[subsample_index, :]

        # Set index to exp_time_hrs
        #output_data.set_index('exp_time_hrs', drop=True, inplace=True)

        # Plot it!
        plot_kwargs['y'] = gs_plot_order
        plot_kwargs['label'] = gs_legend_names
        #plot_kwargs['legend'] = True
        self._generic_df_plot(output_data, plot_kwargs)

        self.generic_set_grid(plot_kwargs['ax'], yticks_minor=True)

    def _format_maria_2019(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Mass (kg)",
                'title'         : r"Cumulative sum of bedload output and flume mass balance",
                #'legend_labels' : [r"Shear from surface", r"Shear from bed"],
                'legend_labels' : ["Cumulative\nbedload", "Flume\nmass balance"],
                }
        self.format_generic_figure(fig, axs, plot_kwargs, fig_kwargs)

    def get_feed_GSDs(self):
        feed_data_dict = {}
        for exp_code in self.omnimanager.experiments.keys():
            experiment = self.omnimanager.experiments[exp_code]

            data = experiment.feed_data
            if data is None:
                continue
            for sample in data.index:
                feed_data_dict[(exp_code, sample)] = data.loc[sample, :].copy()
        feed_data = pd.DataFrame.from_dict(feed_data_dict)

        return feed_data


    # General functions to be given to PeriodData instances (out of date tite?)

    def load_Qs_data(self, accumulate_kwargs=None):
        #self.experiments = self.loader.load_pickle(self.exp_omnipickle)
        if accumulate_kwargs is None:
            accumulate_kwargs = {
                    'check_ignored_fu' : self._check_ignore_period,
                    }
        self.omnimanager.reload_Qs_data(kwargs=accumulate_kwargs)

    def gather_Qs_data(self, gather_kwargs={}, return_outliers=False):
        Qs_data = {}
        for exp_code in self.omnimanager.experiments.keys():
            experiment = self.omnimanager.experiments[exp_code]
            Qs_data[exp_code] = experiment.accumulated_data

        Qs_data, outlier_meta_dict = self.clean_Qs_data(Qs_data)
        if return_outliers:
            return Qs_data, outlier_meta_dict
        else:
            return Qs_data

    def clean_Qs_data(self, Qs_data):
        self.logger.write_blankline()
        p_window = 5 # Note, this includes the center
        p_tolerance = 250 # g/s
        self.logger.write(f"Cleaning Qs data")
        self.logger.increase_global_indent()
        self.logger.write(f"Note: Outliers are currently calculated as " +\
                f"any point +-{p_tolerance} g/s from the average " +\
                f"of the neighboring {p_window-1} nodes (prominence)")

        outlier_meta = {}
        outlier_meta['prominence_pd_dict'] = {}
        outlier_meta['outliers_pd_dict'] = {}

        for exp_code in Qs_data.keys():
            self.logger.write(f"Cleaning {exp_code}")
            raw_pd_data = Qs_data[exp_code]
            bedload_all_pd = raw_pd_data['Bedload all']
            exp_time_hrs = raw_pd_data['exp_time_hrs'].values
            
            # Find prominence
            # Measure of how different the target node is from the average of 
            # the adjacent cells. I can't find a rolling window type that 
            # excludes the center node (kinda like image processing kernels 
            # can) so I can average only the adjacent nodes, hence the more 
            # complicated prominence equation.
            p_roller = bedload_all_pd.rolling(
                    p_window, min_periods=1, center=True)
            p_sum = p_roller.sum()
            prominence = \
                    (1 + 1/(p_window-1)) * bedload_all_pd - \
                    p_sum / (p_window - 1)
            prominence.index = exp_time_hrs

            # Identify outliers
            very_pos = prominence > p_tolerance
            very_neg = prominence < -p_tolerance
            is_outlier = (very_pos | very_neg).values
            outlier_idx = np.where(is_outlier)[0]

            outliers = bedload_all_pd[is_outlier].copy()
            outliers.index = exp_time_hrs[is_outlier]

            # Save data and remove outliers
            outlier_meta['is_outlier'] = is_outlier
            outlier_meta['outlier_idx'] = outlier_idx
            outlier_meta['prominence_pd_dict'][exp_code] = prominence.copy()
            outlier_meta['outliers_pd_dict'][exp_code] = outliers

            bedload_columns = [
                    'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                    'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                    'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                    'Bedload 22', 'Bedload 32', 'Bedload 45',
                    ]
            col_idx = np.where(np.in1d(raw_pd_data.columns, bedload_columns))[0]
            raw_pd_data.iloc[outlier_idx, col_idx] = np.nan

        self.logger.decrease_global_indent()
        return Qs_data, outlier_meta


    def make_yield_feed_plot(self):
        # Very hacky plotting function....

        #labels = ['1A', '1B', '2A', '3A', '3B', '4A', '5A',]
        rfeed  = [   0,    0,  400,  700,  700,  100,  400,]
        ryield = [ 521,  479,  707,  873,  830,  277,  631,]
        ffeed  = [   0,    0,  400,  100,  100,  700,  400,]
        fyield = [  93,   82,   92,  309,  226,  319,  274,]
        #plt.scatter(rfeed, ryield, label='Rising Limb')
        #plt.scatter(ffeed, fyield, label='Falling Limb')
        #plt.ylabel('Bedload Transport Yield (kg)')
        #plt.xlabel('Total Sediment Feed for Limb (kg)')
        #plt.show()

        # Calculate ols trends. Save for plotting later.
        ols_lines = {}
        for lfeed, lyield, lname in [[rfeed, ryield, "Rising"], [ffeed, fyield, "Falling"]]:
            series = pd.Series(lyield, index=pd.Index(lfeed,
                name=f'{lname} Feed'), name=f'{lname} Yield')
            ols_out = self._ols(series)
            print(ols_out)
            ols_lines[lname] = ols_out

        data = {
             #label: [[ rf,  ff], [ ry,  fy]]
              '1A' : [[  0,   0], [521,  93]],
              '1B' : [[  0,   0], [479,  82]],
              '2A' : [[400, 400], [707,  92]],
              '3A' : [[700, 100], [873, 309]],
              '3B' : [[700, 100], [830, 226]],
              '4A' : [[100, 700], [277, 319]],
              '5A' : [[400, 400], [631, 274]],
              }

        figure = plt.figure()
        ax = plt.gca()
        colors = list(plt.get_cmap('Dark2').colors)
        legend_points = []
        legend_labels = []
        handler_map = {}
        for exp_code, exp_data in data.items():
            lfeed, lyield = exp_data
            rf, ff = lfeed
            ry, fy = lyield
            color = colors.pop(0)

            rp = plt.scatter(rf, ry, marker='P', color=color)
            fp = plt.scatter(ff, fy, marker='s', color=color)

            legend_points.append((rp, fp))
            legend_labels.append(f"{exp_code}")
        
        for name, ols_out in ols_lines.items():
            xlim = np.array(plt.xlim())
            m = ols_out['slope']
            b = ols_out['intercept']
            rsqr = ols_out['r-sqr']
            line_fu = lambda x: m*x + b
            y = line_fu(xlim)
            lstyle = '-' if name == 'Rising' else '--'
            plt.plot(xlim, y, color='k', linestyle=lstyle)
            plt.xlim(xlim)

            an_x = 250
            an_y = line_fu(an_x)
            label = rf"{name} Limb ($r^2 = {rsqr:0.2f}$)" + '\n' + \
                    rf"$y = {m:0.2f}x + {b:0.0f}$"
            print(an_x, an_y, label)
            ax.annotate(label,
                    xy=(an_x, an_y), xycoords='data',
                    xytext=(-75, 50), textcoords='offset pixels',
                    horizontalalignment='left', verticalalignment='top',
                    )

        plt.legend(legend_points, legend_labels)
        plt.ylabel('Bedload Transport Yield (kg)')
        plt.xlabel('Total Sediment Feed for Limb (kg)')

        self.figure_subdir = self.figure_subdir_dict['synthesis']
        figure_name = "yield_vs_feed.png"
        self.save_figure(figure_name)

        plt.show()


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


    # Processing and text exporting
    def process_and_export_data(self, mode='all_Qs_data'):
        mode_dict = {
                'all_Qs_data' : self._process_export_all_Qs_data,
                'smoothed_Qs' : self._process_export_smoothed_Qs_data,
                '5min_Qs' : self._process_export_5min_Qs_data,
                '5min_suite' : self._process_export_5min_data,
                'all_data_xlsx' : self._process_export_all_data_xlsx, 
                }
        mode_dict[mode]()
        #reload_fu = self.omnimanager.reload_Qs_data
        #reload_fu = self.omnimanager.reload_depth_data
        #reload_fu = self.omnimanager.reload_gsd_data

    def _process_export_all_data_xlsx(self):
        # Export all data into excel files. Each type of data gets an excel 
        # file with separate sheets for each experiment.
        self.export_subdir = 'all_data'

        ## Export lighttable data
        self.__export_all_Qs_xlsx()

        ## Export depth data
        self.__export_all_depths_xlsx()

        ## Export surface gsd
        self.__export_all_surface_gsd_xlsx()
        
        ## Export trap sieve totals and gsd data
        self.__export_all_trap_total_xlsx()
        self.__export_all_trap_gsd_xlsx()

    def __export_all_Qs_xlsx(self):
        # Export Qs data
        exp_codes = self.omnimanager.get_exp_codes()
        self.omnimanager.reload_Qs_data(
                add = ['limb', 'period_range'],
                )

        all_Qs_data_dict = self.gather_Qs_data()
        for Qs_data in all_Qs_data_dict.values():
            Qs_data.reset_index(inplace=True)
            Qs_data.set_index(
                    ['limb', 'discharge', 'period_range',
                        'exp_time', 'exp_time_hrs'], 
                    inplace=True)
            Qs_data.drop(columns=['index'], inplace=True)

        Qs_filename = 'all_lighttable_data'
        self.export_xlsx(all_Qs_data_dict, Qs_filename, key_order=exp_codes)

    def __export_all_depths_xlsx(self):
        # Export depth data
        exp_codes = self.omnimanager.get_exp_codes()
        self.omnimanager.reload_depth_data()

        all_depth_data_dict = self.gather_depth_data({}) 

        for depth_data in all_depth_data_dict.values():
            depth_data.reset_index(inplace=True)
            depth_data.sort_values(
                    by=['exp_time', 'location'], ascending=[True, False],
                    inplace=True)
            depth_data['exp_time_hrs'] = depth_data['exp_time'] / 60

            depth_data.set_index(
                    ['limb', 'discharge', 'period_time', 'location',
                        'exp_time', 'exp_time_hrs'], 
                    inplace=True)

        depth_filename = 'all_depth_data'
        self.export_xlsx(all_depth_data_dict, depth_filename, 
                key_order=exp_codes)

    def __export_all_trap_total_xlsx(self):
        # Export trap mass data
        exp_codes = self.omnimanager.get_exp_codes()
        self.omnimanager.reload_masses_data()

        all_masses_data_dict = self.gather_masses_data({
            'add_cols' : ['limb', 'discharge', 'exp_time_hrs', 'period_range'],
            }) 

        for masses_data in all_masses_data_dict.values():
            masses_data.reset_index(inplace=True)

            masses_data.set_index(
                    ['limb', 'discharge', 'period_range',
                        'exp_time', 'exp_time_hrs'], 
                    inplace=True)

        masses_filename = 'all_trap_total_data'
        self.export_xlsx(all_masses_data_dict, masses_filename, 
                key_order=exp_codes)

    def __export_all_surface_gsd_xlsx(self):
        # Export trap mass data
        exp_codes = self.omnimanager.get_exp_codes()
        self.omnimanager.reload_gsd_data()

        all_gsd_data_dict = self.gather_gsd_data({
            'add_cols' : ['limb', 'discharge', 'exp_time_hrs'],
            }) 

        for gsd_data in all_gsd_data_dict.values():
            gsd_data.reset_index(inplace=True)

            gsd_data.sort_values(by=['exp_time', 'sta_str'], inplace=True)

            gsd_data.set_index(
                    ['limb', 'discharge', 'period',
                        'exp_time', 'exp_time_hrs',
                        'scan_length', 'sta_str',
                        ], 
                    inplace=True)
            gsd_data.drop(columns=['exp_code', 'step', 'scan_name'], 
                    inplace=True)

        gsd_filename = 'all_surface_gsd_data'
        self.export_xlsx(all_gsd_data_dict, gsd_filename, 
                key_order=exp_codes)

    def __export_all_trap_gsd_xlsx(self):
        # Export trap mass data
        exp_codes = self.omnimanager.get_exp_codes()
        self.omnimanager.reload_sieve_data()

        all_gsd_data_dict = self.gather_sieve_data({
            'add_cols' : ['limb', 'discharge', 'exp_time_hrs', 'period_range'],
            })

        for gsd_data in all_gsd_data_dict.values():
            gsd_data.reset_index(inplace=True)
            gsd_data.sort_values(by='exp_time', inplace=True)

            gsd_data.set_index(
                    ['limb', 'discharge', 'period_range',
                        'exp_time', 'exp_time_hrs',
                        ], 
                    inplace=True)

        gsd_filename = 'all_trap_gsd_data'
        self.export_xlsx(all_gsd_data_dict, gsd_filename, 
                key_order=exp_codes)


    def export_xlsx(self, data_dict, filename, key_order=None, xlsx_kwargs={}):
        export_dir = ospath_join(self.export_destination, self.export_subdir)
        hm.ensure_dir_exists(export_dir)

        filepath = ospath_join(export_dir, f"{filename}.xlsx")
        self.logger.write(f"Exporting data to {filepath}")
        
        self.data_loader.save_xlsx(
                data_dict, filepath, 
                key_order = key_order,
                add_path = False,
                )

    def _process_export_all_Qs_data(self):
        self.omnimanager.reload_Qs_data(
                add = ['limb', 'period_range'],
                )
        all_data_dict = self.gather_Qs_data()

        del all_data_dict['2B']
        keys = sorted(all_data_dict.keys())

        all_data = pd.concat(all_data_dict, names=['exp_code', 'period_time'],
                keys=keys)

        all_data.reset_index(inplace=True)
        all_data.set_index([
            'exp_code', 'limb', 'discharge', 'period_range', 'period_time'], 
                inplace=True)

        drop_cols = ['missing ratio', 'vel', 'sd vel', 'number vel',]
        all_data.drop(columns=drop_cols, inplace=True)

        filename = 'all_Qs_data.csv'
        #print(all_data_dict['2A'].index)
        #print(all_data_dict['2A'].columns)
        #print(all_data)
        #print(all_data.index)
        #print(all_data.columns)
        #print()
        #print(all_data.loc[('3B', 'rising', 't00-t60', slice(None)), :])

        self.export_data(all_data, filename)

    def _process_export_smoothed_Qs_data(self):
        self.omnimanager.reload_Qs_data(
                add = ['limb', 'period_range'],
                )
        all_data_dict = self.gather_Qs_data()
        rolled_data_dict = {}

        del all_data_dict['2B']
        keys = sorted(all_data_dict.keys())

        #for key in keys:
        #    exp_data = all_data_dict[key]

        all_data = pd.concat(all_data_dict, names=['exp_code', 'period_time'],
                keys=keys)
        all_data.reset_index(inplace=True)
        all_data.set_index([
            'exp_code', 'limb', 'discharge', 'period_range', 'period_time', 
            'timestamp', 'exp_time', 'exp_time_hrs'], 
                inplace=True)

        drop_cols = ['missing ratio', 'vel', 'sd vel', 'number vel',]
        all_data.drop(columns=drop_cols, inplace=True)

        grouped = all_data.groupby(level='exp_code', axis=0)
        window = 5*60
        rolled = grouped.rolling(
                window=window, 
                min_periods=1,
                center=True,
                )
        sum_data = rolled.sum()
        sum_data.drop(columns=[
            'D10', 'D16', 'D25', 'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax'
            ], inplace=True)
        """
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
        """

        sum_filename = 'smoothed_5min_sums_Qs_data.csv'
        self.export_data(sum_data, sum_filename)

        mean_data = rolled.mean()
        mean_filename = 'smoothed_5min_means_Qs_data.csv'
        self.export_data(mean_data, mean_filename)

    def _process_export_5min_Qs_data(self):
        self.omnimanager.reload_Qs_data(
                add = ['limb', 'period_range'],
                )
        all_data_dict = self.gather_Qs_data()
        rolled_data_dict = {}

        del all_data_dict['2B']
        keys = sorted(all_data_dict.keys())

        #for key in keys:
        #    exp_data = all_data_dict[key]

        # Prepare all the data
        all_data = pd.concat(all_data_dict, names=['exp_code', 'period_time'],
                keys=keys)
        all_data.reset_index(inplace=True)
        all_data.set_index([
            'exp_code', 'limb', 'discharge', 'period_range', 'period_time', 
            #'timestamp', 
            'exp_time', 'exp_time_hrs'], 
                inplace=True)
        drop_cols = ['missing ratio', 'vel', 'sd vel', 'number vel',]
        all_data.drop(columns=drop_cols, inplace=True)

        # Group into periods
        group_cols = ['exp_code', 'limb', 'discharge', 'period_range']
        grouped = all_data.groupby(level=group_cols, axis=0)

        def sort_subset(df):
            # Sort by timestamp to get right limb order
            # For some reason sort_remaining=False for sort_index() isn't 
            # working
            df = df.sort_values(by='timestamp', ascending=True)
            df = df.drop(columns='timestamp')
            df = df.drop(columns=[
                'D10', 'D16', 'D25', 'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax',
                ])

            # Sort the experiments by splitting into a dict, then concatenating
            df_dict = {}
            for name, group in df.groupby(axis='index', level='exp_code'):
                df_dict[name] = group
            df = pd.concat(df_dict, keys=keys)
            df.index = df.index.droplevel(level=1)
            df.index.rename('exp_code', level=0, inplace=True)

            return df

        # Get heads and tails of each period
        window = 5*60
        heads = grouped.head(n=window)
        tails = grouped.tail(n=window)

        # Calc sum of each head or tail group
        head_groups = heads.groupby(level=group_cols, axis=0)
        head_sums = head_groups.sum()
        tail_groups = tails.groupby(level=group_cols, axis=0)
        tail_sums = tail_groups.sum()

        # Reformat the data to get the right order
        head_sums = sort_subset(head_sums)
        tail_sums = sort_subset(tail_sums)

        """
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
        """

        head_filename = 'head_5min_sums_Qs_data.csv'
        self.export_data(head_sums, head_filename)

        tail_filename = 'tail_5min_sums_Qs_data.csv'
        self.export_data(tail_sums, tail_filename)

        difference_filename = 'difference_5min_sums_Qs_data.csv'
        self.export_data(head_sums - tail_sums, difference_filename)

    def _process_export_5min_data(self):
        ## Load data
        # Load depth data
        self.omnimanager.reload_depth_data()
        depth_data_dict = self.gather_depth_data({
            #'new_index' : ['limb', 'discharge', 'period_],
            })

        # Load bed gsd data
        self.omnimanager.reload_gsd_data()
        gsd_data_dict = self.gather_gsd_data({
            #'columns'   : [y_name[-3:]],
            'new_index' : ['limb', 'discharge', 'sta_str'],
            })

        # Load Qs data
        self.omnimanager.reload_Qs_data(
                add = ['limb', 'period_range'],
                )
        Qs_data_dict = self.gather_Qs_data()

        ## Remove experiments
        rolled_data_dict = {}
        remove_keys = ['2B', '1A', '3A', '4A', '5A']
        for key in remove_keys:
            del Qs_data_dict[key]
            del depth_data_dict[key]
        keys = sorted(Qs_data_dict.keys())

        Qs_head, Qs_tail = self.__process_Qs_5min(Qs_data_dict, keys)
        geom_head, geom_tail = self.__process_depth_5min(depth_data_dict, keys)
        gsd_tail = self.__process_gsd_5min(gsd_data_dict, keys)

        ## Combine depth and Qs data
        head_data = pd.concat([Qs_head, geom_head], axis=1)
        tail_data = pd.concat([Qs_tail, gsd_tail, geom_tail], axis=1)

        # Hacky way to get the column order
        column_order = tail_data.columns.values
        assert(np.all(np.in1d(head_data.columns.values, column_order)))

        # Combine head and tail data into one dataframe
        combined_data = pd.concat(
                {'first_5min' : head_data, 'last_5min' : tail_data},
                axis='columns', names=['time_set', 'variable'],
                )

        # Sort index
        combined_data.sort_values(by=('first_5min', 'order_key'), inplace=True)

        # Reorder and sort columns
        print(column_order)
        print(combined_data)

        combined_data = combined_data.reorder_levels(
                ['variable', 'time_set'],
                axis=1)
        combined_data.sort_index(axis=1, inplace=True)
        combined_data = combined_data.reindex(
                columns=column_order, level='variable')

        combined_data.drop(columns='order_key', inplace=True)

        print(combined_data)

        filename = '5min_data.csv'
        self.export_data(combined_data, filename)
        
    def __process_Qs_5min(self, Qs_data_dict, keys):
        ## Combine the Qs dict dataframes and set the index
        Qs_data = pd.concat(Qs_data_dict, names=['exp_code', 'period_time'],
                keys=keys)
        Qs_data.reset_index(inplace=True)
        Qs_data.set_index([
            'exp_code', 'limb', 'discharge', 'period_range', 'period_time', 
            'timestamp', 
            'exp_time', 'exp_time_hrs'], 
                inplace=True)

        # Drop some columns
        drop_cols = ['missing ratio', 'vel', 'sd vel', 'number vel',
                #'Bedload all', 
                #'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                #'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                #'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                #'Bedload 22', 'Bedload 32', 'Bedload 45',
                ## Grain counts
                'Count all', 
                'Count 0.5', 'Count 0.71', 'Count 1', 'Count 1.4',
                'Count 2', 'Count 2.8', 'Count 4', 'Count 5.6', 'Count 8',
                'Count 11.2', 'Count 16', 'Count 22', 'Count 32', 'Count 45',
                # Statistics
                'D10', 'D16', 'D25', 'D75', 'D90', 'D95', 'Dmax',
                'D50', 'D84', 
                ]
        Qs_data.drop(columns=drop_cols, inplace=True)
        
        # Rename gsd columns to floats
        def rename(name):
            if name == 'Bedload all':
                return 'bedload_total'
            elif 'Bedload ' in name:
                return np.float(name.split()[1])
            elif name == 'Count all':
                return 'count_total'
            elif 'Count ' in name:
                return np.float(name.split()[1])
            else:
                return name
        Qs_data.columns = [rename(s) for s in Qs_data.columns.values]

        # Create a column to track the order to allow for easy recovery of the 
        # original order
        order_key_np = np.arange(Qs_data.shape[0])
        Qs_data.insert(0, 'order_key', order_key_np)

        # Group into steps
        group_cols = ['exp_code', 'limb', 'discharge']
        Qs_grouped = Qs_data.groupby(level=group_cols, axis=0)

        # Get heads and tails of each step group (first and last 5 minutes)
        window = 5*60
        Qs_heads = Qs_grouped.head(n=window)
        Qs_tails = Qs_grouped.tail(n=window)

        Qs_combined = {}
        for name, subset in ('head', Qs_heads), ('tail', Qs_tails):
            # Calc sum of each head or tail group
            subset_groups = subset.groupby(level=group_cols, axis=0)
            sum = subset_groups.sum()

            # Split bedload total from bedload distributions
            series_list = [
                    sum.pop('order_key'),
                    sum.pop('bedload_total'),
                    ]

            sum.columns = sum.columns.astype(np.float)
            # Calculate Di values
            for i in [50, 84]:
                Di = self.calc_Di(sum, target_Di=i)
                Di.name = f"bedload_{Di.name}"
                series_list.append(Di)

            # Merge Di series
            Qs_combined[name] = pd.concat(series_list, axis=1)

        return Qs_combined['head'], Qs_combined['tail']

    def __process_depth_5min(self, depth_data_dict, keys):
        ## Prepare depth and slope
        # Merge depth data dict
        depth_data_all = pd.concat(depth_data_dict, names=['exp_code'],
                keys=keys)
        depth_data_all /= 100 # convert to m
        depth_data_all.drop(columns='exp_time', inplace=True)
        depth_data = depth_data_all.xs('depth', level='location')
        surface_data = depth_data_all.xs('surface', level='location')

        # Calc mean depth and extract 10min and 50min times
        depth_mean = depth_data.mean(axis=1)
        depth_mean.name = 'depth_mean'

        # Calc water surface slope
        ols_out, flume_elevations = self._calc_retrended_slope(
                surface_data, flume_elevations=None, intercept=None)
        slope = ols_out['slope']
        #ols_out.drop(columns=['r-sqr', 'intercept'], inplace=True)

        # Calc shear
        rho_w = 1000 # kg/m**3
        g = 9.81 # m/s**2
        shear = rho_w * g * depth_mean * slope
        shear.name = "shear"

        channel_geom = pd.concat([depth_mean, slope, shear], axis=1)

        geom_head = channel_geom.xs(10, level='period_time')
        geom_tail = channel_geom.xs(50, level='period_time')

        return geom_head, geom_tail

    def __process_gsd_5min(self, gsd_data_dict, keys):
        ## Prepare bed surface gsd
        # Merge depth data dict
        gsd_all = pd.concat(gsd_data_dict, names=['exp_code'],
                keys=keys)
        gsd_drop_cols = ['exp_code', 'step', 'period', 'scan_length', 
                'Sigmag', 'Dg', 'La', 'D90', 'D84', 'D50', 'D16', 'D10',
                'Fsx', 'exp_time', 'scan_name']
        gsd_all.drop(columns=gsd_drop_cols, inplace=True)

        # Split into steps
        group_cols = ['exp_code', 'limb', 'discharge']
        gsd_groups = gsd_all.groupby(level=group_cols, axis=0)
        gsd_sum = gsd_groups.sum(axis=1)

        # Add all the gsds for each step
        float_gsd_sums = gsd_sum.astype(np.float)
        float_gsd_sums.columns = float_gsd_sums.columns.astype(np.float)

        # Calculate Di series
        Di_list = []
        for i in [50, 84]:
            Di = self.calc_Di(float_gsd_sums, target_Di=i)
            Di.name = f"bed_{Di.name}"
            Di_list.append(Di)

        # Merge Di series
        Di_combined = pd.concat(Di_list, axis=1)

        return Di_combined


class GSCategory:

    def __init__(self, new_col_name, gs_cats, legend_name=None):
        # new_col_name is a dataframe friendly version of the name
        # gs_cats is a list of GS sizes for the light table (will 
        # convert for sieve data)
        # legend_name is a plot friendly version of the name
        #
        # For use with Maria 2019 plotting function
        self.col_name = new_col_name

        if legend_name is None:
            # Do a series of replacements to convert column name into 
            # legend name
            name = new_col_name
            name = name.replace('lt_', r'$<$')
            name = name.replace('geq_', r'$\geq$')
            name = name.replace('_', '-')
            name = name.replace('mm', ' mm')
            name = name.capitalize() # 'total' to 'Total'
            self.legend_name =  name
        else:
            self.legend_name = legend_name
        
        self.bedload_cols = []
        self.sieve_cols = []

        if self.col_name == 'total':
            self.bedload_cols = ['Bedload all']
            self.sieve_cols = [0.50, 0.71, 1.00, 1.41, 2.00, 2.83, 
                    4.00, 5.66, 8.00, 11.20, 16.00, 22.30, 32.00, 
                    45.00]
        else:
            for cat in gs_cats:
                self.bedload_cols.append(f"Bedload {cat}")
                if cat == '1.4':
                    sieve_col = 1.41
                elif cat == '2.8':
                    sieve_col = 2.83
                elif cat == '5.6':
                    sieve_col = 5.66
                elif cat == '22':
                    sieve_col = 22.3
                else:
                    sieve_col = float(cat)
                self.sieve_cols.append(sieve_col)



if __name__ == "__main__":
    # Run the script
    grapher = UniversalGrapher(fig_debug = True)

    ### Export data
    #grapher.process_and_export_data(mode='5min_suite')
    #grapher.process_and_export_data(mode='all_data_xlsx')

    ### Trap data plots
    #grapher.make_simple_feed_plots()
    #grapher.make_sieve_di_plots(Di=84)
    #grapher.make_simple_sieve_plots()
    #grapher.make_simple_masses_plots()

    ### DEM data plots
    #grapher.make_dem_subplots(plot_2m=True)
    #grapher.make_dem_subplots(plot_2m=False)
    #grapher.make_dem_semivariogram_plots()
    #grapher.make_dem_stats_plots()
    #grapher.make_dem_stats_plots()
    #grapher.make_dem_roughness_plots()

    ### Depth and slope plots
    #grapher.make_loc_shear_plots()
    #grapher.make_flume_shear_plots()
    #grapher.make_mobility_plots(t_interval='step')
    #grapher.make_shields_plots(Di_target='D50')
    #grapher.make_avg_depth_plots(plot_2m=True)
    #grapher.make_avg_slope_plots(plot_2m=False, plot_trends='surface')

    ### Surface GSD data plots
    #!grapher.make_box_gsd_plots(x_name='exp_time', y_name='D50')
    #!grapher.make_box_gsd_plots(x_name='sta_str',  y_name='D50')
    #!grapher.make_mean_gsd_time_plots(y_name=['D16', 'D50', 'D84'])
    #!grapher.make_mean_gsd_time_plots(y_name=['Fsx'])#, 'D90'])
    #grapher.make_gsd_plots(x_name='exp_time', y_name='D84')
    #!grapher.make_gsd_plots(x_name='sta_str',  y_name='D50')

    ### Light table bedload plots
    #!grapher.make_manual_Di_plots()
    #grapher.make_Qs_plots(y_name='Bedload all')
    #grapher.make_Qs_plots(y_name='D50')
    #grapher.make_Qs_plots(y_name='D84')
    #grapher.make_Di_ratio_plots()
    #!grapher.make_experiment_plots()
    #!grapher.make_bedload_feed_plots()

    ### Hysteresis plots
    # 'bed-D50', 'bed-D84'
    #grapher.make_pseudo_hysteresis_plots(y_name='bed-D84', plot_2m=True)
    # 'depth', 'slope'
    #grapher.make_pseudo_hysteresis_plots(y_name='slope', plot_2m=True)
    # 'Bedload all', 'D50', 'D84'
    #grapher.make_pseudo_hysteresis_plots(y_name='Bedload all', plot_2m=False)
    #!grapher.make_hysteresis_plots()

    ### Mass balance and miscellaneous plots
    #!grapher.make_cumulative_plots()
    #grapher.make_cumulative_mass_bal_plots()
    #!grapher.make_transport_histogram_plots()
    #grapher.make_yield_feed_plot()
    grapher.make_maria_2019_plot()
