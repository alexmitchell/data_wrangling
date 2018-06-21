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
from helpyr_misc import *
from logger import Logger
from data_loading import DataLoader

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
    def generic_greeting(self, name, load_fu, plot_fu):
        self.logger.write([f"Making {name} plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(load_fu,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(plot_fu,
                before_msg=f"Plotting {name}",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def create_experiment_subplots(self, rows=4, cols=2):
        # So that I can create a standardized grid for the 8 experiments
        fig, axs = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=(16,10))
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

    def roll_data(self, data, roll_kwargs={}, plot_kwargs={}):
        # Make rolling averages on the provided data.

        time = data.loc[:, 'exp_time_hrs']
        y_var = roll_kwargs.pop('y')
        y = data.loc[:, y_var]
        series = pd.Series(data=y.values, index=time)

        rolled = series.rolling(**roll_kwargs)
        roll_kwargs['y'] = y_var

        #rolled = data.rolling(**roll_kwargs)
        average = rolled.mean()

        if self.rolling_av is None:
            self.rolling_av = average
        else:
            self.rolling_av = self.rolling_av.append(average, verify_integrity=True)

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
        filepath = ospath_join(self.figure_destination, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')

    
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
        df.plot(**plot_kwargs)
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
        exp_codes = list(data.keys())
        exp_codes.sort()
        for exp_code, ax in zip(exp_codes, axs.flatten()):
            self.logger.write(f"Plotting experiment {exp_code}")

            plot_kwargs['ax'] = ax
            experiment = self.omnimanager.experiments[exp_code]
            ax.set_title(f"Experiment {exp_code} {experiment.name}")

            plot_fu(exp_code, data, plot_kwargs)

        post_plot_fu(fig, axs, plot_kwargs)

        # Generate a figure name and save the figure
        if save:
            self.save_figure(figure_name)
        plt.show()

    def format_generic_figure(self, fig, axs, plot_kwargs, fig_kwargs):
        # Must add xlabel, ylabel, and title to fig_kwargs
        x = plot_kwargs['x']
        y = plot_kwargs['y']
        xlabel = fig_kwargs.pop('xlabel') # Common xlabel
        ylabel = fig_kwargs.pop('ylabel') # Common ylabel
        title_str = fig_kwargs.pop('title') # Common title
        plot_labels = [] if 'legend_labels' not in fig_kwargs \
                else fig_kwargs.pop('legend_labels') # Common legend labels

        # Set the spacing and area of the subplots
        fig.tight_layout()
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
        fig.text(0.01, 0.5, ylabel, va='center', usetex=True,
                fontsize=fontsize, rotation='vertical')
        
        # Make a title
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)


    # Functions to plot only dem data
    def make_dem_subplots(self):
        self.generic_greeting("dem semivariogram",
                self.omnimanager.reload_dem_data,
                self.plot_dem_subplots)

    def plot_dem_subplots(self):
        # Can't use the generic_plot_experiments function because of the way I 
        # want to display and save them.
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
        filename_base = f"dems_{{}}_2m.png"

        # Start plotting; 1 plot per experiment
        color_min, color_max = settings.dem_color_limits
        exp_codes = list(dem_data.keys())
        exp_codes.sort()
        for exp_code in exp_codes:
            self.logger.write(f"Creating plots for {exp_code}")
            self.logger.increase_global_indent()
            
            # Create the subplot for this experiment
            fig, axs = (plt.subplots(3, 3, sharey=True, sharex=True,
                    figsize=(18,10)))
            last_image=None
            axs = axs.flatten()

            # Calculate and plot the semivariograms for this experiment
            for period_key, period_dem in dem_data[exp_code].items():
                limb, discharge, time, exp_time = period_key
                self.logger.write(f"Plotting {limb} {discharge} {time}")
                assert(time == 't60')
                ax = axs[get_ax_index(limb, discharge)]

                ax.set_title(f"{limb} {discharge} {exp_time//60}hrs")

                px_y, px_x = period_dem.shape
                #ax.set_ybound(0, px_y)
                ax.imshow(period_dem, vmin=color_min, vmax=color_max)
                #ax.set_xlim(0, px_x)
                #ax.set_ylim(0, px_y)
                #plt.hist(data.flatten(), 50, normed=True)
                #plt.xlim((120, 220))

            self.logger.decrease_global_indent()

            # Format the figure
            xlabel = rf"Longitudinal distance (px)"
            ylabel = rf"Transverse distance (px)"
            title_str = rf"{exp_code} DEM 2m subsections with wall trim"
            fontsize = 16

            plt.tight_layout()
            fig.subplots_adjust(top=0.9, left=0.05, bottom=0.075, right=0.95)
            plt.xlim((0, px_x))
            plt.ylim((0, px_y))
            
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
            partial_filepath = ospath_join("dem_subplots", filename)
            self.save_figure(partial_filepath)

            #plt.show()


    def make_dem_semivariogram_plots(self):
        self.generic_greeting("dem semivariogram",
                self.omnimanager.reload_dem_data,
                self.plot_dem_semivariograms)

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
        exp_codes = list(dem_data.keys())
        exp_codes.sort()
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
            partial_filepath = "dem_variograms/" + filename
            self.save_figure(partial_filepath)

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
        self.generic_greeting("dem stats time",
                self.omnimanager.reload_dem_data,
                self.plot_dem_stats)

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
        exp_codes = list(dem_data.keys())
        exp_codes.sort()
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
        m2cm = 100
        cm2m = 1/100

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
        m2cm = 100
        cm2m = 1/100

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
        m2cm = 100
        cm2m = 1/100
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
        m2cm = 100
        cm2m = 1/100
        ax = plot_kwargs['ax']

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


    def make_avg_slope_plots(self):
        self.logger.write([f"Making flume-averaged slope time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_depth_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_avg_slope,
                before_msg=f"Plotting slope vs time",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_avg_slope(self):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'slope'
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
        exp_codes = list(depth_data.keys())

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        figure_name = f"comp_avg-slope_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_avg_slope, self._format_avg_slope, 
            depth_data, plot_kwargs, figure_name)

    def _plot_avg_slope(self, exp_code, depth_data, plot_kwargs):
        # Do stuff during plot loop

        m2cm = 100
        cm2m = 1/100

        # Get the data
        exp_data = depth_data[exp_code] * cm2m
        #stats_data_frames = {}

        init_bed_height = settings.init_bed_height * cm2m
        flume_elevations = None

        #slice_all = slice(None)
        for location in ['surface', 'bed']:
            data = exp_data.xs(location, level='location')
            #data = exp_data.loc[(slice_all, location), :]

            intercept = init_bed_height if location == 'bed' else None
            #stats_data_frames[location] = ols_out
            
            ols_out, flume_elevations = self._calc_retrended_slope(data, flume_elevations, intercept)
            # Not actually grouped, but can still use self.plot_group
            self.plot_group(ols_out, plot_kwargs)

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


    def make_avg_depth_plots(self):
        self.logger.write([f"Making flume-averaged depth time plots..."])

        indent_function = self.logger.run_indented_function

        indent_function(self.omnimanager.reload_depth_data,
                before_msg="Loading data", after_msg="Data Loaded!")

        indent_function(self.plot_avg_depth,
                before_msg=f"Plotting depth vs time",
                after_msg="Finished Plotting!")

        self.logger.end_output()

    def plot_avg_depth(self):
        # Do stuff before plot loop
        x_name = 'exp_time'
        y_name = 'depth'
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
        exp_codes = list(depth_data.keys())

        filename_x = x_name.replace('_', '-').lower()
        filename_y = y_name.replace('_', '-').lower()
        #figure_name = f"box_flume-depth_v_{filename_x}.png"
        figure_name = f"flume-depth_v_{filename_x}.png"

        # Start plot loop
        self.generic_plot_experiments(
            self._plot_avg_depth, self._format_avg_depth, 
            depth_data, plot_kwargs, figure_name)

    def _plot_avg_depth(self, exp_code, depth_data, plot_kwargs):
        # Do stuff during plot loop

        draw_boxplot = False

        # Get the data
        exp_data = depth_data[exp_code]
        depth = exp_data.xs('depth', level='location')
        if draw_boxplot:
            depth.T.boxplot(ax=plot_kwargs['ax'], showfliers=False)
        else:
            avg_depths = depth.mean(axis=1)
            avg_depths.rename('depth', inplace=True)

            # Not actually grouped, but can still use self.plot_group
            self.plot_group(avg_depths, plot_kwargs)

    def _format_avg_depth(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Depth (cm)",
                'title'         : r"Flume-averaged water depth",
                'legend_labels' : [r"Water depth"],
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
            sort_index = kwargs['new_index'][0]
            combined_data = pd.concat(frames)
            depth_data[exp_code] = combined_data.sort_index(level=sort_index)
        return depth_data

    def _gather_depth_data(self, period, kwargs):
        # Have periods add themselves to a precursor of the overall depth 
        # dataframe dict
        #cols = kwargs['columns']
        new_index = kwargs['new_index']
        exp_code = period.exp_code

        data = period.depth_data
        if data is not None:
            index_names = list(data.index.names)
            drop_list = list(set(index_names) - set(new_index))
            data.reset_index(inplace=True)
            data.set_index(new_index, inplace=True)
            data.drop(drop_list, axis=1, inplace=True)
            if exp_code not in self.depth_data_frames:
                self.depth_data_frames[exp_code] = []
            self.depth_data_frames[exp_code].append(data.loc[:, :])

    
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
        exp_codes = list(gsd_data.keys())
        exp_codes.sort()
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
            ax.set_title(f"Experiment {exp_code} {experiment.name}")

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
        plt.suptitle(title_str, fontsize=fontsize, usetex=True)


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
        y_column = 'Bedload all'
        #y_column = 'D50'
        roll_window = 10 #minutes

        plot_kwargs = {
                'x'    : x_column,
                'y'    : y_column,
                'kind' : 'scatter',
                'logy' : True,
                'xlim' : (0.5, 8.5),
                'ylim' : (0.001, 5000), # for use with logy
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
        figure_name = f"{filename_y_col}_roll-{roll_window}min{logy_str}.png"

        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16,10))
        #twin_axes = []

        # Make one plot per experiment
        exp_codes = list(self.omnimanager.experiments.keys())
        exp_codes.sort()
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
        #exp_codes = list(self.omnimanager.experiments.keys())
        #exp_codes.sort()
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

        figure_name = f"cumulative_mass_balance.png"

        plot_kwargs = {
                'x'      : x_column,
                'y'      : [cumsum_column, massbal_column],
                #'y'      : [cumsum_column],
                'kind'   : 'line',
                'xlim'   : (0, 8),
                #'ylim'   : (0.001, 5000),
                #'ylim'   : (0, 200),
                'legend' : False,
                'color'  : ['r', 'g'],
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
        cumsum_col = plot_kwargs['y'][0] # Not very safe...

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
        feed_rate = tokens.PeriodData.feed_rates[exp_code]
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
        mass_balance = feed_cumsum - data[cumsum_col]
        data['mass_balance'] = mass_balance
        #print(feed_cumsum.astype(np.int))
        #print(feed_cumsum.astype(np.int)[8])
        #print([int(calc_cumsum_feed(t)) for t in [0,1.5,2,2.5,3,8]])

        # Plot it!
        self._time_plot_prep(data, plot_kwargs)
        plot_kwargs['ax'].grid(True, color='#d8dcd6')

    def _format_cumulative_mass_bal(self, fig, axs, plot_kwargs):
        # Format the figure after the plot loop
        fig_kwargs = {
                'xlabel'        : r"Experiment time (hours)",
                'ylabel'        : r"Mass (kg)",
                'title'         : r"Cumulative sum of bedload output and flume mass balance",
                #'legend_labels' : [r"Shear from surface", r"Shear from bed"],
                'legend_labels' : ["Cumulative\nbedload", "Bedload\nmass balance"],
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


    def load_Qs_data(self):
        #self.experiments = self.loader.load_pickle(self.exp_omnipickle)
        accumulate_kwargs = {
                'check_ignored_fu' : self._check_ignore_period,
                }
        self.omnimanager.reload_Qs_data(kwargs=accumulate_kwargs)

    def gather_Qs_data(self, gather_kwargs={}):
        Qs_data = {}
        for exp_code in self.omnimanager.experiments.keys():
            experiment = self.omnimanager.experiments[exp_code]
            Qs_data[exp_code] = experiment.accumulated_data
        return Qs_data


    # General functions to be given to PeriodData instances
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
    #grapher.make_dem_subplots()
    grapher.make_dem_semivariogram_plots()
    #grapher.make_dem_stats_plots()
    #grapher.make_dem_stats_plots()
    #grapher.make_dem_roughness_plots()
    #grapher.make_loc_shear_plots()
    #grapher.make_flume_shear_plots()
    #grapher.make_avg_depth_plots()
    #grapher.make_avg_slope_plots()
    #grapher.make_box_gsd_plots(x_name='exp_time', y_name='D50')
    #grapher.make_box_gsd_plots(x_name='sta_str',  y_name='D50')
    #grapher.make_mean_gsd_time_plots(y_name=['D16', 'D50', 'D84'])
    #grapher.make_mean_gsd_time_plots(y_name=['Fsx'])#, 'D90'])
    #grapher.make_gsd_plots(x_name='exp_time', y_name='D50')
    #grapher.make_gsd_plots(x_name='sta_str',  y_name='D50')
    #grapher.make_experiment_plots()
    #grapher.make_hysteresis_plots()
    #grapher.make_cumulative_plots()
    #grapher.make_cumulative_mass_bal_plots()
    #grapher.make_transport_histogram_plots()
