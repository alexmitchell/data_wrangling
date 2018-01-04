import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
import statsmodels.api as sm

from xlrd.biffh import XLRDError

# From helpyr
import data_loading
from helpyr_misc import ensure_dir_exists
from logger import Logger
from crawler import Crawler

import global_settings

class PlottingCrawler (Crawler):

    def __init__(self, log_filepath="./log-files/log-plotting-crawler.txt", no_log=False):
        logger = Logger(log_filepath, default_verbose=True, no_log=no_log)
        Crawler.__init__(self, logger)

        self.mode_dict['plot-hr-av-profiles'] = self.run_hr_av_profiles
        self.mode_dict['plot-slope-profiles'] = self.run_slope_profiles

    def set_fig_dir(self, fig_dir):
        self.fig_dir = fig_dir
        ensure_dir_exists(fig_dir)

    def get_pickle_data(self, pickle_targets=['??-profile-*.pkl']):
        # Load targeted pickles from a predetermined path.
        # Default for bed and surface profile data.
        self.logger.write_section_break()
        self.logger.write(["Loading profile pickles"])

        data_path = self.root
        pickle_path = self.root
        self.loader = data_loading.DataLoader(data_path, pickle_path, self.logger)

        pickle_names = self.get_target_files(pickle_targets)
        profile_data = self.loader.load_pickles(pickle_names, add_path=False)

        self.logger.write(["Pickles Loaded"])
        return profile_data


    def run_hr_av_profiles(self):
        # Load pickles and hand off profile data to plotter. Each experiment 
        # will have one surface and one bed dataframe.
        profile_data = self.get_pickle_data()
        self.plot_experiments(profile_data, self.make_hr_av_plot)

    def run_slope_profiles(self):
        # Get pickled data and hand off the profile data to plotter. Each 
        # experiment will have one surface and one bed dataframe.
        profile_data = self.get_pickle_data()
        subplot_positions = {
                '1A' : (0,0),
                '1B' : (0,1),
                '2A' : (1,0),
                '2B' : (1,1),
                '3A' : (2,0),
                '3B' : (2,1),
                '4A' : (3,0),
                '5A' : (3,1),
                'nrows' : 4,
                'ncols' : 2,
                }
        self.subplot_experiments(profile_data, subplot_positions,
                self.make_slope_subplot, self.save_fig_slopes)


    def subplot_experiments(self, data, subplot_positions, plotting_function, saving_function=None):
        # Generic function which separates the experiments and then calls the 
        # provided plotting function in a subplot.

        self.logger.write("Plotting profiles...")
        self.logger.increase_global_indent()
        pathnames = data.keys()
        regex = re.compile(".*surface.pkl")
        surface_pathnames = filter(regex.match, pathnames)

        n_rows = subplot_positions['nrows']
        n_cols = subplot_positions['ncols']
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)

        for surface_pathname in surface_pathnames:
            # loading data
            fname = lambda fpath: os.path.split(fpath)[1]
            experiment = fname(surface_pathname)[0:2]
            bed_pathname = surface_pathname.replace("surface", "bed")
            self.logger.write(f"Plotting profiles for experiment {experiment}")
            self.logger.write([fname(surface_pathname),
                               fname(bed_pathname)], local_indent=1)

            exp_data = {'surface': data[surface_pathname],
                        'bed' : data[bed_pathname]
                       }

            subplot_pos = subplot_positions[experiment]
            ax = axs[subplot_pos]
            plotting_function(experiment, exp_data, fig, ax)
            self.logger.write_blankline()

        if saving_function is not None:
            filename = f"slope-profiles.png"
            saving_function(fig, filename)

        plt.show()
        self.logger.decrease_global_indent()
        self.logger.write(["Profile plotting complete"])

    def plot_experiments(self, data, plotting_function):
        # Generic function which separates the experiments and then calls the 
        # provided plotting function.

        self.logger.write("Plotting profiles...")
        self.logger.increase_global_indent()
        pathnames = data.keys()
        regex = re.compile(".*surface.pkl")
        surface_pathnames = filter(regex.match, pathnames)

        for surface_pathname in surface_pathnames:
            # loading data
            def fname(fpath):
                return os.path.split(fpath)[1]
            experiment = fname(surface_pathname)[0:2]
            bed_pathname = surface_pathname.replace("surface", "bed")
            self.logger.write(f"Plotting profiles for experiment {experiment}")
            self.logger.write([fname(surface_pathname),
                               fname(bed_pathname)], local_indent=1)

            exp_data = {'surface': data[surface_pathname],
                        'bed' : data[bed_pathname]
                       }

            plotting_function(experiment, exp_data)
            self.logger.write_blankline()

        self.logger.decrease_global_indent()
        self.logger.write(["Profile plotting complete"])


    def make_slope_subplot(self, experiment, data_dic, fig, ax):
        # Make slope plots
        self.logger.write([f"Making average slope plots for experiment {experiment}"])

        surface = data_dic['surface']
        bed = data_dic['bed']

        positions = surface.columns.values
        start = np.amin(positions)
        flume_slope = global_settings.flume_slope
        m2cm = 100
        cm2m = 1/100
        flume_elevations = positions * flume_slope

        profiles = {"surface" : [surface, "blue"], 
                    "bed"     : [bed, "black"],
                   }
        for name, (profile, plot_color) in profiles.items():

            trended_profile = profile * cm2m + flume_elevations
            profile_indices = trended_profile.index

            # Can't seem to find a good function to do the linear regression on 
            # each row...  So I'll have to loop over every row.
            
            regressions = pd.DataFrame(index = profile_indices,
                    columns=['Slope', 'Intercept', 'R-sqr'])
            xtick_labels = []
            for index in profile_indices.values:
                # Do regression
                dependent = trended_profile.loc[index].values
                independent = sm.add_constant(positions)#*m2cm)
                if not np.all(np.isnan(dependent)):
                    results = sm.OLS(dependent, independent, missing='drop').fit()
                    p = results.params

                    regressions.loc[index, 'Intercept'] = p[0]
                    regressions.loc[index, 'Slope'] = p[1]
                    regressions.loc[index, 'R-sqr'] = results.rsquared

                xtick_labels.append("{0[1]} {0[2]}L/s {0[3]}mins".format(index))

            legend_label =  f"{name.capitalize()} surface slope"
            regressions['Slope'].plot(ax=ax, color=plot_color,
                    label=legend_label)

        xtick_loc = np.arange(len(regressions))-1
        #plt.xticks(xtick_loc, xtick_labels, rotation=45)
        ax.set_xticks(xtick_loc)
        ax.set_xticklabels(xtick_labels, rotation=45)
        ax.set_xlabel("Experiment progression")
        ax.set_ylabel("Slope")
        ax.set_title(f"Experiment {experiment} average slope profiles")
        ax.tick_params(axis='both', top=True, right=True)

        plt.legend()

    def save_fig_slopes(self, figure, filename):
        # Save the figure
        
        # Format the plot layout
        figure = plt.gcf()
        figure.set_dpi(80)
        figure.set_size_inches(24, 13.5)
        plt.subplots_adjust(left=0.04, bottom = 0.15, right=0.98, top=0.95) # Reduce margins

        # Save it
        fig_dir = self.fig_dir
        self.logger.write([f"Saving figure {filename} to {fig_dir}"])
        save_args = {
                #'dpi'   : 80,
                'orientation' : 'landscape',
                }
        fpath = os.path.join(fig_dir, filename)
        figure.savefig(fpath, **save_args)

    def make_hr_av_plot(self, experiment, data_dic):
        fig_dir = self.fig_dir
        # Make hour averaged plots
        self.logger.write([f"Making hour-averaged longitudinal profile plots for experiment {experiment}",
                           f"Saving figure to {fig_dir}"])

        surface = data_dic['surface']
        bed = data_dic['bed']
        depth = surface - bed

        # Plot surface, bed, and depth values
        ax = None
        data_color_pairs = ((surface, "br"),
                            (bed, "gb"),
                            (depth, "rg"),
                           )
        for profiles, color in data_color_pairs:
            hour_av = profiles.mean(axis=0, level='Exp time')
            ax = self.plot_profiles(hour_av, ax=ax, color=color)

        # Make an overall legend title that acts like legend column titles
        leg_words = ['Water', 'Surface', 'Bed', ' ', 'Water', 'Depth']
        leg_word_order = [0, 2, 4, 1, 3, 5]
        leg_title_row = '{{{}: <10}}'*3
        leg_blank = leg_title_row + '\n' + leg_title_row + ' '*20
        leg_blank = leg_blank.format(*leg_word_order)
        title = leg_blank.format(*leg_words)

        # Ignore current legend labels. Make new set where first two columns don't 
        # have labels and last one has the shared labels.
        handles, labels = ax.get_legend_handles_labels()
        labels = ['']*16 \
                + ['Rising {}L/s'.format(Q) for Q in [50, 62, 75, 87, 100]] \
                + ['Falling {}L/s'.format(Q) for Q in [87, 75, 62]]

        # Format plot labels and ticks
        plt.title("Experiment {} hour-averaged profiles (Flow towards left)".format(experiment))
        ax.set_xlabel("Distance from flume exit (m)")
        ax.set_ylabel("Height (cm)")
        ax.set_xlim((2,8))
        ax.set_ylim((0,35))
        plt.tick_params(axis='y', right=True, labelright=True)
        plt.tick_params(axis='x', top=True)#, labelright=True)

        # Format the plot layout
        figure = plt.gcf()
        figure.set_dpi(80)
        figure.set_size_inches(24, 13.5)
        ax.legend(handles, labels, ncol=3, title=title,
                loc='upper right', bbox_to_anchor=(0.99, 0.16), borderaxespad=0.0)
        plt.subplots_adjust(left=0.04, bottom = 0.05, right=0.98, top=0.95) # Reduce margins

        #plt.show()

        # Save the figure
        save_args = {
                #'dpi'   : 80,
                'orientation' : 'landscape',
                }
        fname = "profiles-hr-av-{}.png".format(experiment)
        fpath = os.path.join(fig_dir, fname)
        figure.savefig(fpath, **save_args)


    def plot_profiles(self, profiles, color, ax=None):
        n_profiles = profiles.shape[0]

        # Make some color scales
        mid_scale = self.get_scale(0.20, .80, n_profiles)
        light_scale = self.get_scale(0.15, 1, n_profiles)
        dark_scale = self.get_scale(0, 0.85, n_profiles)
        zeros = np.zeros(n_profiles)

        scales = {
                "gr"  : np.stack((mid_scale, mid_scale[::-1], zeros), axis=1),
                "bg" : np.stack((zeros, mid_scale, mid_scale[::-1]), axis=1),
                "rb" : np.stack((mid_scale[::-1], zeros, mid_scale), axis=1),
                "red" : np.stack((light_scale, zeros, zeros), axis=1),
                "blue" : np.stack((zeros, zeros, light_scale), axis=1),
                "grey" : np.stack((dark_scale, dark_scale, dark_scale), axis=1),
                }
        #styles = [':', '-.', '--', '-', ':', '-', '--', '-.']
        styles = (-(-n_profiles//4) * [':', '-', '-.', '--'])[0:n_profiles]
        #markers = ['o', 'v', '^', 's', '*', '+', 'x', 'D']

        t_profiles = profiles.transpose()
        if ax is None:
            return t_profiles.plot(color=scales[color], style=styles)
        else:
            return t_profiles.plot(ax=ax, color=scales[color], style=styles)

    def get_scale(self, min, max, n_colors):
        return (max-min) * np.linspace(0, 1, n_colors) + min




if __name__ == "__main__":
    crawler = PlottingCrawler()#no_log=True)
    exp_root = '/home/alex/ubc/research/feed-timing/data/'
    crawler.set_root(f"{exp_root}data-links/manual-data/pickles")
    crawler.set_fig_dir(f"{exp_root}data-links/manual-data/figures")
    #crawler.setup_loader()
    crawler.run('plot-slope-profiles')
    crawler.end()
