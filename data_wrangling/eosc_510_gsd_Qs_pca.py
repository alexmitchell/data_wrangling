#!/usr/bin/env python3

from os.path import join as pjoin
from os.path import isfile as os_isfile
import numpy as np
import pandas as pd

from scipy.stats import linregress as stats_linregress
from scipy.cluster import hierarchy as scipy_hierarchy

from sklearn import decomposition as skl_decomposition
from sklearn.model_selection import train_test_split as skl_train_test_split
import sklearn.cluster as skl_cluster

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

from helpyr import logger
from helpyr import data_loading
from helpyr import kwarg_checker
from helpyr import figure_helpyr

from data_wrangling.omnipickle_manager import OmnipickleManager
import data_wrangling.global_settings as settings

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


## Saved PCA Data will have the following format
# pca_dict = {
#         'evectors' : evectors or {exp_code : evectors},
#         'evalues' : evalues or {exp_code : evalues},
#         'PCs' : PCs or {exp_code : PCs},
#         'keep_modes' : keep_modes or {exp_code : keep_modes}, # suggested k
#         'is_concat' : is_concat,
# }


class EOSC510Project:

    def __init__(self, debug_mode=False):
        
        # Start up logger
        self.log_filepath = f"{settings.log_dir}/eosc510_project.txt"
        self.logger = logger.Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("EOSC 510 project")

        # Reload omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()
        
        # Initialize variables
        self.pca_data = None
        self.rolling_missing_tolerance = 0.8
        self.p1_links = {} #{exp_code : p1_agglom_links <- note: never changes
        
        # Setup plotting variables
        self.stable_subplots = figure_helpyr.StableSubplots()
        self.figure_saver = figure_helpyr.FigureSaver(
                figure_extension='png',
                logger = self.logger,
                debug_mode = debug_mode,
                figure_root_dir = settings.figure_destination,
                figure_sub_dir = 'eosc510',
                )
        self.save_figure = self.figure_saver.save_figure # for older uses

    
    def load_raw_data(self):
        # Reload lighttable (Qs) data from files into omnipickle
        reload_kwargs = {
                'add_feed' : True,
                #'cols_to_keep' : 
                }
        self.omnimanager.reload_Qs_data(**reload_kwargs)

        # Load all Qs data into a local variable
        self.Qs_data = {}
        for exp_code in self.omnimanager.experiments.keys():
            experiment = self.omnimanager.experiments[exp_code]
            self.Qs_data[exp_code] = experiment.accumulated_data

        self.clean_Qs_data()

        # Load surface gsd data
        #self.omnimanager.reload_gsd_data()

    def clean_Qs_data(self):
        self.logger.write_blankline()
        self.outliers = {}
        self.prominence = {}
        self.p_window = 5 # Note, this includes the center
        self.p_tolerance = 250
        self.logger.write(f"Note: Outliers are currently calculated as " +\
                f"any point +-{self.p_tolerance} g/s from the average " +\
                f"of the neighboring {self.p_window-1} nodes (prominence)")
        for exp_code in self.Qs_data.keys():
        #for exp_code in ['1A']:
            self.logger.write(f"Cleaning {exp_code}")
            experiment = self.omnimanager.experiments[exp_code]
            raw_pd_data = self.Qs_data[exp_code]
            bedload_all_pd = raw_pd_data['Bedload all']
            exp_time_hrs = raw_pd_data['exp_time_hrs'].values
            
            # Find prominence
            # Measure of how different the target node is from the average of 
            # the adjacent cells. I can't find a rolling window type that 
            # excludes the center node (kinda like image processing kernels 
            # can) so I can average only the adjacent nodes, hence the more 
            # complicated prominence equation.
            p_roller = bedload_all_pd.rolling(
                    self.p_window, min_periods=1, center=True)
            p_sum = p_roller.sum()
            prominence = \
                    (1 + 1/(self.p_window-1)) * bedload_all_pd - \
                    p_sum / (self.p_window - 1)

            # Identify outliers
            very_pos = prominence > self.p_tolerance
            very_neg = prominence < -self.p_tolerance
            is_outlier = very_pos | very_neg
            outliers_idx = np.where(is_outlier)[0]

            # Set outliers to NaN
            #outliers = bedload_all_pd[is_outlier].copy()
            #bedload_all_pd.loc[is_outlier] = np.NaN
            #
            ## Some debugging plots
            #plt.figure()
            #plt.scatter(exp_time_hrs, bedload_all_pd, color='b')
            #plt.scatter(exp_time_hrs[is_outlier], outliers, color='r')
            #plt.title(f"bedload all {exp_code}")

            #plt.figure()
            #plt.scatter(exp_time_hrs, prominence)
            #plt.hlines([-250, 250], 0, 8)
            #plt.title(f"prominence {exp_code}")
            #plt.show()
            #assert(False)
            #if False:
            
            # Remove outliers
            # Note this changes self.Qs_data
            bedload_columns = [
                    'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                    'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                    'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                    'Bedload 22', 'Bedload 32', 'Bedload 45',
                    ]
            #outliers_bedload = bedload_all_pd.iloc[outliers_idx].copy().values

            # Save data
            self.prominence[exp_code] = prominence.copy()
            self.prominence[exp_code].index = exp_time_hrs
            outliers = bedload_all_pd[is_outlier].copy()
            outliers.index = exp_time_hrs[is_outlier]
            #outliers.reset_index(drop=True, inplace=True)
            self.outliers[exp_code] = outliers
            col_idx = np.where(np.in1d(raw_pd_data.columns, bedload_columns))[0]
            raw_pd_data.iloc[outliers_idx, col_idx] = np.nan

    def _plot_outliers_prominence(self, save_plots=True):
        # Plots the outliers and prominance for each experiment. Assumes 
        # clean_Qs_data has been called and that it uses the prominence method

        self.logger.write_blankline()
        self.logger.write("Plotting outliers")

        fig, axs2D = plt.subplots(nrows=4, ncols=4, sharex=True, figsize=(12,8),
                gridspec_kw={'height_ratios' : [2,1.5,2,1.5],},
                )
        axs = axs2D.flatten()

        exp_codes = sorted(self.Qs_data.keys())
        p_tolerance = self.p_tolerance
        d_min, d_max = [0, 1]
        p_min, p_max = [-2*p_tolerance, 2*p_tolerance]

        data_color = '#1f77b4' # faded blue
        outlier_color = 'r'
        for ax_id, exp_code in enumerate(exp_codes):
            # Get the right data and prominence axes
            ax_id += 4 if ax_id > 3 else 0
            d_ax, p_ax = axs[ax_id], axs[ax_id + 4]

            # Get the data
            raw_pd_data = self.Qs_data[exp_code]
            outliers = self.outliers[exp_code]
            prominence = self.prominence[exp_code]

            # Plot the bedload data that will be kept
            bedload_all = raw_pd_data['Bedload all'].values
            exp_time_hrs = raw_pd_data['exp_time_hrs'].values
            d_ax.scatter(exp_time_hrs, bedload_all, label='Keep',
                    marker='.', c=data_color)

            # Plot the outlier data that will not be kept
            outlier_bedload = outliers.values
            outlier_hrs = outliers.index
            d_ax.scatter(outlier_hrs, outlier_bedload, label='Outlier',
                    marker='o', color=outlier_color, facecolors='none')

            # Plot the prominence values
            is_outlier = prominence.index.isin(outlier_hrs)
            is_keeper = np.logical_not(is_outlier)
            p_ax.scatter(prominence.index[is_keeper], 
                    prominence.values[is_keeper], label='Keep',
                    marker='.', c=data_color)
            p_ax.scatter(prominence.index[is_outlier], 
                    prominence.values[is_outlier], label='Outlier', 
                    marker='o', color=outlier_color, facecolors='none')
            p_ax.hlines([-p_tolerance, p_tolerance], 0, 8)

            # Get the data and prominence plot limits
            d_ylim = d_ax.get_ylim()
            d_min = d_min if d_min < d_ylim[0] else d_ylim[0]
            d_max = d_max if d_max > d_ylim[1] else d_ylim[1]
            p_ylim = p_ax.get_ylim()
            p_min = p_min if p_min < p_ylim[0] else p_ylim[0]
            p_max = p_max if p_max > p_ylim[1] else p_ylim[1]

        for ax_id, exp_code in enumerate(exp_codes):
            ax_id += 4 if ax_id > 3 else 0
            
            # Fix the y axis limits so like plots share the y limits
            axs[ax_id].set_ylim((d_min, d_max))
            axs[ax_id + 4].set_ylim((p_min, p_max))
            axs[ax_id + 4].yaxis.set_minor_locator(AutoMinorLocator())

            # Format ticks and add exp_code text
            for ax in (axs[ax_id], axs[ax_id+4]):
                ax.tick_params( bottom=True, top=True, left=True, right=True,
                        which='both', labelbottom=False, labelleft=False,
                        )
                ax.text(0.95, 0.95, exp_code, va='top', ha='right',
                        bbox={
                            'fill' : False,
                            'visible' : False,
                            'pad' : 0},
                        transform=ax.transAxes)

        # Label fontsize
        label_fontsize = 25

        # Format left column
        for row in [0,1,2,3]:
            label_rotation = 'vertical'
            if row in [0,2]:
                axs2D[row,0].tick_params(labelleft=True,)
                #axs2D[row,0].set_ylabel(f"Bedload (g/s)")
                # Set common y label
                fig.text(0.001, 0.625, f"Bedload (g/s)", va='center', 
                        usetex=True, fontsize=label_fontsize, 
                        rotation=label_rotation)
            elif row in [1,3]:
                axs2D[row,3].tick_params(labelright=True,)
                #axs2D[row,3].set_ylabel(f"Prominence (g/s)")
                # Set common y label
                fig.text(0.965, 0.375, f"Prominence (g/s)", va='center', 
                        usetex=True, fontsize=label_fontsize, 
                        rotation=label_rotation)

        # Format bottom row
        for col in [0,1,2,3]:
            axs2D[3,col].tick_params(labelbottom=True)
            #axs2D[3,col].set_xlabel(f"Experiment time (hrs)")
        # Set common x label
        fig.text(0.5, 0.01, f"Experiment time (hrs)", ha='center', usetex=True,
                    fontsize=label_fontsize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95, left=0.065, bottom=0.075, right=0.925)
        
        # Add legend to one of the subplots
        #axs2D[0,0].legend(loc = 'upper right', bbox_to_anchor = (0.99, 0.85),
        #        handletextpad = 0.1, borderpad = 0.1)
        axs2D[0,3].legend(loc = 'upper left', bbox_to_anchor = (1.01, 0.85),
                handletextpad = 0.1, borderpad = 0.1)

        if save_plots:
            self.save_figure(fig_name_parts=[f"outliers_prominence_all_exp"])
        plt.show()

    def _plot_outliers_mean(self, exp_code, time, bedload, o_time, o_bedload, threshold, std_scale, rrmean, rrstd, crmean, crstd):
        ## This function is outdated.
        plt.figure()
        plt.title(f"{exp_code} Bedload all")
        plt.scatter(time, bedload,
                label='Keep')
        plt.scatter(o_time, o_bedload,
                label='Remove')

        plt.xlim((-1, 10))
        xlim = plt.xlim()
        plt.ylim((0, 900))
        plt.yticks(np.linspace(0, 900, 10))

        plt.hlines([threshold], xlim[0], xlim[1],
                linewidth=0.5, label='Threshold')
        plt.plot(time, rrmean, c='r',
                linewidth=0.5, label='Old moving mean')
        plt.plot(time, rrmean + std_scale * rrstd, c='orange',
                linewidth=0.5, label=rf"Old {std_scale}$\sigma$")
        #plt.plot(time, rrmean - std_scale * rrstd, c='b', linewidth=0.5)

        plt.plot(time, crmean, c='g',
                linewidth=0.5, label='New moving mean')
        plt.plot(time, crmean + std_scale * crstd, c='c',
                linewidth=0.5, label=rf"New {std_scale}$\sigma$")
        #plt.plot(time, crmean - std_scale * crstd, c='b', linewidth=0.5)
        plt.legend()

        if save_plots:
            self.save_figure(fig_name_parts=[f"outliers_{exp_code}"])


    def analyze_pca_individually(self, **kwargs):
        """ Perform PCA on each experiment separately. """

        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        save_output = check('save_output', default=False)
        make_plots = check('make_plots', default=True)

        self.logger.write_blankline()
        pca_output = {
                'evectors' : {},
                'evalues' : {},
                'PCs' : {},
                'keep_modes' : {},
                'is_concat' : False,
                }
        for exp_code in self.Qs_data.keys():
            output = self.logger.run_indented_function(
                    self._analyze_pca_experiment, kwargs={
                        'exp_code' : exp_code, 'save_output' : save_output,
                        'make_plots' : make_plots,
                        'save_plots' : False,
                        },
                    before_msg=f"Analyzing {exp_code}",
                    after_msg="")
            if save_output:
                pca_output['evectors'][exp_code] = output['evectors']
                pca_output['evalues'][exp_code] = output['evalues']
                pca_output['PCs'][exp_code] = output['PCs']
                pca_output['keep_modes'][exp_code] = output['keep_modes']
        if save_output:
            file_name = "pca-output_individual-data"
            self.save_pca_output(file_name, pca_output)

    def analyze_pca_all(self, **kwargs):
        """ Perform PCA on all experiments combined. """

        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        save_output = check('save_output', default=False)
        make_plots = check('make_plots', default=True)
        save_plots = check('save_plots', default=True)

        self.logger.write_blankline()
        frames = []
        #frames = self.Qs_data.values()
        for i, exp_code in enumerate(sorted(self.Qs_data.keys())):
            data = self.Qs_data[exp_code]
            data['exp_time_hrs'] += 8*i
            frames.append(data)

        all_data = pd.concat(frames)

        self.logger.run_indented_function(
                self._analyze_pca_all, kwargs={'raw_pd_data' : all_data,
                    'save_output' : save_output, 'make_plots' : make_plots,
                    'save_plots' : save_plots},
                before_msg=f"Analyzing all data",
                after_msg="")

    def _analyze_pca_experiment(self, exp_code, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        save_output = check('save_output', default=False)
        make_plots = check('make_plots', default=True)
        save_plots = check('save_plots', default=False)

        experiment = self.omnimanager.experiments[exp_code]
        raw_pd_data = self.Qs_data[exp_code]
        fig_name_parts = [exp_code]

        extra_cols = self.prep_raw_data(raw_pd_data)
        exp_time_hrs = extra_cols['exp_time_hrs']
        grain_sizes = extra_cols['grain_sizes']
        bedload_all = extra_cols['bedload_all']

        pca_output = self.do_pca(raw_pd_data)
        model = pca_output['model']
        PCs = pca_output['PCs']
        evectors = pca_output['evectors']
        evalues = pca_output['evalues']
        data_stddev = pca_output['data_stddev']
        data_mean = pca_output['data_mean']

        pca_codes = pca_output.pop('pca_codes')
        fig_name_parts += pca_codes

        self._plot_eigenvalues(
                exp_code = exp_code, 
                evalues = evalues, 
                fig_name_parts = fig_name_parts, 
                save_plots=save_plots)
        # Record the number of modes that sum to over 90% or 95% of variance
        # Based on the eigenvalues plots
        self.keep_modes_95 = {
                '1A' : 8,
                '1B' : 7,
                '2A' : 8,
                '2B' : 8,
                '3A' : 7,
                '3B' : 8,
                '4A' : 8,
                '5A' : 8,
                }
        self.keep_modes_90 = {
                '1A' : 7,
                '1B' : 6,
                '2A' : 6,
                '2B' : 7,
                '3A' : 6,
                '3B' : 6,
                '4A' : 6,
                '5A' : 7,
                }
        var_threshold = 90
        if var_threshold == 95:
            keep_modes_dict = self.keep_modes_95
        elif var_threshold == 90:
            keep_modes_dict = self.keep_modes_90

        if exp_code in keep_modes_dict:
            keep_modes = keep_modes_dict[exp_code]
            explained = np.sum(evalues[0:keep_modes])
            self.logger.write(f"Keeping {keep_modes} modes ({explained:0.2%} of variance)")
            fig_name_parts_k = [f"{var_threshold}p-k{keep_modes}"] + fig_name_parts
        else:
            plt.show()
            return

        if make_plots:
            # Plot PCA variables
            self._plot_PCs(
                    exp_code = exp_code, 
                    exp_time_hrs = exp_time_hrs, 
                    PCs = PCs, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    ylim=(-25, 50),
                    save_plots=save_plots)
            self._plot_PCs(
                    exp_code = exp_code, 
                    exp_time_hrs = exp_time_hrs, 
                    PCs = PCs, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    save_plots=save_plots)
            self._plot_PCs_comparisons(
                    exp_code = exp_code, 
                    PCs = PCs, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    save_plots=save_plots)
            self._plot_eigenvectors(
                    exp_code = exp_code, 
                    grain_sizes = grain_sizes, 
                    evectors = evectors, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    save_plots=save_plots)
        
        # Reconstruct 
        recon_k = PCs[:, 0:keep_modes] @ evectors[0:keep_modes, :]
        recon_1 = np.outer(PCs[:, 0], evectors[0, :])

        # Rescale
        rescale_fu = lambda recon_data: recon_data * data_stddev + data_mean
        bedload_all_fu = lambda data: np.sum(data, axis=1)
        recon_bedload_all_k = bedload_all_fu(rescale_fu(recon_k))
        recon_bedload_all_1 = bedload_all_fu(rescale_fu(recon_1))

        if make_plots:
            # Plot reconstructions
            recon_name = "bedload all"
            recon_fig_name = recon_name.replace(' ', '-')
            fig_name_parts_k = [recon_fig_name] + fig_name_parts_k

            self._plot_single_reconstruction(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = keep_modes, 
                    time = exp_time_hrs, 
                    original = bedload_all, 
                    recon = recon_bedload_all_k,
                    fig_name_parts=fig_name_parts_k, 
                    save_plots=save_plots)
            self._plot_reconstruction_fit_single(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = keep_modes, 
                    original = bedload_all, 
                    recon = recon_bedload_all_k,
                    fig_name_parts=fig_name_parts_k, 
                    save_plots=save_plots)

            fig_name_parts_1 = [recon_fig_name, f"k1"] + fig_name_parts
            self._plot_single_reconstruction(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = 1, 
                    time = exp_time_hrs, 
                    original = bedload_all, 
                    recon = recon_bedload_all_1,
                    fig_name_parts=fig_name_parts_1, 
                    save_plots=save_plots)
            self._plot_reconstruction_fit_single(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = 1, 
                    original = bedload_all, 
                    recon = recon_bedload_all_1,
                    fig_name_parts=fig_name_parts_1, 
                    save_plots=save_plots)

            #plt.show()
            plt.close('all')

        if save_output:
            # Save the pca output for other use.
            pca_output = {
                    'evectors' : evectors,
                    'evalues' : evalues,
                    'PCs' : PCs,
                    'keep_modes' : keep_modes,
            }
            return pca_output
        else:
            return None

    def _analyze_pca_all(self, raw_pd_data, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        save_output = check('save_output', default=False)
        make_plots = check('make_plots', default=True)
        save_plots = check('save_plots', default=True)
        exp_code = 'All data' # Hacky solution
        fig_name_parts = ['all-Qs']

        extra_cols = self.prep_raw_data(raw_pd_data)
        exp_time_hrs = extra_cols['exp_time_hrs']
        grain_sizes = extra_cols['grain_sizes']
        bedload_all = extra_cols['bedload_all']

        pca_output = self.do_pca(raw_pd_data)
        model = pca_output['model']
        PCs = pca_output['PCs']
        evectors = pca_output['evectors']
        evalues = pca_output['evalues']
        data_stddev = pca_output['data_stddev']
        data_mean = pca_output['data_mean']

        pca_codes = pca_output.pop('pca_codes')
        fig_name_parts += pca_codes

        if make_plots:
            self._plot_eigenvalues(
                    exp_code = exp_code, 
                    evalues = evalues, 
                    fig_name_parts = fig_name_parts,
                    save_plots=save_plots)
        # Record the number of modes that sum to over 90% or 95% of variance
        # Based on the eigenvalues plots

        self.keep_modes_all_k3 = 3
        self.keep_modes_all_90 = 7
        self.keep_modes_all_95 = 8
        var_threshold = 3
        if var_threshold == 95:
            keep_modes = self.keep_modes_all_95
        elif var_threshold == 90:
            keep_modes = self.keep_modes_all_90
        elif var_threshold == 3:
            keep_modes = self.keep_modes_all_k3

        if keep_modes is not None:
            explained = np.sum(evalues[0:keep_modes])
            self.logger.write(f"Keeping {keep_modes} modes ({explained:0.2%} of variance)")
            fig_name_parts_k = [f"{var_threshold}p-k{keep_modes}"] + fig_name_parts
        else:
            if make_plots:
                plt.show()
            else:
                self.logger.write("No modes specified. Aborting.")
            return

        if make_plots:
            # Plot PCA variables
            self._plot_PCs(
                    exp_code = exp_code, 
                    exp_time_hrs = exp_time_hrs, 
                    PCs = PCs, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    ylim=(-25, 50), 
                    save_plots=save_plots)
            #self._plot_PCs_comparisons(
            #        exp_code = exp_code, 
            #        PCs = PCs, 
            #        keep_modes = keep_modes, 
            #        fig_name_parts = fig_name_parts_k, 
            #        save_plots=save_plots)
            self._plot_eigenvectors(
                    exp_code = exp_code, 
                    grain_sizes = grain_sizes, 
                    evectors = evectors, 
                    keep_modes = keep_modes, 
                    fig_name_parts = fig_name_parts_k, 
                    save_plots=save_plots)
        
        # Reconstruct 
        recon_k = PCs[:, 0:keep_modes] @ evectors[0:keep_modes, :]
        recon_1 = np.outer(PCs[:, 0], evectors[0, :])

        # Rescale
        rescale_fu = lambda recon_data: recon_data * data_stddev + data_mean
        bedload_all_fu = lambda data: np.sum(data, axis=1)
        recon_bedload_all_k = bedload_all_fu(rescale_fu(recon_k))
        recon_bedload_all_1 = bedload_all_fu(rescale_fu(recon_1))

        if make_plots:
            # Plot reconstructions
            recon_name = "bedload all"
            recon_fig_name = recon_name.replace(' ', '-')
            fig_name_parts_k = [recon_fig_name] + fig_name_parts_k

            self._plot_single_reconstruction(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = keep_modes, 
                    time = exp_time_hrs, 
                    original = bedload_all, 
                    recon = recon_bedload_all_k,
                    fig_name_parts=fig_name_parts_k, 
                    save_plots=save_plots)
            self._plot_reconstruction_fit_single(
                    exp_code = exp_code, 
                    name = recon_name, 
                    k = keep_modes, 
                    original = bedload_all, 
                    recon = recon_bedload_all_k,
                    fig_name_parts=fig_name_parts_k, 
                    save_plots=save_plots)

            fig_name_parts_1 = [recon_fig_name, f"k1"] + fig_name_parts
            #self._plot_single_reconstruction(
            #        exp_code = exp_code, 
            #        name = recon_name, 
            #        k = 1, 
            #        time = exp_time_hrs, 
            #        original = bedload_all, 
            #        recon = recon_bedload_all_1,
            #        fig_name_parts=fig_name_parts_1, 
            #        save_plots=save_plots)
            #self._plot_reconstruction_fit_single(
            #        exp_code = exp_code, 
            #        name = recon_name, 
            #        k = 1, 
            #        original = bedload_all, 
            #        recon = recon_bedload_all_1,
            #        fig_name_parts=fig_name_parts_1, 
            #        save_plots=save_plots)

            plt.show()
            plt.close('all')

            if save_output:
                # Save the pca output for other use.
                pca_output = {
                        'evectors' : evectors,
                        'evalues' : evalues,
                        'PCs' : PCs,
                        'keep_modes' : keep_modes,
                        'is_concat' : True,
                }
                file_name = "pca-output_all-data"
                self.save_pca_output(file_name, pca_output)


    def do_pca(self, raw_gsd_data, normalize=False, standardize=True):
        # raw_gsd_data is dataframe of only grain size classes over time.
        # normalize is whether distributions should be normalized by distr. sum
        # standardize is whether GS classes should be standardized over time

        # Get numpy array for raw data
        raw_data = raw_gsd_data.values
        #raw_data = np.random.random(raw_data.shape)

        # Normalize distributions by mass moved
        # Want to compare shapes, not magnitudes of distributions ?
        if normalize:
            norm_data = raw_data / np.sum(raw_data, axis=1)[:, None]
        else:
            norm_data = raw_data

        # Get mean and std
        data_stddev = np.nanstd(norm_data, axis=0)
        data_mean = np.nanmean(norm_data, axis=0)

        # Standardize the data
        # Temporarily convert data_stddev == 0 to 1 for the division
        if standardize:
            all_zeros = [data_stddev == 0]
            data_stddev[tuple(all_zeros)] = 1
            std_data = (norm_data - data_mean) / data_stddev
            data_stddev[tuple(all_zeros)] = 0
        else:
            std_data = norm_data

        # Split into training and testing data
        #train_data, test_data = skl_train_test_split(std_data)
        train_data = std_data

        # Perform the PCA
        model = skl_decomposition.PCA()
        PCs = model.fit_transform(train_data)
        evectors = model.components_ # MODES ARE IN ROWS!!!
        evalues = model.explained_variance_ratio_

        #tested_output = model.transform(test_data)
        #pca_outputs[exp_code] = model
        #pca_inputs[exp_code] = std_data

        self.logger.write([
            f"Shape of input data {std_data.shape}",
            f"Evectors shape {evectors.shape}",
            f"Evalues shape {evalues.shape}", 
            f"PCs shape {PCs.shape}"]
            )

        std_str = 'std' if standardize else 'nostd'
        norm_str = 'norm-distr' if normalize else 'nonnorm-distr'
        pca_codes = [std_str, norm_str]

        pca_output = {
                'model' : model,
                'PCs' : PCs,
                'evectors' : evectors,
                'evalues' : evalues,
                'data_stddev' : data_stddev,
                'data_mean' : data_mean,
                'pca_codes' : pca_codes,
                }
        return pca_output

    def prep_raw_data(self, raw_pd_data):
        # Drop Nan rows
        raw_pd_data.dropna(axis=0, inplace=True)

        # Pull out columns of interest
        extra_cols = {
                'bedload_all' : raw_pd_data.pop('Bedload all').values,
                'exp_time' : raw_pd_data.pop('exp_time').values,
                'exp_time_hrs' : raw_pd_data.pop('exp_time_hrs').values,
                'discharge' : raw_pd_data.pop('discharge').values,
                'feed' : raw_pd_data.pop('feed').values,
                }

        # Drop columns we don't care about
        drop_cols = ['timestamp', 'missing ratio', 'vel', 'sd vel',
                'number vel', 'Count all', 'Count 0.5', 'Count 0.71',
                'Count 1', 'Count 1.4', 'Count 2', 'Count 2.8', 'Count 4',
                'Count 5.6', 'Count 8', 'Count 11.2', 'Count 16',
                'Count 22', 'Count 32', 'Count 45', 'D10', 'D16', 'D25',
                'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax',]
        raw_pd_data.drop(columns=drop_cols, inplace=True)

        # Get grain sizes 
        extra_cols['grain_sizes'] = np.array([ float(s.split()[1]) 
            for s in raw_pd_data.columns])
        return extra_cols

    def save_pca_output(self, name, pca_output):
        source_dir = settings.pca_pickle_dir
        destination_dir = source_dir
        loader = data_loading.DataLoader(source_dir, destination_dir, 
                self.logger)

        ## Saved PCA Data will have the following format
        # pca_dict = {
        #         'evectors' : evectors or {exp_code : evectors},
        #         'evalues' : evalues or {exp_code : evalues},
        #         'PCs' : PCs or {exp_code : PCs},
        #         'keep_modes' : keep_modes or {exp_code : keep_modes}, # suggested k
        #         'is_concat' : is_conca
        # }
        
        loader.produce_pickles({name : pca_output}, overwrite=True)


    ## Plotting functions
    def _plot_reconstruction_fit_single(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        name = check('name', required=True)
        k = check('k', required=True)
        original = check('original', required=True)
        recon = check('recon', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        save_plots = check('save_plots', default=True)

        # Do linear regression to get r2
        m, b, r, p, stderr = stats_linregress(recon, original)

        # Plot it
        plt.figure()
        plt.title(rf"{exp_code} Calibration plot for {name} (k={k})")
        plt.scatter(recon, original)
        plt.xlabel("Reconstruction")
        plt.ylabel("Observed")
        plt.gca().set_aspect('equal')

        # Get limit info
        xlim = plt.xlim()
        ylim = plt.ylim()
        limits = list(zip(xlim, ylim))
        min = np.max(limits[0])
        max = np.min(limits[1])

        # Plot 1:1 line
        plt.plot([min, max], [min, max], c='k', label='1 to 1', linewidth=1)

        # Plot regression line
        x = np.linspace(min, max, 10)
        plt.plot(x, m*np.array(x) + b, c='r',
                label=rf"Regression $r^2$={r**2:0.3f}", 
                linestyle='--', linewidth=1)

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.legend()

        if save_plots:
            self.save_figure(fig_name_parts=[f"recon_fit"] + fig_name_parts)

    def _plot_single_reconstruction(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        name = check('name', required=True)
        k = check('k', required=True)
        time = check('time', required=True)
        original = check('original', required=True)
        recon = check('recon', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        save_plots = check('save_plots', default=True)

        # Plot it
        plt.figure()
        plt.title(f"{exp_code} Reconstructed {name} (k={k})")
        plt.plot(time, original, label='Original', c='b', linewidth=0.25)
        plt.plot(time, recon, label='Reconstructed', c='orange', linewidth=0.25)
        
        # Find moving mean
        window = 400
        tolerance = self.rolling_missing_tolerance
        orig_roller = pd.DataFrame(original).rolling(window,
                min_periods=int(window * tolerance), center=True)
        orig_mean = orig_roller.mean()
        recon_roller = pd.DataFrame(recon).rolling(window,
                min_periods=int(window * tolerance), center=True)
        recon_mean = recon_roller.mean()

        plt.plot(time, orig_mean, label='Original mean', c='g', linewidth=0.5)
        plt.plot(time, recon_mean, label='Reconstructed mean', c='r',
                linewidth=0.5)

        plt.legend()

        if save_plots:
            self.save_figure(fig_name_parts=[f"recon"] + fig_name_parts)

    def _plot_eigenvectors(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        grain_sizes = check('grain_sizes', required=True)
        evectors = check('evectors', required=True)
        keep_modes = check('keep_modes', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        save_plots = check('save_plots', default=True)

        # plot each eigenvector
        # sklearn puts evector modes in rows
        fig, axs, axs_shape= self._get_squarish_subplots_grid(keep_modes)
        nrows, ncols = axs_shape
        plt.suptitle(f"{exp_code} Eigenvectors")

        for k, ax in zip(np.arange(axs.size), axs.flatten()):
            if k >= keep_modes:
                break
            ax.set_title(f"Mode {k+1}")
            evector = evectors[k,:]
            #ax.scatter(grain_sizes, evector * stddev + mean)
            ax.scatter(grain_sizes, evector)
            ax.set_xscale('log')
            #ax.set_ylabel(f"mass?")
            ax.axhline()
            ax.grid(which='both')

        # Label the bottom row of the grid
        bottom_row = []
        if nrows >= 1 and ncols > 1:
            bottom_row = axs[-1, :]
        elif nrows >= 1 and ncols == 1:
            bottom_row = [axs[-1]]
        for ax in bottom_row:
            ax.set_xlabel("Grain size (mm)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            self.save_figure(fig_name_parts=[f"evectors"] + fig_name_parts)

    def _plot_PCs_comparisons(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        PCs = check('PCs', required=True)
        keep_modes = check('keep_modes', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        save_plots = check('save_plots', default=True)

        # plot comparisons between PCs
        fig, axs, axs_shape = self._get_squarish_subplots_grid(keep_modes-1)
        nrows, ncols = axs_shape
        plt.suptitle(f"{exp_code} PCs comparisons")

        in_p_subplot = 2
        fig.set_figheight(in_p_subplot * nrows)
        fig.set_figwidth(in_p_subplot * ncols)
        for k, ax in zip(np.arange(axs.size)+1, axs.flatten()):
            if k >= keep_modes:
                ax.axis('off')
                continue
            ax.set_title(f"PCs modes 1 and {k+1}")
            ax.scatter(PCs[:,0], PCs[:, k])
            ax.set_ylabel(f"PC {k+1}")
            ax.grid(which='both')
            ax.set_xlim((-25, 100))
            ax.set_ylim((-25, 100))
            ax.set_aspect('equal')

        # Label the bottom row of the grid
        bottom_row = []
        if nrows >= 1 and ncols > 1:
            bottom_row = axs[-1, :]
        elif nrows >= 1 and ncols == 1:
            bottom_row = [axs[-1]]
        for ax in bottom_row:
            ax.set_xlabel("PC 1")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            self.save_figure(fig_name_parts=[f"PCs_comparisons"] + fig_name_parts)

    def _plot_PCs(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        exp_time_hrs = check('exp_time_hrs', required=True)
        PCs = check('PCs', required=True)
        keep_modes = check('keep_modes', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        ylim = check('ylim', default=None)
        save_plots = check('save_plots', default=True)

        # Plot each PC
        fig, axs = plt.subplots(nrows=keep_modes, sharex=True, sharey=True)
        plt.suptitle(f"{exp_code} PCs for {keep_modes} modes")
        for k, ax in zip(np.arange(keep_modes), axs):
            ax.plot(exp_time_hrs, PCs[:, k])
            ax.set_ylabel(f"PC {k+1}", rotation='horizontal', labelpad=15)
            ax.grid(which='both')
            if ylim is not None:
                ax.set_ylim(ylim)

        axs[-1].set_xlabel(f"Time (hrs)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_plots:
            self.save_figure(fig_name_parts=[f"PCs"] + fig_name_parts)

    def _plot_eigenvalues(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        exp_code = check('exp_code', required=True)
        evalues = check('evalues', required=True)
        fig_name_parts = check('fig_name_parts', required=True)
        save_plots = check('save_plots', default=True)

        fig, axs = plt.subplots(nrows=2)
        x_modes = np.arange(evalues.size)+1
        axs[0].set_title(f"{exp_code} Variance")
        axs[0].plot(x_modes, evalues)
        axs[0].set_ylabel('Variance fraction')
        axs[0].set_ylim((0, 1))
        axs[0].yaxis.set_ticks(np.linspace(0, 1, 11))

        axs[1].set_title(f"{exp_code} Cumulative sum of variance")
        axs[1].plot(x_modes, np.cumsum(evalues))
        #axs[1].axhline(0.75, c='k', linewidth=0.5)
        axs[1].axhline(0.90, c='k', linewidth=0.5)
        axs[1].axhline(0.95, c='k', linewidth=0.5)
        axs[1].set_ylabel('Cumulative variance')
        axs[1].set_ylim((0.5, 1))
        axs[1].yaxis.set_ticks(np.linspace(0.5, 1, 6))
        
        for ax in axs:
            ax.set_xlabel('Mode')
            ax.grid(which='both')
            ax.xaxis.set_ticks(x_modes)

        plt.tight_layout()

        if save_plots:
            self.save_figure(fig_name_parts=[f"evalues"] + fig_name_parts)

    def plot_3d_centroids(self, **kwargs):
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        centroids = check('centroids', required=True)
        ax = check('ax', required=True)
        label = check('label', default='')
        show_legend = check('show_legend', default=False)
        title = check('title', default='')

        assert(centroids.shape[1] == 3)
        kx, ky, kz = [centroids[:,i] for i in range(3)]
        ax.scatter(kx, ky, kz, label=label)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.set_title(title)
        #ax.set_title(f"{exp_code} Agglomerative PCs centroids")
        if show_legend:
            ax.legend()


    ## General plotting support functions
    def _get_squarish_subplots_grid(self, n):
        nrows = int(np.ceil(np.sqrt(n)))
        ncols = int(np.ceil(n / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                sharex=True, sharey=True)
        return fig, axs, (nrows, ncols)


    ## Clustering
    def reload_pca_pickle(self, name):
        self.logger.write_blankline()
        source_dir = settings.pca_pickle_dir
        destination_dir = source_dir
        loader = data_loading.DataLoader(source_dir, destination_dir, 
                self.logger)

        file_name = f"pca-output_{name}-data"
        assert(loader.is_pickled(file_name))
        self.pca_data = loader.load_pickle(file_name)

    def calculate_centroids(self, source_data, clustering_out):
        # Calculate the mean (centroid) and variance of each agglomerative 
        # cluster
        #
        # source_data will be the n x m data to be averaged to get centroids. 
        # eg. original PC points or centroids of another clustering method.
        #
        # cluster_out is a dict of cluster outputs. Must have 'labels' (n x 1) 
        # and 'ids' (k_clusters x 1; basically unique(labels))
        #
        labels = clustering_out['labels']
        ids = clustering_out['ids']

        # Setup output variables
        centroids = np.zeros((ids.size, source_data.shape[1]))
        var = np.zeros_like(centroids)

        for id in ids:
            cluster_points = source_data[labels == id]
            centroids[id, :] = np.mean(cluster_points, axis=0)
            var[id, :] = np.var(cluster_points, axis=0)

        return {'centroids' : centroids,
                'var' : var,
                }

    def run_agglomerative(self, PCs, clusters_out=3, links=None):
        if links is None:
            # Do Agglomerative on PCs for this experiment
            links = scipy_hierarchy.linkage(PCs, method='ward')

        # Get Agglomerative cluster labels
        labels = scipy_hierarchy.fcluster(links, 
                t=clusters_out, criterion='maxclust')
        labels -= 1 # Starts at 1 for some reason...
        ids = np.unique(labels)

        return {'links' : links,
                'labels' : labels,
                'ids' : ids,
                }
        


    def analyze_2pass_stability(self, **kwargs):
        """

        Run two pass agglomerative clustering with multiple n_clusters 
        options for each pass. Main output figure is a large subplots showing 
        resulting centroids with different n_cluster selections.

        """

        # Process kwargs
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        make_plots = check('make_plots', True)
        n_p1_clusters_list = check('n_p1_clusters_list', [30])
        n_p2_clusters_list = check('n_p2_clusters_list', [5])
        show_plots = check('show_plots', True)

        # Get subplot shape
        n_p1 = len(n_p1_clusters_list)
        n_p2 = len(n_p2_clusters_list)
        subplots_shape = (n_p1, n_p2)

        # Run the 2 pass agglomerative clustering function
        for n_p1_clusters in n_p1_clusters_list:
            self._analyze_2pass_stability(
                    make_plots = make_plots,
                    n_p1_clusters = n_p1_clusters,
                    n_p2_clusters_list = n_p2_clusters_list,
                    #p2_get_subplots_axis = add_subplot_fu,
                    show_plots = show_plots,
                    subplots_shape = subplots_shape,
                    )

        # Tell the stable subplots to finish
        self.stable_subplots.finish(
                show = show_plots,
                saving_fu = self.save_figure)

    def _analyze_2pass_stability(self, **kwargs):
        """

        Runs two passes of agglomerative clustering on the data. The first 
        pass has one number of resulting clusters. However, the second pass 
        can iterate over multiple values.
        
        """

        # Process kwargs
        check = kwarg_checker.get_check_kwarg_fu(kwargs)
        make_plots = check('make_plots', True)
        n_p1_clusters = check('n_p1_clusters', 30)
        n_p2_clusters_list = check('n_p2_clusters_list', [5])
        show_plots = check('show_plots', True)
        subplots_shape = check('subplots_shape', None)

        # Write log message
        write_log = self.logger.write
        write_log(f'Running 2 pass clustering ' + \
                        f"({n_p1_clusters} -> {n_p2_clusters_list} clusters)")

        # Get pca data
        assert(self.pca_data is not None)
        evectors_dict = self.pca_data['evectors']
        evalues_dict = self.pca_data['evalues']
        PCs_dict = self.pca_data['PCs']
        keep_modes_dict = self.pca_data['keep_modes']
        is_concat = self.pca_data['is_concat']
        assert(~is_concat)
        exp_codes = sorted(evectors_dict.keys())

        ## Pass 1. Agglomerative clustering on individual experiments
        p1_centroids_dict = {}
        for ax_id, exp_code in enumerate(exp_codes):
            write_log(f'Running pass 1 clustering for {exp_code} ({n_p1_clusters})')
            evectors = evectors_dict[exp_code]
            evalues = evalues_dict[exp_code]
            PCs = PCs_dict[exp_code]
            keep_modes = keep_modes_dict[exp_code]

            keep_modes = 3 # Ignoring the 90% keep modes included in dict

            #k_PCs = PCs[:, :keep_modes]
            k_PCs = PCs[:5000, :keep_modes] # smaller size for debugging

            if exp_code in self.p1_links:
                existing_links = self.p1_links[exp_code]
                p1_agglom_out = self.run_agglomerative(k_PCs, n_p1_clusters,
                        links=existing_links)
            else:
                p1_agglom_out = self.run_agglomerative(k_PCs, n_p1_clusters)
                self.p1_links[exp_code] = p1_agglom_out['links']
            #p1_agglom_out = self.run_agglomerative(k_PCs, n_p1_clusters)
            p1_centroids_out = self.calculate_centroids(k_PCs, p1_agglom_out)

            p1_centroids_dict[exp_code] = p1_centroids_out['centroids']
        
        ## Plot pass 1
        write_log('Plotting Pass 1')
        self._plot_p1_centroids_subplots(
                p1_centroids_dict = p1_centroids_dict, 
                exp_codes = exp_codes, 
                n_p1_clusters = n_p1_clusters)

        ## Pass 2. Agglomerative clustering on pass 1 centroids
        for n_p2_clusters in n_p2_clusters_list:
            write_log(f'Running pass 2 clustering ({n_p2_clusters})')
            
            # Combine the centroids from each experiment. Order doesn't matter.
            p1_all_centroids = np.concatenate(list(p1_centroids_dict.values()))

            #n_p2_clusters = 5
            p2_agglom_out = self.run_agglomerative(
                    p1_all_centroids, n_p2_clusters)
            p2_centroids_out = self.calculate_centroids(
                    p1_all_centroids, p2_agglom_out)
            p2_centroids = p2_centroids_out['centroids']
                
            # Plot pass 2
            write_log('Plotting Pass 2')

            self._plot_p2_centroids_subplot(
                    p2_centroids = p2_centroids, 
                    subplots_shape = subplots_shape, 
                    n_p1_clusters = n_p1_clusters, 
                    n_p2_clusters = n_p2_clusters)

    def _plot_p1_centroids_subplots(self, p1_centroids_dict, exp_codes, n_p1_clusters):
        # Plot the subplots figure for a p1 agglom run

        # Note, there is some dendrogram stuff here as legacy code... Needs to 
        # be moved into it's own function.
        #dengrogram_fig, dendrogram_axs = plt.subplots(4, 2, sharey=True)
        #dendrogram_axs = dendrogram_axs.flatten()

        p1_centroids_fig3d_name = '_'.join([f'agglom', 'PCs-centroids',
            '3d', 'individuals', f'1pass-{n_p1_clusters}'])
        #max_dist = 0
        for exp_code in exp_codes:
            # Get subplot axis
            p1_centroids_ax3d = self.stable_subplots.add_subplot(
                    p1_centroids_fig3d_name,
                    subplots_shape = (2,4),
                    fig_kwargs = {'figsize' : (16,8)},
                    ax_kwargs = {'projection' : '3d'},
                    suptitle = \
                            f"1 Pass agglomerative clustering\n" + \
                            f"({n_p1_clusters} clusters)"
                    )

            # Plot each experiment's centroids in one large subplots figure
            centroids = p1_centroids_dict[exp_code]
            self.plot_3d_centroids(
                    centroids = centroids, 
                    ax = p1_centroids_ax3d,
                    title=f"{exp_code}",
                    )
            #dendrogram = scipy_hierarchy.dendrogram(agglom_links, ax=ax)
            #largest_dist = np.amax(agglom_links[:,2])
            #max_dist = max(max_dist, largest_dist)
            #ax.set_xticklabels([])
            ##ax.set_xlabel("KMeans cluster ID")
            ##ax.set_ylabel("Link distance")
            #ax.set_title(f"{exp_code}")
        #for ax in dendrogram_axs:
        #    ax.set_ylim((0, 1.1 * max_dist))
        #self.save_figure(fig_name_parts=[f'agglom', 'dendrograms', 'individuals'],
        #        figure=dengrogram_fig)

        self.stable_subplots.finish(
                fig_name = p1_centroids_fig3d_name, 
                show = False,
                saving_fu = self.save_figure)

    def _plot_p2_centroids_subplot(self, p2_centroids, subplots_shape, n_p1_clusters, n_p2_clusters):
        # Meant for use within the cluster analysis for loop
        assert(subplots_shape is not None)
        
        p2_centroids_subplots_fig3d_name = '_'.join(['agglom', 'centroids',
                '3d', 'individuals', '2pass-subplots',])
        p2_centroids_subplots_ax3d = self.stable_subplots.add_subplot(
                p2_centroids_subplots_fig3d_name,
                subplots_shape=subplots_shape,
                fig_kwargs={'figsize' : (14, 10)},
                ax_kwargs = {'projection' : '3d'},
                suptitle= f"2 Pass agglomerative clustering\n",
                )
        self.plot_3d_centroids(
                centroids = p2_centroids, 
                ax = p2_centroids_subplots_ax3d,
                title=f"{n_p1_clusters} -> {n_p2_clusters} agglom",
            )


    def analyze_clusters_all(self, make_plots=True):
        assert(self.pca_data is not None)

        evectors = self.pca_data['evectors']
        evalues = self.pca_data['evalues']
        PCs = self.pca_data['PCs']
        keep_modes = self.pca_data['keep_modes']
        is_concat = self.pca_data['is_concat']

        keep_modes = 3
        k_PCs = PCs[:, :keep_modes];

        assert(is_concat)

        kmeans_clusters = 500
        dendrogram_clusters = 3
        n = 3
        dendrogram_fig, dendrogram_axs = plt.subplots(n, n, sharey=True, figsize=(20,11))
        plt.suptitle(f"All data, KMeans {kmeans_clusters}")
        centroids_fig, centroid_axs = plt.subplots(1, 2, sharex=True)
        agglom_centroids_fig3d = plt.figure()
        agglom_centroids_ax3d = agglom_centroids_fig3d.add_subplot(111, projection='3d')
        kmeans_centroids_fig3d = plt.figure()
        kmeans_centroids_ax3d = kmeans_centroids_fig3d.add_subplot(111, projection='3d')
        dendrogram_axs = dendrogram_axs.flatten()
        max_dist = 0
        for axid, ax in enumerate(dendrogram_axs):

            print('Running new clustering')
            # Data set too large for Agglomerative
            #clustering = skl_cluster.AgglomerativeClustering(memory='__pycache__')
            kmeans = skl_cluster.MiniBatchKMeans(n_clusters=kmeans_clusters,
                    init_size=3*kmeans_clusters)
            #clustering = clustering.fit(PCs)
            kmeans_labels = kmeans.fit_predict(k_PCs)
            kmeans_centroids = kmeans.cluster_centers_

            # Do Agglomerative on kmeans centroids
            agglom_links = scipy_hierarchy.linkage(kmeans_centroids, 
                    method='ward')
            #agglom = skl_cluster.AgglomerativeClustering(memory='__pycache__')
            #agglom_labels = agglom.fit_predict(kmeans_centroids)

            # Get Agglomerative cluster labels
            agglom_labels = scipy_hierarchy.fcluster(agglom_links, 
                    t=dendrogram_clusters, criterion='maxclust')
            agglom_labels -= 1 # Starts at 1 for some reason...
            agglom_ids = np.unique(agglom_labels)

            # Get the agglomerative cluster associated with each k_PCs point
            kmeans_agglom_ids = agglom_labels[list(kmeans_labels)]

            # Calculate the mean (centroid) and variance of each agglomerative 
            # cluster
            PCs_agglom_centroids = np.zeros((agglom_ids.size, k_PCs.shape[1]))
            PCs_agglom_var = np.zeros_like(PCs_agglom_centroids)
            kmeans_agglom_centroids = np.zeros((agglom_ids.size, k_PCs.shape[1]))
            kmeans_agglom_var = np.zeros_like(kmeans_agglom_centroids)
            for agglom_id in agglom_ids:
                kmeans_points = kmeans_centroids[agglom_labels == agglom_id]
                cluster_points = k_PCs[kmeans_agglom_ids == agglom_id]
                #cluster_points = kmeans_points
                PCs_agglom_centroids[agglom_id, :] = np.mean(cluster_points, axis=0)
                PCs_agglom_var[agglom_id, :] = np.var(cluster_points, axis=0)
                kmeans_agglom_centroids[agglom_id, :] = np.mean(kmeans_points, axis=0)
                kmeans_agglom_var[agglom_id, :] = np.var(kmeans_points, axis=0)

            #mode_ids = np.tile([1,2,3], 3)
            #centroid_axs[0].scatter(mode_ids, agglom_centroids.flatten(), label=f"Run {axid}")
            #centroid_axs[1].scatter(mode_ids, agglom_var.flatten(), label=f"Run {axid}")
            mode_ids = np.tile([1,2,3], (3,1)).T
            centroid_axs[0].plot(mode_ids, PCs_agglom_centroids.T, label=f"Run {axid}")
            centroid_axs[1].plot(mode_ids, PCs_agglom_var.T, label=f"Run {axid}")

            kx, ky, kz = [PCs_agglom_centroids[:,i] for i in range(3)]
            agglom_centroids_ax3d.scatter(kx, ky, kz, label=f"Run {axid}")
            agglom_centroids_ax3d.set_xlabel("PC 1")
            agglom_centroids_ax3d.set_ylabel("PC 2")
            agglom_centroids_ax3d.set_zlabel("PC 3")

            kx, ky, kz = [kmeans_agglom_centroids[:,i] for i in range(3)]
            kmeans_centroids_ax3d.scatter(kx, ky, kz, label=f"Run {axid}")
            kmeans_centroids_ax3d.set_xlabel("PC 1")
            kmeans_centroids_ax3d.set_ylabel("PC 2")
            kmeans_centroids_ax3d.set_zlabel("PC 3")

            largest_dist = np.amax(agglom_links[:,2])
            max_dist = max(max_dist, largest_dist)

            #fig = plt.figure()
            dendrogram = scipy_hierarchy.dendrogram(agglom_links, ax=ax)
            ax.set_xticklabels([])
            #ax.set_xlabel("KMeans cluster ID")
            #ax.set_ylabel("Link distance")
        centroid_axs[0].set_xlabel(f"PCA mode")
        centroid_axs[0].set_title(f"Cluster centroids")
        centroid_axs[0].legend()
        centroid_axs[1].set_xlabel(f"PCA mode")
        centroid_axs[1].set_title(f"Cluster variance")
        centroid_axs[1].legend()

        agglom_centroids_ax3d.set_title("Agglomerative PCs centroids")
        agglom_centroids_ax3d.legend()
        kmeans_centroids_ax3d.set_title("Agglomerative kmeans centroids")
        kmeans_centroids_ax3d.legend()

        for ax in dendrogram_axs:
            ax.set_ylim((0, 1.1 * max_dist))
            xlim = ax.get_xlim()
            offset = (xlim[1] - xlim[0]) * 0.05
            ax.set_xlim((xlim[0] - offset, xlim[1] + offset))

        #t = np.arange(kmeans_labels.size)
        #fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        #for i in [0,1,2]:
        #    axs[i].plot(t, k_PCs[:,i])
        #    axs[i].set_ylabel(f"PC {i+1}")

        #fig = plt.figure()
        #ax3d = fig.add_subplot(111, projection='3d')
        #for k in range(kmeans_clusters):
        #    points = k_PCs[labels == k]
        #    plt.scatter(points[:,0], points[:,1], points[:,2])
        #ax3d.set_xlabel("PC 1")
        #ax3d.set_ylabel("PC 2")
        #ax3d.set_zlabel("PC 3")

        #fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        #axs = axs.flatten()
        #for k in range(kmeans_clusters):
        #    in_class = labels == k
        #    points = k_PCs[in_class]
        #    count = sum(in_class)
        #    print(f"Class {k+1} has {count} points ({count/in_class.size:0.2%})")
        #    label = f"Class {k+1}"
        #    marker = '+'
        #    axs[0].scatter(points[:,0], points[:,1], label=label, marker=marker)
        #    axs[1].scatter(points[:,1], points[:,2], label=label, marker=marker)
        #    axs[2].scatter(points[:,0], points[:,2], label=label, marker=marker)
        #axs[0].set_xlabel("PC 1")
        #axs[1].set_xlabel("PC 2")
        #axs[2].set_xlabel("PC 1")
        #axs[0].set_ylabel("PC 2")
        #axs[1].set_ylabel("PC 3")
        #axs[2].set_ylabel("PC 3")
        #axs[0].legend()
        #axs[1].legend()
        #axs[2].legend()

        #plt.figure()
        #plt.scatter(t, labels)
        self.save_figure(fig_name_parts=[f'kmeans-{kmeans_clusters}', 'agglom', 'dendrograms', 'all'])
        plt.show()
        



if __name__ == "__main__":
    # Run the script
    project = EOSC510Project(debug_mode=False)
    project.load_raw_data()
    project._plot_outliers_prominence()
    #project.analyze_pca_individually(make_plots=True, save_output=True)
    #project.analyze_pca_all(make_plots=True, save_output=True, save_plots=True)

    #project.reload_pca_pickle('all')
    #project.analyze_clusters_all()

    #project.reload_pca_pickle('individual')
    #project.analyze_2pass_stability(
    #        n_p1_clusters_list=[10, 30, 40, 50], 
    #        n_p2_clusters_list=[2,3,5,7],
    #        show_plots=True)
