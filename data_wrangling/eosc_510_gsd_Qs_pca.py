#!/usr/bin/env python3

from os.path import join as pjoin
from os.path import isfile as os_isfile
import numpy as np
import pandas as pd
from scipy.stats import linregress as stats_linregress
from sklearn import decomposition as skl_decomposition
from sklearn.model_selection import train_test_split as skl_train_test_split
import matplotlib.pyplot as plt

from helpyr import logger
from helpyr import helpyr_misc as hpm

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


class EOSC510Project:

    def __init__(self):
        
        # Start up logger
        self.log_filepath = f"{settings.log_dir}/eosc510_project.txt"
        self.logger = logger.Logger(self.log_filepath, default_verbose=True)
        self.logger.begin_output("EOSC 510 project")

        # Reload omnimanager
        self.omnimanager = OmnipickleManager(self.logger)
        self.omnimanager.restore()

        self.figure_destination = settings.figure_destination
        self.figure_subdir = 'eosc510'
        self.fig_extension = '.png'
        hpm.ensure_dir_exists(self.figure_destination)

        self.rolling_missing_tolerance = 0.8
    
    def load(self):
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
        for exp_code in self.Qs_data.keys():
        #for exp_code in ['1A']:
            self.logger.write(f"Cleaning {exp_code}")
            experiment = self.omnimanager.experiments[exp_code]
            raw_pd_data = self.Qs_data[exp_code]
            bedload_all_pd = raw_pd_data['Bedload all']
            bedload_all = bedload_all_pd.values
            exp_time_hrs = raw_pd_data['exp_time_hrs'].values

            # Roll raw data
            window = 400
            tolerance = self.rolling_missing_tolerance
            raw_roller = bedload_all_pd.rolling(window,
                    min_periods=int(window * tolerance), center=True)
            rrstd = raw_roller.std()
            rrmean = raw_roller.mean()

            # Find "outliers"
            # Outlier if above both a threshold and a multiple of the std dev
            std_scale = 3
            threshold = 500
            with np.errstate(invalid='ignore'):
                above_threshold = bedload_all > threshold
                above_std = bedload_all > (rrmean + rrstd * std_scale).values 
            outliers_idx = np.where(
                    np.logical_and(above_threshold, above_std))[0]

            # Remove outliers
            # Note this changes self.Qs_data
            bedload_columns = [
                    'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                    'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                    'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                    'Bedload 22', 'Bedload 32', 'Bedload 45',
                    ]
        #    outliers_bedload = bedload_all_pd.iloc[outliers_idx].copy().values

            col_idx = np.where(np.in1d(raw_pd_data.columns, bedload_columns ))[0]
            raw_pd_data.iloc[outliers_idx, col_idx] = np.nan

        #    # Reroll on clean data
        #    bedload_all_pd = raw_pd_data['Bedload all']
        #    clean_roller = bedload_all_pd.rolling(window,
        #            min_periods=int(window * tolerance), center=True)
        #    crstd = clean_roller.std()
        #    crmean = clean_roller.mean()

        #    self._plot_outliers(exp_code,
        #            exp_time_hrs, raw_pd_data['Bedload all'],
        #            exp_time_hrs[outliers_idx], outliers_bedload,
        #            threshold, std_scale, rrmean, rrstd, crmean, crstd)
        #plt.show()
        #plt.close('all')
        #assert(False)

    def _plot_outliers(self, exp_code, time, bedload, o_time, o_bedload, threshold, std_scale, rrmean, rrstd, crmean, crstd):
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

        self.save_figure([f"outliers_{exp_code}"])


    def analyze(self):
        self.logger.write_blankline()
        for exp_code in self.Qs_data.keys():
        #for exp_code in ['1A']:
            self.logger.run_indented_function(
                    self._analyze_experiment, kwargs={'exp_code' : exp_code},
                    before_msg=f"Analyzing {exp_code}",
                    after_msg="")

    def _analyze_experiment(self, exp_code):
        experiment = self.omnimanager.experiments[exp_code]
        raw_pd_data = self.Qs_data[exp_code]
        fig_name_parts = [exp_code]

        # Drop Nan rows
        raw_pd_data.dropna(axis=0, inplace=True)

        # Pull out columns of interest
        bedload_all = raw_pd_data.pop('Bedload all').values
        exp_time = raw_pd_data.pop('exp_time').values
        exp_time_hrs = raw_pd_data.pop('exp_time_hrs').values
        discharge = raw_pd_data.pop('discharge').values
        feed = raw_pd_data.pop('feed').values

        # Drop columns we don't care about
        drop_cols = ['timestamp', 'missing ratio', 'vel', 'sd vel',
                'number vel', 'Count all', 'Count 0.5', 'Count 0.71',
                'Count 1', 'Count 1.4', 'Count 2', 'Count 2.8', 'Count 4',
                'Count 5.6', 'Count 8', 'Count 11.2', 'Count 16',
                'Count 22', 'Count 32', 'Count 45', 'D10', 'D16', 'D25',
                'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax',]
        raw_pd_data.drop(columns=drop_cols, inplace=True)

        # Get grain sizes 
        grain_sizes = np.array([ float(s.split()[1]) 
            for s in raw_pd_data.columns])

        model, PCs, evectors, evalues, data_stddev, data_mean, pca_codes = \
                self.do_pca(raw_pd_data)
        fig_name_parts += pca_codes

        self._plot_eigenvalues(exp_code, evalues, fig_name_parts)
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

        # Plot PCA variables
        self._plot_PCs(exp_code, exp_time_hrs, PCs, keep_modes, 
                fig_name_parts_k, ylim=(-25, 50))
        self._plot_PCs(exp_code, exp_time_hrs, PCs, keep_modes, 
                fig_name_parts_k)
        self._plot_PCs_comparisons(exp_code, PCs, keep_modes, fig_name_parts_k)
        self._plot_eigenvectors(exp_code, grain_sizes, evectors, keep_modes, 
                fig_name_parts_k)
        
        # Reconstruct 
        recon_k = PCs[:, 0:keep_modes] @ evectors[0:keep_modes, :]
        recon_1 = np.outer(PCs[:, 0], evectors[0, :])

        # Rescale
        rescale_fu = lambda recon_data: recon_data * data_stddev + data_mean
        bedload_all_fu = lambda data: np.sum(data, axis=1)
        recon_bedload_all_k = bedload_all_fu(rescale_fu(recon_k))
        recon_bedload_all_1 = bedload_all_fu(rescale_fu(recon_1))

        # Plot reconstructions
        recon_name = "bedload all"
        recon_fig_name = recon_name.replace(' ', '-')
        fig_name_parts_k = [recon_fig_name] + fig_name_parts_k

        self._plot_single_reconstruction(exp_code, recon_name, keep_modes, 
                exp_time_hrs, bedload_all, recon_bedload_all_k,
                fig_name_parts=fig_name_parts_k)
        self._plot_reconstruction_fit_single(exp_code, recon_name,
                keep_modes, bedload_all, recon_bedload_all_k,
                fig_name_parts=fig_name_parts_k)

        fig_name_parts_1 = [recon_fig_name, f"k1"] + fig_name_parts
        self._plot_single_reconstruction(exp_code, recon_name, 1, 
                exp_time_hrs, bedload_all, recon_bedload_all_1,
                fig_name_parts=fig_name_parts_1)
        self._plot_reconstruction_fit_single(exp_code, recon_name,
                1, bedload_all, recon_bedload_all_1,
                fig_name_parts=fig_name_parts_1)

        #plt.show()
        plt.close('all')

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

        return model, PCs, evectors, evalues, data_stddev, data_mean, pca_codes


    ## Plotting functions
    def _plot_reconstruction_fit_single(self, exp_code, name, k, original, recon, fig_name_parts):
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

        self.save_figure([f"recon_fit"] + fig_name_parts)

    def _plot_single_reconstruction(self, exp_code, name, k, time, original, recon, fig_name_parts):
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

        self.save_figure([f"recon"] + fig_name_parts)

    def _plot_eigenvectors(self, exp_code, grain_sizes, evectors, keep_modes, fig_name_parts):
        # plot each eigenvector
        # sklearn puts evector modes in rows
        fig, axs = self._get_squarish_subplots_grid(keep_modes)
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

        for ax in axs[2, :]:
            ax.set_xlabel("Grain size (mm)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        self.save_figure([f"evectors"] + fig_name_parts)

    def _plot_PCs_comparisons(self, exp_code, PCs, keep_modes, fig_name_parts):
        # plot comparisons between PCs
        fig, axs = self._get_squarish_subplots_grid(keep_modes-1)
        plt.suptitle(f"{exp_code} PCs comparisons")
        nrows, ncols = axs.shape
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

        for ax in axs[2, :]:
            ax.set_xlabel("PC 1")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        self.save_figure([f"PCs_comparisons"] + fig_name_parts)

    def _plot_PCs(self, exp_code, exp_time_hrs, PCs, keep_modes, fig_name_parts, ylim=None):
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
        
        self.save_figure([f"PCs"] + fig_name_parts)

    def _plot_eigenvalues(self, exp_code, evalues, fig_name_parts):
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

        self.save_figure([f"evalues"] + fig_name_parts)


    ## General plotting support functions
    def _get_squarish_subplots_grid(self, n):
        nrows = int(np.ceil(np.sqrt(n)))
        ncols = int(np.ceil(n / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                sharex=True, sharey=True)
        return fig, axs

    def save_figure(self, figure_name_parts=[]):
        figure_name = '_'.join(figure_name_parts) + self.fig_extension

        filepath = pjoin(self.figure_destination,
                self.figure_subdir, figure_name)
        self.logger.write(f"Saving figure to {filepath}")
        plt.savefig(filepath, orientation='landscape')



if __name__ == "__main__":
    # Run the script
    project = EOSC510Project()
    project.load()
    project.analyze()
