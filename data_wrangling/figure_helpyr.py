#!/usr/bin/env python3

import os
from os.path import join as pjoin

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Removing dependency on helpyr
#from helpyr import kwarg_checker
#from helpyr import helpyr_misc as hpm

# Functions from helpyr
def ensure_dir_exists(dir_path, logger=None):
    # Creates directory if not already created
    # Return True if directory created or False if directory already 
    # exists.
    # logger is a Logger object (found in Helpyr)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        msg = "Making directory at {}.".format(dir_path)
        if logger is None:
            print(msg)
        else:
            logger.write([msg])
        return True
    return False


class FigureSaver:
    """ 

    Provides basic file saving functionality that I've needed in several 
    projects. Mostly provides convenient filename assembly options. 

    """

    def __init__(self, **kwargs):
        self.figure_extension = kwargs.get('figure_extension', 'png')
        self.logger = kwargs.get('logger', None)
        self.debug_mode = kwargs.get('debug_mode', False)

        self.figure_root_dir = kwargs.get('figure_root_dir', './')
        sub_dir = kwargs.get('figure_sub_dir', '')
        self.figure_dir = pjoin(self.figure_root_dir, sub_dir)

        ensure_dir_exists(self.figure_root_dir)
        ensure_dir_exists(self.figure_dir)

    def save_figure(self, **kwargs):
        """

        Save a figure function. Provides convenient filename assembly options. 

        Kwargs:
        'figure_name' is a complete name for a figure. It will be used directly 
        as the filename. Can't define with 'figure_name_parts'.

        'figure_name_parts' or 'fig_name_parts' is the ordered sequence of file
        name chunks. The chunks will be joined with '_' to generate the file 
        name. Can't define with 'figure_name'.

        'sep' is the separator used to join the figure_name_parts

        'figure' is the handle for the figure to save.

        'alt_subdir' is the optional alternate subdirectory to use instead of the one
        provided in the constructor.

        Any unused kwargs are passed to plt.savefig or fig.savefig

        """
        kwargs_copy = kwargs.copy()
        figure_name = kwargs.get('figure_name', None)
        figure_name_parts = kwargs.get('figure_name_parts', None)
        fig_name_parts = kwargs.get('fig_name_parts', None)
        sep = kwargs.get('sep', '_')
        figure = kwargs.get('figure', None)
        alt_subdir = kwargs.get('alt_subdir', None)

        try:
            assert( (figure_name is not None) ^ 
                    (figure_name_parts is not None) ^
                    (fig_name_parts is not None) ) # XOR
        except AssertionError:
            print(figure_name)
            print(figure_name_parts)
            print(fig_name_parts)
            raise

        if figure_name is None:
            if figure_name_parts is None:
                figure_name = sep.join(fig_name_parts)
            else:
                figure_name = sep.join(figure_name_parts)

        # Check if an alternate subdirectory is desired
        if alt_subdir is None:
            destination_dir = self.figure_dir
        else:
            destination_dir = pjoin(self.figure_root_dir, alt_subdir)
            ensure_dir_exists(destination_dir)

        filename = f"{figure_name}.{self.figure_extension}"
        filepath = pjoin(destination_dir, filename)
        
        if self.debug_mode:
            msgs = ["!!!", f"   Not saving figure to {filepath}", "!!!"]
            if self.logger is None:
                for msg in msgs:
                    print(msg)
            else:
                self.logger.write(msgs)
        else:
            msg = f"Saving figure to {filepath}"
            if self.logger is None:
                print(msg)
            else:
                self.logger.write(msg)

            if figure is not None:
                # Save target figure
                figure.savefig(filepath, orientation='landscape', **kwargs_copy)
            else:
                # Save current figure
                plt.savefig(filepath, orientation='landscape', **kwargs_copy)



class StableSubplots:

    """
    
    Allows for creating subplot grids and adding axes from inside a loop. 
    The first call to add_subplot with a new figure name will create a new 
    figure. Subsequent calls to add_subplot with the same figure name will 
    reuse the existing figure.
    
    Useful for long/messy loops or if the looping structure is not in the same
    code location as the plotting function. Keeps the plotting parameters in 
    one location where it is relevant rather than bloating the loop 
    initialization. 
    
    """

    existing_plots = {} # {name : [fig_reference, axs_used]}
    
    def __init__(self, logger=None):
        self.logger = logger
        if logger is not None:
            logger.write("Initializing a StableSubplots object")

    def add_subplot(self, fig_name, **kwargs):
        """

        Add an axis to either a new or existing figure based on fig_name
        fig_name will also be used as a filename. 

        """

        # Get kwargs
        subplots_shape = kwargs.get('subplots_shape', (1,1))
        fig_kwargs = kwargs.get('fig_kwargs', {})
        ax_kwargs = kwargs.get('ax_kwargs', {})
        suptitle = kwargs.get('suptitle', '')

        nrows, ncols = subplots_shape

        # Reuse existing figure or create new one
        if fig_name in StableSubplots.existing_plots:
            # Use existing figure
            fig, axs = StableSubplots.existing_plots[fig_name]
            ax_id = len(axs) + 1
        else:
            # Make new figure
            fig = plt.figure(**fig_kwargs)
            fig.suptitle(suptitle)
            axs = np.array([])
            ax_id = 1
        assert(1 <= ax_id <= nrows * ncols)

        # Print some output
        is_new_str = 'new' if ax_id == 1 else 'existing'
        msg = f"Creating new axis for {is_new_str} figure {fig_name}"
        if self.logger is None:
            print(msg)
        else:
            self.logger.write(msg)

        # Create the subplot axis
        ax = fig.add_subplot(nrows, ncols, ax_id, **ax_kwargs)

        # Update existing_plots
        axs = np.append(axs, ax)
        StableSubplots.existing_plots[fig_name] = [fig, axs]

        return ax

    def share_3D_scales(self, fig_name):

        fig, axs = StableSubplots.existing_plots[fig_name]

        # Set up scale range for each of three dimensions
        plot_scales = np.zeros((3, 2))

        for ax in axs:
            lim = np.array([
                ax.get_xlim(), 
                ax.get_ylim(),
                ax.get_zlim(),
                ])
            plot_scales[:,0] = np.amin([lim[:,0], plot_scales[:,0]], axis=0)
            plot_scales[:,1] = np.amax([lim[:,1], plot_scales[:,1]], axis=0)

        for ax in axs:
            ax.set_xlim(plot_scales[0,:]), 
            ax.set_ylim(plot_scales[1,:]),
            ax.set_zlim(plot_scales[2,:]),
    def finish(self, fig_name=None, show=True, saving_fu=None):

        existing = StableSubplots.existing_plots
        target_figs = list(existing.keys()) if fig_name is None else [fig_name]
        if saving_fu is None:
            saver = FigureSaver(logger=self.logger)
            saving_fu = saver.save_figure

        # Save the target figures
        for name in target_figs:
            fig, axs = existing[name]
            if '_3d_' in name:
                self.share_3D_scales(name)

            saving_fu(figure_name=name, figure=fig)

        # Show figures if applicable, otherwise close them
        if show:
            plt.show()
        else:
            for name in target_figs:
                fig, axs = existing[name]
                plt.close(fig)

        # Remove figures from the existing_plots dictionary
        for name in target_figs.copy():
            del StableSubplots.existing_plots[name]


