#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# prototyping code to read depth data

datapath = '/home/alex/feed-timing/data/data-links/manual-data/'
figpath = '/home/alex/feed-timing/data/data-links/manual-data/figures/'
filename = 'flow-depths-5A.xlsx'
filepath = datapath + filename

kwargs = {
        'sheetname'  : 'Sheet1',
        'header'     : 0,
        'skiprows'   : 1,
        'index_col'  : [0, 1, 2, 3],
        'parse_cols' : range(1,18)
        }

def get_data():
    data = pd.read_excel(filepath, **kwargs)
    data.sort_index(inplace=True)

    def orderizer(args):
        weights = {'rising'  : 0,
                   'falling' : 1,
                   50  : 0,
                   62  : 1,
                   75  : 2,
                   87  : 3,
                   100 : 4
                  }
        w_Qmax = weights[100]
        w_limb = weights[args[0]]
        w_Q = weights[args[1]]
        order = w_Q  + 2 * w_limb * (w_Qmax - w_Q)
        return order


    # Rename provided level names
    new_names = ['Limb', 'Q', 't', 'Loc']
    data.index.names = new_names
    #data.index.names = ['Limb', 'Discharge (L/s)', 'Time (min)', 'Location']
        
    # Add a new level providing the row order
    data['Order'] = data.index
    data['Order'] = data['Order'].map(orderizer)
    data.set_index('Order', append=True, inplace=True)

    # Make the Order index level zero
    new_names.insert(0, 'Order')
    data = data.reorder_levels(new_names)

    data.sort_index(inplace=True)
    return data

def get_loc(data, loc):
    slice_all = slice(None)
    d = data.loc[(slice_all, slice_all, slice_all, slice_all, loc),:]

    d.index = d.index.droplevel(4)
    return d

def get_scale(min, max, n_colors):
    return (max-min) * np.linspace(0, 1, n_colors) + min

def plot_profiles(profiles, color, figure=None, ax=None):
    n_profiles = profiles.shape[0]

    # Make some color scales
    light_scale = get_scale(0.25, 1, n_profiles)
    dark_scale = get_scale(0, 0.75, n_profiles)
    zeros = np.zeros(n_profiles)

    scales = {
            "red" : np.stack((light_scale, zeros, zeros), axis=1),
            "blue" : np.stack((zeros, zeros, light_scale), axis=1),
            "grey" : np.stack((dark_scale, dark_scale, dark_scale), axis=1),
            }
    styles = [':'] + ['-']*4 + ['-.']*3

    t_profiles = profiles.transpose()
    if ax is None:
        if figure is None:
            return t_profiles.plot(color=scales[color], style=styles)
        else:
            return t_profiles.plot(figure=figure, color=scales[color], style=styles)
    else:
        if figure is None:
            return t_profiles.plot(ax=ax, color=scales[color], style=styles)
        else:
            return t_profiles.plot(figure=figure, ax=ax, color=scales[color], style=styles)

def make_graph(experiment, surface, bed):

    depth = surface - bed

    # Plot surface, bed, and depth values
    ax = None
    data_color_pairs = (
            (surface, "blue"),
            (bed, "grey"),
            (depth, "red"),
            )
    for profiles, color in data_color_pairs:
        hour_av = profiles.mean(axis=0, level='Order')
        ax = plot_profiles(hour_av, ax=ax, color=color)

    # Make an overall legend title that acts like column titles
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

    save_args = {
            'fname' : "{}-hr-av.png".format(experiment),
            #'dpi'   : 80,
            'orientation' : 'landscape',
            }
    return figure, save_args

def fumain():
    data = get_data()

    surface = get_loc(data, 'surface')
    bed = get_loc(data, 'bed')

    figures = {}
    figure, save_args = make_graph("5A", surface, bed)
    figures[figure] = save_args

    for fig in figures:
        saveargs = figures[fig]
        fname = save_args.pop('fname')
        fpath = fname #figpath + fname
        fig.savefig(fpath, **saveargs)


if __name__ == "__main__":
    fumain()
