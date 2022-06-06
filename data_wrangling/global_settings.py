#!/usr/bin/env python3
import numpy as np
from os.path import join as pjoin

gravity = 9.81 #m/s^2
water_density = 1000 # kg/m^3
sediment_density = 2650 # kg/m^3

flume_slope = 0.022
flume_width = 1 #m
init_bed_height = 21 #cm ### Double check this!

stationing_2m = [4500, 6500] # stationing in mm
dem_resolution = 2 #mm/px
dem_long_offset = 1124 # mm stationing of most downstream edge
dem_wall_trim = 100 # mm from each wall to throw away
dem_color_limits = [100, 260] # mm bed elevation range limits

lighttable_bedload_cutoff = 800 # g/s max rate

discharge_order = ['r50L', 'r62L', 'r75L', 'r87L', 'r100L',
                   'f87L', 'f75L', 'f62L']
# Feed scenario rates per step (kg/hr)
sum_feed_Di = {
    'D16' :  3.468005, # mm; calculated from 5 feed samples
    'D50' :  7.829806, # mm; calculated from 5 feed samples
    'D84' : 16.770694, # mm; calculated from 5 feed samples
}
feed_rates = {
        # An 8-element array representing feed rate (kg) per hour
        '1A' : np.zeros(8),
        '1B' : np.zeros(8),
        '2A' : np.concatenate(([0], np.ones(7))) * 800 / 7,
        '2B' : np.concatenate(([0], np.ones(7))) * 800 / 7,
        '3A' : np.concatenate(([0], np.ones(4), np.zeros(3))) * 800 / 4,
        '3B' : np.concatenate(([0], np.ones(4), np.zeros(3))) * 800 / 4,
        '4A' : np.concatenate(([0], np.zeros(3), np.ones(4))) * 800 / 4,
        '5A' : np.array([0, 64, 96, 139, 202, 139, 96, 64]),
        }


# General file settings
research_dir = "/home/alex/ubc/research/feed-timing"
data_dir = pjoin(research_dir, "data_reorganized")
figure_destination = pjoin(data_dir, "extras/prelim_figures")
#export_destination = pjoin(feed_timing_dir, "exported_data")
log_dir = pjoin(research_dir, "code/log-files")

## omnipickle dirs
#omnipickle_name = "omnipickle"
#omnipickle_path = pjoin(data_dir, f"{omnipickle_name}.pkl")

# lighttable dirs
lighttable_data_dir = pjoin(data_dir, "light_table_data/lt_results_by_period")
#lighttable_data_dir = pjoin(data_dir, "extracted-lighttable-results")
#Qs_raw_pickles = pjoin(lighttable_data_dir, "raw-pickles")
#Qs_primary_pickles = pjoin(lighttable_data_dir, "primary-processed-pickles")
#Qs_secondary_pickles = pjoin(lighttable_data_dir, "secondary-processed-pickles")
#Qs_tertiary_pickles = pjoin(lighttable_data_dir, "tertiary-processed-pickles")

# cart data dirs
photos_2m_data_dir = pjoin(data_dir, "2m_photo_data")
photos_8m_data_dir = pjoin(data_dir, "8m_photo_data")
dem_2m_data_dir = pjoin(data_dir, "8m_dem_data")
#cart_data_dir = pjoin(data_dir, "cart")
#cart_pickles_dir = pjoin(cart_data_dir, "cart-pickles")

# manual data dirs
flow_depth_data_dir = pjoin(data_dir, "flow_depth_data")
sieve_data_dir = pjoin(data_dir, "sieve_data")
trap_totals_data_dir = pjoin(data_dir, "trap_totals_data")
#manual_data_dir = pjoin(data_dir, "manual-data")
#manual_pickles_dir = pjoin(manual_data_dir, "manual-pickles")

# gsd data dirs
grain_size_data_dir = pjoin(data_dir, "gsd_data")

# pca output data dirs
pca_data_dir = pjoin(data_dir, "pca_data")
#pca_pickle_dir = pjoin(pca_dir, "pca-pickles")
pca_pickle_dir = pjoin(pca_data_dir, 'temp')
#pca_dir = pjoin(data_dir, "pca")
#pca_pickle_dir = pjoin(pca_dir, "pca-pickles")

