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
root_dir = "/home/alex/feed-timing/data"
figure_destination = pjoin(root_dir, "prelim-figures")
log_dir = "/home/alex/feed-timing/code/log-files"

# omnipickle dirs
omnipickle_name = "omnipickle"
omnipickle_path = pjoin(root_dir, f"{omnipickle_name}.pkl")

# lighttable dirs
lighttable_data_dir = pjoin(root_dir, "extracted-lighttable-results")
Qs_raw_pickles = pjoin(lighttable_data_dir, "raw-pickles")
Qs_primary_pickles = pjoin(lighttable_data_dir, "primary-processed-pickles")
Qs_secondary_pickles = pjoin(lighttable_data_dir, "secondary-processed-pickles")
Qs_tertiary_pickles = pjoin(lighttable_data_dir, "tertiary-processed-pickles")

# cart data dirs
cart_data_dir = pjoin(root_dir, "cart")
cart_pickles_dir = pjoin(cart_data_dir, "cart-pickles")

# manual data dirs
manual_data_dir = pjoin(root_dir, "manual-data")
manual_pickles_dir = pjoin(manual_data_dir, "manual-pickles")


