#!/usr/bin/env python3
import numpy as np

gravity = 9.81 #m/s^2
water_density = 1000 # kg/m^3
sediment_density = 2650 # kg/m^3

flume_slope = 0.022
flume_width = 1 #m
init_bed_height = 21 #cm ### Double check this!


discharge_order = ['r50L', 'r62L', 'r75L', 'r87L', 'r100L',
                   'f87L', 'f75L', 'f62L']
# Feed scenario rates per step (kg/hr)
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
figure_destination = f"{root_dir}/prelim-figures"
log_dir = "/home/alex/feed-timing/code/log-files"

# lighttable dirs
lighttable_data_dir = f"{root_dir}/extracted-lighttable-results"
Qs_pickles_source = f"{lighttable_data_dir}/secondary-processed-pickles"

# cart data dirs
cart_data_dir = f"{root_dir}/cart"
cart_pickles_dir = f"{cart_data_dir}/cart-pickles"

# manual data dirs
manual_data_dir = f"{root_dir}/manual-data"
manual_pickles_dir = f"{manual_data_dir}/manual-pickles"


