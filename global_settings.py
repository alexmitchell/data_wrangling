#!/usr/bin/env python3

gravity = 9.81 #m/s^2
water_density = 1000 # kg/m^3
sediment_density = 2650 # kg/m^3

flume_slope = 0.022
flume_width = 1 #m
init_bed_height = 21 #cm ### Double check this!

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


