#!/usr/bin/env python3

# This script will load (partial) Qs# pickles according to the metapickle and 
# combine them into complete Qs pickles
#
# Qs_metapickle = {period_path: Qs_pickle_paths}
#
# Qs# pickles are panda dataframes directly translated from the raw txt files

import os
import numpy as np
from time import asctime

# From Helpyr
import data_loading
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from logger import Logger

# Outline:
# 1) load Qs_metapickle
# 2) load Qs pickles for a period
# 3) error check raw Qs dataframes
#   - conflict between Qs and Qs#?
# 4) combine Qs dataframes
# 5) error check combined Qs dataframe

# Script setup variables
root_dir = "/home/alex/feed-timing/data/extracted-lighttable-results"
pickle_source = f"{root_dir}/raw-pickles"
pickle_destination = f"{root_dir}/processed-pickles"
log_filepath = "./log-files/Qs_pickle_processor.txt"

logger = Logger(log_filepath, default_verbose=True)
ensure_dir_exists(pickle_destination, logger)

logger.write_section_break()
logger.write(["Begin pickle processor output", asctime()])

loader = data_loading.DataLoader(pickle_source, pickle_destination, logger)


# Load Qs_metapickle
metapickle_name = "Qs_metapickle"
metapickle = loader.load_pickle(metapickle_name)

print(metapickle)
periods = metapickle.keys()


