#!/usr/bin/env python3
# Updates the Omnipickle tree definitions.
from os.path import join as pjoin
from time import asctime

# From helpyr
from helpyr.logger import Logger

from omnipickle_manager import OmnipickleManager
import global_settings as settings

# Start up logger
log_filepath = pjoin(settings.log_dir, "update_tokens.txt")
logger = Logger(log_filepath, default_verbose=True)
logger.write(["Begin Token Update output", asctime()])

# Reload omnimanager
omnimanager = OmnipickleManager(logger)
omnimanager.restore()
omnimanager.update_tree_definitions()
omnimanager.store()
