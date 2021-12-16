#!/usr/bin/env python3

import Qs_pickle_processor as QsPP
primary = QsPP.PrimaryPickleProcessor(output_txt=True)
primary.run()
secondary = QsPP.SecondaryPickleProcessor() # Creates the omnimanager
secondary.run()

import manual_processor as MP
manual_processor = MP.ManualProcessor()
manual_processor.run()

import dem_processor as DP
dem_processor = DP.DEMProcessor()
dem_processor.run()

import gsd_processor as GP
gsd_processor = GP.GSDProcessor()
gsd_processor.run()
