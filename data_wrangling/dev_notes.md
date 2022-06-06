################################################################################
26/11/2021

extraction_crawler.py -- pickle the raw lighttable data
Qs_pickle_processor.py -- clean and combine raw data pickles.
Qs_grapher.py -- graph the data

linking_crawler.py -- create sym links in one loc for distributed target files


################################################################################
1) Collecting data -- Find all the files and combine them into fewer files. 
(one file per experiment per data type). Need to check file sizes though. 
Maybe split by hour instead of by experiment (8hrs)?

2) Processing -- Process step by step, saving data in csv files if it might 
potentially be used on it's own.

3) Plotting -- Using a core set of functions in a plotting utility class, many 
satellite scripts can generate customized plots.
