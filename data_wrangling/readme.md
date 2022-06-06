################################################################################
################################################################################
# 05/06/2022

# eosc_510_gsd_Qs_pca_2022-update.py
The original eosc_510 code was originally written in fall 2019 as part of a 
term project for EOSC 510 (Data Analysis in Earth Science). It explored the 
light table data produced from my master's flume experiments to try to identify 
if there were patterns linked to the feed rates. 

The eosc 510 code was an offshoot of my larger data_wrangling code base and 
used some code from another repository I wrote (Helpyr). The data wrangling 
code read all of the data types (light table, sieve, etc.) from all of the 
experiments (1A, 1B, 2A, etc.) into a tree of objects representing periods and 
experiments and all contained within one large data structure, called the 
omnipickle manager. To speed up reading and writing, the code made heavy use of 
the pickles package to write/read Python objects directly as binary files.

However, I've come to realize a few significant deficiencies in this code:
- Have a single object containing all the data and then storing data in 
  customized objects containing time chunks (with all the types in that time 
  chunk) turns out to be a big pain to use, especially for other people or 
  myself after enough time. I think it would be better to have a set of 
  utilities to read and write each type of data from the raw files so the 
  individual processing scripts or plotting scripts can operate independently. 
  Also processing scripts should write back out to a human-readable format.
- Pickles are a security risk. They can contain arbitrary Python and would be 
  easy to exploit. Don't use pickles you didn't make.
- The helpyr repo is nice for me, but would be a pain for others to use. Also 
  some of the more used functionality (a kwarg checker) has the same 
  functionality as built in functions that I didn't know about at the time.

In the current version, I unfortunately don't have the time to make most of the 
changes that I want. I focused on getting the PCA code running by removing the 
Helpyr dependency, removing the omnipickled dependency, and reading directly 
from the light table files in the reorganized data directory. However, I did 
not otherwise refactor the code or improve for clarity. It's still a bit of a 
mess.

The legacy Data Wrangler code may or may not run. If you have all the right 
python modules installed on your computer, then it should. However, most of it 
is the data manager that I think should be thrown away, and the rest of it just 
plots data from different perspectives. I wasn't able to do much heavy 
processing before the PCA project. So, I think you would want to write a new 
system for reading, processing, and plotting data. Also I radically condensed 
the directory structure to make it significantly easier to use.

The data_collectors folder has some modules that aspired to provide read/write 
utilities for each data type while maintaining a uniform API. That way, any 
processing/plotting scripts can easily load only the data it needs and is 
cleaner code to read. However, I unfortunately have to abandon this to move on 
to other tasks.


################################################################################
################################################################################
# 02/06/2022
I've reorganized all of the data files and renamed most of them so that they 
are all in an easily accessible location with unique and descriptive names.

However, I'm abandoning efforts for a full reimplementation of the data 
wrangling code. I ran out of time (and motivation) to do this. Instead, I will 
try to get the PCA code running and then return the drive to the lab.

################################################################################
################################################################################
# 26/11/2021

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
