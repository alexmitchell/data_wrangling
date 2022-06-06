################################################################################
Brief installation instructions:
> conda create --name data_wrangling_dev
> conda activate data_wrangling_dev
# Optional to create .env and .unenv files to make activation automatic
> conda install --file requirements.txt
> python setup.py develop
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
New goals (26/11/2021):
- Break the larger program into smaller scripts that can be run individually.
- Simplify code so that other programmers (or myself at a later date) can 
  easily make sense of it and use it.
- Have the code produce csv files for intermediate and final data, so I can 
  easily generate datasets for non-programmers who don't know how to run the 
  scripts.
- Remove dependencies to non-standard libraries (like helpyr).

################################################################################
################################################################################
Started this file way too late...

Lighttable workflow:
extraction_crawler.py -- pickle the raw lighttable data
Qs_pickle_processor.py -- clean and combine raw data pickles.
Qs_grapher.py -- graph the data



Other:
linking_crawler.py -- create sym links in one loc for distributed target files




On usability of this code:
After another person attempted to use this code for their analysis, it became 
clear early on that most of it cannot be used without significant alterations.  
From the second Qs processor onwards, the code is heavily dependent on my file 
structure. Due to iterative "improvements" to parts of the program, trying to 
adapt the code to a new project will be very difficult. Some of the updates 
should ideally be quite useful, but the stitched together nature of it (with 
changing paradigms for handling data) makes the code fragile.

Rather than adapt the code for a new project, I recommend rebuilding it 
completely. Use this one as a prototype for the second version. Here are some 
comments and ideas:
- Split the Qs primary processor out of the workflow and make it a simplified 
  standalone program. Being able to stitch the Qs# files together into a text 
  file appears to be a useful on its own. (It also does not require knowledge 
  of the file structure)

- The Omnipickle/omnimanager seems to be quite helpful. It keeps track of all 
  the different data sources, provides a common interface, and easy ways to 
  store the whole data tree.

- Helpyr modules have been quite useful too. You may want to clean up the API 
  and code a bit. Much of it is old code that can be ugly. In particular, get 
  rid of the 'printer' function from the DataLoader.  Also rename to 
  DataLoader.  Don't worry about breaking legacy code (Branch or fork it?)

- I like the DataSet class (in tokens.py). It provides a nice endpoint for data 
  managed by the OmniManager. 

- You will have to debate whether it is worth it to subdivide the data into the 
  smallest relative units (eg. periods for me). I found that I spent much 
  effort dealing the data out to experiments (1A, 1B, 2A, etc.) and periods 
  ('rising 75 t20-t40') just to spend as much effort recombining it for 
  graphing. However, the data is not necessarily defined for the same times 
  (different times, time resolutions, labels, etc.). Splitting down into 
  periods helped with keeping track of which data chunks were related to each 
  other and for non-dataframe data. The original intent was to allow easy 
  comparisons and calculations between different types of data within one 
  period. However, I have yet to use the data in this way. (the simple 
  calculations I've done so far either operate on the whole dataset or used 
  groupby)
  
  The alternative is highest level class contains the entire data set for each 
  type of data. (eg. one var for all Qs data, one var for all gsd data, etc.) 
  Not sure what the pitfalls will be for this method.
  
  If you choose to split, make a generic division class which allows arbitrary 
  number of division levels. Let the user choose what each level will split on 
  (eg. first level is experiment codes, second is period codes). In my project, 
  there is a lot of common functions between the Experiment and PeriodData 
  classes. It should be possible to merge them, perhaps moving specialized 
  functions to the Omnimanager. (or generic division inheriting classes?)

- Use the processor to convert all the labels in the raw data to match a common 
  set of labels. (eg. "rising-75L" vs "r75L" or "t20-t40" vs "t40") Then the 
  Omnimanager and data tree will be cleaner and easier to use.

- Remove the dependence on Qs being processed first. It is bothersome that the 
  abstract data tree is build from a particular data set. Say you don't have Qs 
  data, then the code would be useless.

- The Universal Grapher is just a hot mess and I'm not sure of a way to make it 
  better. Each type of graph NEEDS its own specific plotting and formatting 
  code. Even if several different plots have similar code structure, it would 
  be impossible/impracticable to abstract it more. Some effort has been make to 
  use general functions though. It would be good to make the functions/argument 
  style consistent. I kept switching between using kwargs, **kwargs, sticking 
  kwargs in other kwargs, and other ways to pass information around.

- It would be nice to abstract the Omnimanager from data types. That would make 
  the code more portable to other types of research, not just flume experiments 
  in our lab.

- In the same vein, perhaps a good idea for the Data Wrangling project in 
  general is to abstract it from any particular research project. It provides 
  the data processing structure (and a template for a grapher), but the user 
  would create a new project that uses/inherits the Data Wrangling classes.  
  Perhaps include base classes for processing common data types like the 
  lighttable or sieve samples. (New projects can create new processor base 
  classes too)



