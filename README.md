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



