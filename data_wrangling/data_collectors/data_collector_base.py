#!/usr/bin/env python3

from os import walk as os_walk
from os.path import join as path_join
from os.path import (
        basename,
        dirname,
        splitext,
        )
import fnmatch

class DataCollectorBase:

    def __init__(self, data_root_dir, output_dir):
        # Initialize the Collector base.
        #
        # data_root_dir is the root of the directory tree where the raw data 
        # are stored
        # output_dir is where the process data will be saved (or the root 
        # directory for the saved data)
        self.root_dir = data_root_dir
        self.output_dir = output_dir

        self.name = None

    """ Functions that inheriting collector class needs to override"""
    def is_my_file(self, filepath):
        """ Test if the provided filepath matches what this collector is 
        supposed to process.

        Needs to be defined by the inheriting Collector class.

        Parameters
        ----------
        filepath : string or string-like path object
            The full filepath of the file to be checked.

        Returns
        -------
        bool
            True indicates that the filepath is the right type of file for the 
            inheriting collector (e.g.  a light table file for the 
            LightTableCollector).
        """
        raise NotImplementedError(
                "Inheriting collector class needs to define this function"
                )

    def load_data_file(self, filepath):
        """ Load a single data file.

        The inheriting Collector class needs to define this function to read a 
        file into a Pandas DataFrame.

        Parameters
        ----------
        filepath : string or string-like path object
            The full filepath of the file to be loaded.

        Returns
        -------
        pandas.DataFrame
        """
        raise NotImplementedError(
                "Inheriting collector class needs to define this function"
                )

    def combine_data(self, data_dict):
        """ Combines all the Pandas DataFrames in data_dict.

        The inheriting Collector class needs to define this function to combine 
        all the Pandas DataFrames provided in data_dict into one large 
        DataFrame. It should include any basic cleaning and organization (e.g. 
        index sorting and getting the correct timestamps)

        Parameters
        ----------
        data_dict : dictionary
            A dictionary of all the data in the format {filename : DataFrame}.

        Returns
        -------
        pandas.DataFrame
            One large DataFrame containing all the smaller DataFrames in data_dict
        """
        raise NotImplementedError(
                "Inheriting collector class needs to define this function"
                )

    def save_file(self, data, extension):
        """ Save a Pandas DataFrame to a file.

        The inheriting Collector class needs to define this function. It should 
        use self.output_dir when generating the filepath for the new file.

        Parameters
        ----------
        data : Pandas DataFrame
            A DataFrame containing all the data for this Collector.
        extension : str
            Indicates the type of file to save it as.

        Returns
        -------
        str
            Returns the filepath of the newly created file
        """
        raise NotImplementedError(
                "Inheriting collector class needs to define this function"
                )

    """ Functions that inheriting collector class can use as is"""
    def run_collector(self, save_file_ext=None):
        """ Save a Pandas DataFrame to a file.

        Runs the typical sequence of Collector methods automatically. Searches 
        for files, loads data, combines data, and then saves combined data.

        Parameters
        ----------
        save_file_ext : str, default None
            Indicates the type of file to save it as. If the value is None, 
            then the combined data will be returned instead of saved to a file. 
            If the value is not None (e.g. 'csv'), then the combined data will 
            be saved to a file and the filepath is returned.

        Returns
        -------
        Pandas DataFrame or str
            Depending of the value of save_file_ext, returns either a Pandas 
            DataFrame or a filepath to a newly created file containing the 
            final combined data. See Parameters for more info.
        """

        # Find files in data tree
        filepaths = self.gather_filepaths()
        if len(filepaths) == 0:
            print("{self.name or ''} collector could not find any data")
            return None

        # Load data
        data_dict = self.load_data_files(filepaths)

        # Combine data entries in data_dict
        all_data = self.combine_data(data_dict)
        
        # Save to file or return
        if save_file_ext is None:
            return all_data
        else:
            return self.save_file(all_data, save_file_ext)

    def load_data_files(self, filepaths, strip_extension=False):
        """ Load multiple data files.

        Uses the inheriting collector's load_data_file to read multiple data 
        files and then return all the data in a dictionary.

        Parameters
        ----------
        filepaths : list of string or string-like path objects
            The full filepaths of multiple data files to be loaded.

        strip_file_ext : bool, default False
            Indicates if the file extension should be removed when setting keys 
            for the returning data dictionary.

        Returns
        -------
        dictionary
            A dictionary containing all the data in the form {filename : data}
        """
        if strip_extension:
            get_fname = lambda fpath: splitext(basename(fpath))[0]
        else:
            get_fname = basename
        load = self.load_data_file

        data_dict = {get_fname(fpath): load(fpath) for fpath in filepaths}
        return data_dict

    def gather_filepaths(self):
        """ Find all the filepaths desired by inheriting Collector object in 
        the data directory tree.

        This function uses the is_my_file method defined by the inheriting 
        Collector to determine if the file matches or not as it walks through 
        the data directory tree. Note that the os walk does not follow any 
        particular order, so the resulting files list is not necessarily 
        sorted.

        Parameters
        ----------

        Returns
        -------
        list of str
            Returns a list of filepaths that match what the inheriting 
            Collector is looking for.
        """
        filepaths = []
        for parent_dir, subdirs, filenames in os_walk(self.root_dir):
            if len(filenames) == 0:
                # No files in this directory level, continue walking...
                continue

            for fname in filenames:
                # Generate the full filepath to each file
                fpath = path_join(parent_dir, fname)

                # Remember the file if the collector is looking for it.
                if self.is_my_file(fpath):
                    filepaths.append(fpath)

        return filepaths

    def file_match_by_name(self, filepath, match_patterns):
        """ A simple matching function that extracts the filename from a 
        filepath and checks for the provided patterns. Intended for use with 
        the is_my_file method.

        Parameters
        ----------
        filepath : string or string-like path objects
            The filepath to check.

        match_patterns : string or list of strings
            Pattern or list of patterns to check the filename with. These 
            patterns are used directly by fnmatch and accept wildcards like 
            '?', '*' and '[]'. See fnmatch documentation 
            (https://docs.python.org/3/library/fnmatch.html) for more 
            information.

        Returns
        -------
        bool
            Returns true if the filename matches any of the provided patterns.
        """

        if isinstance(match_patterns, str):
            # Convert a single string into a list with one string element
            match_patterns = [match_patterns]

        filename = basename(filepath)
        #directory = dirname(filepath)

        matches = [fnmatch.fnmatch(filename, p) for p in match_patterns]
        return any(matches)

