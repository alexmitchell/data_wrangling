#!/usr/bin/env python3

from data_wrangling.data_collectors.data_collector_base import DataCollectorBase

class DepthCollector(DataCollectorBase):

    def __init__(self, data_root_dir, output_dir):
        super().__init__(data_root_dir, output_dir)
        self.name = 'depth'

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
