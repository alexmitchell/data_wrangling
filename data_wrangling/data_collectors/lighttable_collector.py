#!/usr/bin/env python3


from data_wrangling.data_collectors.data_collector_base import DataCollectorBase

class LightTableCollector(DataCollectorBase):

    def __init__(self, data_root_dir, output_dir):
        super().__init__(data_root_dir, output_dir)
        self.name = 'lighttable'


    """ Overridden methods """
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

        match_patterns = ['Qs?.txt', 'Qs??.txt']

        return self.file_match_by_name(filepath, match_patterns)

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
        
        # Prepare kwargs for reading Qs text files
        Qs_column_names = [
                # Timing and meta data
                #'elapsed-time sec', <- Calculate this column later
                'timestamp', 'missing ratio', 'vel', 'sd vel', 'number vel',
                # Bedload transport masses (g)
                'Bedload all', 'Bedload 0.5', 'Bedload 0.71', 'Bedload 1',
                'Bedload 1.4', 'Bedload 2', 'Bedload 2.8', 'Bedload 4',
                'Bedload 5.6', 'Bedload 8', 'Bedload 11.2', 'Bedload 16',
                'Bedload 22', 'Bedload 32', 'Bedload 45',
                # Grain counts
                'Count all', 'Count 0.5', 'Count 0.71', 'Count 1', 'Count 1.4',
                'Count 2', 'Count 2.8', 'Count 4', 'Count 5.6', 'Count 8',
                'Count 11.2', 'Count 16', 'Count 22', 'Count 32', 'Count 45',
                # Statistics
                'D10', 'D16', 'D25', 'D50', 'D75', 'D84', 'D90', 'D95', 'Dmax'
                ]
        Qs_kwargs = {
                'engine'    : 'python',
                'index_col' : None,
                'header'    : None,
                'dtype'     : None,
                'delimiter' : r'\s+', # any whitespace

                'names'     : Qs_column_names,
                }

        #period_dict = self.build_period_dict(sediment_flux_txt_files)
        #self.pickle_Qs_text_files(period_path, fnames, Qs_kwargs)

        try:
            data = pd.read_csv(filepath, **Qs_kwargs)
        except:
            print("Error reading {filepath}")
            raise

        print(filepath)
        print(data)
        assert False
        return data

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
