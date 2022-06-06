#!/usr/bin/env python3

from time import asctime
import os.path
from pandas import option_context as pd_option_context

# Removing helpyr dependency
#from helpyr.helpyr_misc import ensure_dir_exists

# To do:
# Create an archive file that accumulates all runs and change main file to be 
# overwritten every run.

def ensure_dir_exists(dir_path, logger=None):
    # Creates directory if not already created
    # Return True if directory created or False if directory already 
    # exists.
    # logger is a Logger object (found in Helpyr)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        msg = "Making directory at {}.".format(dir_path)
        if logger is None:
            print(msg)
        else:
            logger.write([msg])
        return True
    return False


class Logger:

    def __init__(self, log_filepath="./log-crawler.txt", default_verbose=True, no_log=False):
        self.verbose = default_verbose
        self.global_indent = 0
        self.indent_str = 4*' '
        self.no_log = no_log or log_filepath is None or "" == log_filepath
        self.start_time = asctime()

        if self.no_log:
            print("Logger is set to not log")

        else:
            self.log_filepath = log_filepath
            ensure_dir_exists(os.path.split(log_filepath)[0], logger=self)

        self.write_section_break()
        self.write_section_break()
        self.write(f"Begin Logger run output at {self.start_time}")

    def begin_output(self, name):
        self.write_section_break()
        self.write([f"Begin {name} output", asctime()])

    def end_output(self):
        self.write([f"Search for '{self.start_time}' to find beginning of run"])
        self.write_section_break()


    def increase_global_indent(self):
        self.global_indent += 1

    def decrease_global_indent(self):
        if self.global_indent >= 1:
            self.global_indent -= 1


    def write_section_break(self):
        self.write([80*'#', 80*' '])

    def write(self, messages, verbose=None, local_indent=0):
        # Write a message to the log file
        if messages is None:
            # don't print anything if message is None
            return

        if isinstance(messages, str):
            # Turn messages into a list of strings
            messages = [messages]

        if isinstance(messages, list):
            msg_indent = (self.global_indent + local_indent) * self.indent_str
        else:
            # If it isn't a list or a string, get the string representation of 
            # of whatever the object is.
            #
            # Don't automatically indent bc its likely a complicated string 
            # e.g. a pandas dataframe or numpy array
            messages = [messages.__str__()]
            msg_indent = ''

        for message in messages:
            message = msg_indent + message
            if not self.no_log:
                with open(self.log_filepath, 'a') as lf:
                    lf.write(message + '\n')

            if (verbose is None and self.verbose) or verbose:
                print(message)

    def write_greeting(self, message):
        # writes greeting message plus a timestamp
        self.write([message, asctime()])

    def write_blankline(self, n=1, verbose=None):
        self.write(['']*n, verbose=verbose)

    def write_dataframe(self, dataframe, name='', float_formatter=None):
        df_str = ""
        #if context_args is None:
        #    context_args = ['display.max_rows', 10000,
        #                    'display.max_columns', 10000,
        #                    'display.expand_frame_repr', False,
        #                    'display.precision', 3,
        #                    ]
        #with pd_option_context(*context_args):
        #    df_str = dataframe.__repr__()
        def fformatter(val):
            return "{:13.3f}".format(val)
        formatter = fformatter if float_formatter is None else float_formatter
        df_str = dataframe.to_string(float_format=formatter)
        if name:
            self.write(name)
        self.write(df_str.split('\n'), local_indent=1)

    def run_indented_function(self, function, kwargs=None, before_msg=None, after_msg=None):
        # Runs the provided function with automatically indented internal log 
        # messages.
        # Function takes only keyworded args
        self.write(before_msg)
        self.increase_global_indent()
        if kwargs is not None:
            out = function(**kwargs)
        else:
            out = function()
        self.decrease_global_indent()
        self.write(after_msg)
        self.write_blankline()

        return out

    def warning(self, messages):
        # Display warning messages. The first message will be the warning 
        # title.
        warning_msg = "Warning: "
        self.write(warning_msg + messages[0] + ":")
        self.write(messages[1:], local_indent=1)
