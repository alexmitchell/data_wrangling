# NOTE: This program only moves files from the input dir instead of copying.  
# Use another program (like rsync on linux) to safely copy all of the files 
# first if you don't want your original directory messed up.

# This program is intended to help move and rename common data files scattered 
# throughout several directory trees. You may need to manually move/rename 
# individual files that did not conform to standard naming practices.

"""
Usage:
    collapse_data_tree.py [--real-run]

Options:
    -- real-run
        Actually performs the moving and renaming of files. Without this flag, 
        this program defaults to a practice run so new people don't accidentally 
        mess up their data directories.
"""
import datetime
import re
import os
from os.path import join as p_join
from os.path import split as p_split
from collections import defaultdict


import docopt
args = docopt.docopt(__doc__)

input_dir = '/media/alex/Alex5/feed_timing_final/copies'
output_dir = '/media/alex/Alex5/feed_timing_final/data'
log_filepath = None

## Collapse info structure:
#collapse_info = {
#        'tree_root' : path_to_relevant_tree_root,
#        'renaming_info' : {
#            input_filename_pattern : {
#                're_groups' : OPTIONAL list of vars in input filename,
#                'dest_subdir' : output_subdir,
#                'dest_filename' : output_filename_pattern,
#                },
#            }
#        }


## Restructure light table data
"""
Move from input_dir:
    computer_2_Alex_backup
        lighttable-data
            extracted-lighttable-results
                [1A,1B,2A,2B,3A,3B,4A,5A]
                    falling-[62,75,87]L OR rising-[50,62,75,87,100]L
                        results-t[time]-t[time] (DIRECTORY)
Move to output_dir:
    light_table_data
        lt-labview-results_[exp_id]_[limb]-[discharge]L_t[time]-t[time] (DIRECTORY)
"""
lighttable_collapse_info = {
        'tree_root' : p_join(input_dir,
            'computer_2_Alex_backup',
            'lighttable-data',
            'extracted-lighttable-results',
            ),
        'tree_levels' : ('[exp_id]', '[limb]-[discharge]L'),
        'renaming_info' : {
            "results-t[time]-t[time]" : {
                're_groups' : ['t_start', 't_end'],
                'dest_subdir' : p_join('light_table_data', 'lt-labview-results'),
                'dest_filename': "lt-labview-results_[exp_id]_[limb]-[discharge]L_t[t_start]-t[t_end]",
                },
            ## directory below is exactly the same as 4A/rising-87L/results-t20-t40
            #"75L-accident-results-t20-t40" : {
            #    'dest_subdir' : p_join('light_table_data', 'lt-labview-results'),
            #    'dest_filename': "lt-labview-results_[exp_id]_[limb]-[discharge]L_t20-t40_accidental-75L",
            #    },
            "readme.txt" : {
                'dest_subdir' : p_join('light_table_data', 'lt-labview-results'),
                'dest_filename': "lt-labview-results_[exp_id]_readme.txt",
                },
            },
        }

## Restructure cart data
"""
Move from input_dir:
    computer_1_Alex_backup
        feed-timing-experiment-data
            cart-data
                [1A,1B,2A,2B,3A,3B,4A,5A]
                    falling-[62,75,87]L OR rising-[50,62,75,87,100]L
                        2m-photos
                            [exp_id]-[r,f][discharge]L-t[time]-2m-Composite.JPG
                        full-bed-photo
                            [exp_id]-[r,f][discharge]L-t60-8m-Composite.JPG
                        full-bed-laser
                            beddata.txt
                            [exp_id]-[r,f][discharge]L-t60-8m-beddatafinal.txt
                            [exp_id]-[r,f][discharge]L-t60-8m-laser.fig
                            [exp_id]-[r,f][discharge]L-t60-8m-laser.png
Move to output_dir:
    2m_photo_data
        2m-composite_[exp_id]_[limb]-[discharge]L_t[time].JPG
    8m_photo_data
        8m-composite_[exp_id]_[limb]-[discharge]L_t60.JPG
    8m_dem_data
        8m-raw-dem_[exp_id]_[limb]-[discharge]L_t60.txt
        8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.txt
        8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.fig
        8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.png
"""
cart_collapse_info = {
        'tree_root' : p_join(input_dir,
            'computer_1_Alex_backup',
            'feed-timing-experiment-data',
            'cart-data',
            ),
        'tree_levels' : ('[exp_id]', '[limb]-[discharge]L'),
        'renaming_info' : {
            "2m-photos/[exp_id]-[r,f][discharge]L-t[time]-2m-Composite.JPG" : {
                're_groups' : ['exp_id', 'limb', 'discharge', 't_end'],
                'dest_subdir' : '2m_photo_data',
                'dest_filename' : "2m-composite_[exp_id]_[limb]-[discharge]L_t[t_end].JPG",
                },
            "2m-photos/4A-r87L-t40-2m-Composite-75L-accident.JPG" : { # 75L accident
                'dest_subdir' : '2m_photo_data',
                'dest_filename' : "2m-composite_4A_rising-87L_t40_accidental-75L.JPG",
                },
            "2m-photos/Scan Information.txt" : {
                'dest_subdir' : '2m_photo_data',
                'dest_filename' : "2m-scan-information_[exp_id]_[limb]-[discharge]L.txt",
                },

            "full-bed-photo/[exp_id]-[r,f][discharge]L-t60-8m-Composite.JPG" : {
                'dest_subdir' : '8m_photo_data',
                'dest_filename' : "8m-composite_[exp_id]_[limb]-[discharge]L_t60.JPG",
                },
            "full-bed-photo/[exp_id]-[r,f][discharge]L-t60-8m-Composite-2nd-scan.JPG" : {
                'dest_subdir' : '8m_photo_data',
                'dest_filename' : "8m-composite_[exp_id]_[limb]-[discharge]L_t60.JPG",
                },
            "full-bed-photo/Scan Information.txt" : {
                'dest_subdir' : '8m_photo_data',
                'dest_filename' : "8m-scan-information_[exp_id]_[limb]-[discharge]L.txt",
                },

            "full-bed-laser/beddata.txt" : {
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-raw-dem_[exp_id]_[limb]-[discharge]L_t60.txt",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-t60-8m-beddatafinal.txt" : { # has '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.txt",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-t60-8m-laser.fig" : { # has '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.fig",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-t60-8m-laser.png" : { # has '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.png",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-8m-beddatafinal.txt" : { # missing '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.txt",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-8m-laser.fig" : { # missing '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.fig",
                },
            "full-bed-laser/[exp_id]-[r,f][discharge]L-8m-laser.png" : { # missing '-t60'
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-cleaned-dem_[exp_id]_[limb]-[discharge]L_t60.png",
                },
            "full-bed-laser/Scan Information.txt" : {
                'dest_subdir' : '8m_dem_data',
                'dest_filename' : "8m-scan-information_[exp_id]_[limb]-[discharge]L.txt",
                },
            },
        }

## Restructure manual data
"""
Move from input_dir:
    computer_1_Alex_backup
        feed-timing-experiment-data
            manual-data
                [1A,1B,2A,2B,3A,3B,4A,5A]
                    [exp_id]-masses.xlsx
                    [exp_id]-flow-depths.xlsx
                    SampleSieveData
                        [exp_id]_[r,f][discharge]_[time]min.xlsx
                        [exp_id]_[r,f][discharge]_[time]min_Sample[sample_n].xlsx
                        ? [exp_id]_[r,f][discharge]_[time]min_s[sample_n].xlsx
                        [exp_id]_feed_Sample[sample_n].xlsx

Move to output_dir:
    trap_totals_data
        trap-totals-data_[exp_id].xlsx
    flow_depth_data
        flow-depth-data_[exp_id].xlsx
    sieve_data
        sieve-data_[exp_id]_[limb]-[discharge]L_t[time]_sample-[1,2].xlsx
        sieve-data_[exp_id]_feed_sample-[1,2].xlsx
"""
manual_collapse_info = {
        'tree_root' : p_join(input_dir,
            'computer_1_Alex_backup',
            'feed-timing-experiment-data',
            'manual-data',
            ),
        'tree_levels' : ('[exp_id]',),
        'renaming_info' : {
            "[exp_id]-masses.xlsx" : {
                'dest_subdir' : "trap_totals_data",
                'dest_filename' : "trap-totals-data_[exp_id].xlsx"
                },
            "[exp_id]-flow-depths.xlsx" : {
                'dest_subdir' : "flow_depth_data",
                'dest_filename' : "flow-depth-data_[exp_id].xlsx"
                },

            "SampleSieveData/[exp_id]_[r,f][discharge]_[_*][t][time][min].xlsx" : { # No sample number given
                're_groups' : ['exp_id', 'limb', 'discharge', 't_end'],
                'dest_subdir' : "sieve_data",
                'dest_filename' : "sieve-data_[exp_id]_[limb]-[discharge]L_t[t_end]_sample-[sample_n].xlsx"
                },
            "SampleSieveData/[exp_id]_[r,f][discharge]_[_*][time]min_[S,s]ample[sample_n].xlsx" : { # includes sample number
                're_groups' : ['exp_id', 'limb', 'discharge', 't_end', 'sample_n'],
                'dest_subdir' : "sieve_data",
                'dest_filename' : "sieve-data_[exp_id]_[limb]-[discharge]L_t[t_end]_sample-[sample_n].xlsx"
                },
            "SampleSieveData/[exp_id]_[r,f][discharge]_[time]min_[S,s][sample_n].xlsx" : {
                're_groups' : ['exp_id', 'limb', 'discharge', 't_end', 'sample_n'],
                'dest_subdir' : "sieve_data",
                'dest_filename' : "sieve-data_[exp_id]_[limb]-[discharge]L_t[t_end]_sample-[sample_n].xlsx"
                },
            "SampleSieveData/4A_r87_40min_75acc.xlsx" : {
                'dest_subdir' : "sieve_data",
                'dest_filename' : "sieve-data_4A_rising-87L_t40_sample-1_accidental-75L.xlsx"
                },

            "SampleSieveData/[exp_id]_feed_[S,s]ample[sample_n].xlsx" : {
                're_groups' : ['exp_id', 'sample_n'],
                'dest_subdir' : "sieve_data",
                'dest_filename' : "sieve-data_[exp_id]_feed_sample-[sample_n].xlsx"
                },

            "masses-template.xlsx" : {
                'dest_subdir' : "misc", 'dest_filename' : "trap-totals_template.xlsx"
                },
            "template-flow-depths.xlsx" : {
                'dest_subdir' : "misc", 'dest_filename' : "flow-depths_template.xlsm"
                },
            "Macro-flow-depths.xlsm" : {
                'dest_subdir' : "misc", 'dest_filename' : "flow-depths_excel-macro.xlsm"
                },
            "SieveDataCleaningMacro.xlsm" : {
                'dest_subdir' : "misc", 'dest_filename' : "sieve-data_excel-macro.xlsm"
                },
            },
        }

def codestr2filename(filename_str, **vars_dict):
    # Convert a filename string with my custom formatting codes to a valid 
    # filename by filling in my custom variables.
    vars_in_name = re.findall('\[(.*?)\]', filename_str)
    log_print(f"Constructing the output filename from '{filename_str}'...")
    log_print(f"    Which has the variables {vars_in_name}")
    for var_name in vars_in_name:
        # Replace the codes with the actual values
        assert var_name in vars_dict, f"Invalid variable code {var_name}"
        if var_name == 'sample_n' and vars_dict['sample_n'] is None:
            vars_dict['sample_n'] = 1
            log_print(f"    Using a default value of 1 for 'sample_n'.")
        assert vars_dict[var_name] is not None, f"Variable code {var_name} has no value yet"
        filename_str = filename_str.replace(f'[{var_name}]', str(vars_dict[var_name]))

    log_print(f"    Filename after substitution:")
    log_print(f"      '{filename_str}'")

    return filename_str

def codestr2re(target_str):
    # Convert my custom formatting codes to a compiled regex pattern
    regex_replacements = {
            # my_format_code : regex_command
            '[exp_id]' : r'([1-5][AB])',
            '[limb]' : r'(rising|falling)',
            '[r,f]' : r'(r|f)',
            '[discharge]' : r'(\d{2,})',
            #'t[time]' : r't(\d{2})',
            #'[time]min' : r'(\d{2})min', 
            '[S,s]' : r'[Ss]',
            '[sample_n]' : r'\#?(\d)',
            'sample_[1,2]' : r'sample_([12])',
            # Special cases:
            '[_*]' : r'_*?',
            '[t]' : r't??',
            '[time]' : r'(\d{2})',
            '[min]' : r'(?:min)??',
            }
    target_cmd = target_str
    for str_cmd, regex_cmd in regex_replacements.items():
        target_cmd = target_cmd.replace(str_cmd, regex_cmd)
    return re.compile(target_cmd)


def identify_moves(collapse_info):
    log_print_breaks(2)
    log_print("Identifying new set of files in order to collapse a tree")
    log_print_collapse_info(collapse_info)
    log_print_breaks(2)

    skipped_list = []
    # [skipped_file.txt, skipped_file.jpg, skipped_dir, ...],
    move_info = {}
    # {
    #   old_path_a : new_path_a,
    #   old_path_b : new_path_b,
    #   ...
    # }

    # Build a dict relating my coded filenames to regex patterns
    # Note, dict values will be a list of regex patterns representing a 
    # sequence of subdirs that the target file must be in.
    regex_patterns_lists = defaultdict(list)
    log_print(f"Building regex lists")
    log_print()
    for target_str in collapse_info['renaming_info'].keys():
        log_print(f"    Target string: '{target_str}'")
        for substring in target_str.split(sep='/'):
            regex_cmd = codestr2re(substring)
            log_print(f"      '{substring}' ==> {regex_cmd}")
            regex_patterns_lists[target_str].append(regex_cmd)
        log_print()
    log_print_breaks(2)

    # Walk the directory tree starting at 'tree_root'
    for parent_dir, dirs, files in os.walk(collapse_info['tree_root'], topdown=True):
        # Check all files matching names
        for filename in files:
            log_print(f"Checking a file at {parent_dir}: {filename}")#{p_join(parent_dir, filename)}")
            old_path, new_path = check_for_match(
                    collapse_info, regex_patterns_lists,
                    parent_dir, filename
                    )
            if new_path is None:
                # File does not match any patterns, record that it is skipped.
                # i.e. File should eventually be deleted
                log_print(f"Skipping file at relative path: {old_path}")
                skipped_list.append(old_path)
            else:
                move_info[old_path] = new_path
            log_print_breaks()

        # Check all directories for matching names
        for dirname in dirs.copy():
            log_print(f"Checking a dir at {parent_dir}: {dirname}")#{p_join(parent_dir, dirname)}")
            old_path, new_path = check_for_match(
                    collapse_info, regex_patterns_lists,
                    parent_dir, dirname,
                    )
            if new_path is not None:
                # The directory matches a pattern, remove it from the walk 
                # because this entire directory will be moved.
                log_print(f"Directory matched, removing {old_path} from walk")
                dirs[:] = [d for d in dirs if d != dirname]
                log_print(f"    Remaining subdirs: {dirs}")
                move_info[old_path] = new_path

            elif not (dirs or files):
                # Directory does not match a pattern but is empty.
                # i.e. It is a tree leaf that should eventually be deleted
                log_print(f"Skipping empty directory at relative path: {old_path}")
                skipped_list.append(old_path)

            else:
                log_print(f"Intermediate directory: {old_path}")
                # This is an intermediate directory in the tree. Do nothing 
                # with it and keep walking through the tree.
            log_print_breaks()

    log_print_breaks(2)

    return move_info, skipped_list

def check_for_match(collapse_info, regex_patterns_lists, parent_dir, os_obj):
    # Check if the file/dir matches any of the provided patterns

    log_print()

    # Get the relative path in the tree for the existing file/dir
    old_absolute_path = p_join(parent_dir, os_obj)
    old_relative_path = os.path.relpath(
            old_absolute_path, collapse_info['tree_root']
            )

    # Check if the file/dir matches any of the provided patterns
    new_relative_path = None
    for target_str, target_patterns in regex_patterns_lists.items():
        new_relative_path = None
        
        # Check patterns starting with the last one (typically the file)
        # i.e. if the file matches, then check if any required parents match 
        # too
        all_matches = {}
        unhandled_path = p_join(parent_dir, os_obj)
        for target_pattern in reversed(target_patterns):
            current_parent, current_obj = p_split(unhandled_path)
            match = target_pattern.fullmatch(current_obj)
            if match:
                log_print(f"Matched! '{current_obj}' and '{target_pattern}'")
                all_matches[target_pattern] = match
                unhandled_path = current_parent
            else:
                log_print(f"Not matched! '{current_obj}' and '{target_pattern}'")
                break

        if len(all_matches) == len(target_patterns):
            # Found a match for the leaf and all required parents
            # Build the new filename (or dirname)
            log_print()
            new_filename = handle_match(
                    collapse_info, 
                    unhandled_path, all_matches,
                    target_str, target_patterns,
                    )

            # Build the new path
            dest_subdir = collapse_info['renaming_info'][target_str]['dest_subdir']
            new_relative_path = p_join(dest_subdir, new_filename)
            log_print()
            log_print(f"Input  relative path: {old_relative_path}")
            log_print(f"Output relative path: {new_relative_path}")
            break

    log_print()

    return old_absolute_path, new_relative_path

def handle_match(collapse_info, unhandled_parent_dir, all_matches, target_str, target_patterns):
    vars_dict = {
            'exp_id' : None,
            'limb' : None,
            'discharge' : None,
            't_start' : None,
            't_end' : None,
            'sample_n' : None,
            }
    log_print(f"Handling a full match at:")
    log_print(f"    Leaf file/dir: '{all_matches[target_patterns[-1]].group(0)}'")
    for tp in reversed(target_patterns[:-1]):
        log_print(f"    In required dir: '{all_matches[tp].group(0)}'")
    log_print(f"    In tree directory: {unhandled_parent_dir}")
    log_print()

    # Get variable values from the tree path
    # 'tree_levels' : ('[exp_id]', '[limb]-[discharge]L'),
    temp_parent_dir = unhandled_parent_dir
    log_print(f"Searching directory tree levels for variables...")
    for level_str in reversed(collapse_info['tree_levels']):
        # Get the parent's name
        path_parts = p_split(temp_parent_dir)
        parent_dirname = path_parts[1]

        # Parse the parent's name for variables
        vars_in_name = re.findall('\[(.*?)\]', level_str)
        log_print(f"    Tree level '{level_str}':")
        parent_match = codestr2re(level_str).fullmatch(parent_dirname)
        #assert parent_match
        if parent_match:
            # Parent's dirname matches the expected level
            temp_parent_dir = path_parts[0] # Safe to remove parent's dirname from path
            for var_idx, var_str in enumerate(vars_in_name):
                assert var_str in vars_dict
                vars_dict[var_str] = parent_match.group(var_idx + 1)
                log_print(f"        '{var_str}' = '{vars_dict[var_str]}'")
        else:
            log_print(f"        Could not find {level_str} in {parent_dirname}")
            # Try again with the same temp_parent_dir path

    # Get variables from the filename when needed
    renaming_info = collapse_info['renaming_info'][target_str]
    if 're_groups' in renaming_info:
        filename_vars = renaming_info['re_groups']
        log_print()
        log_print(f"Searching existing filename for variables ('{target_str}'):")
        log_print(f"    Expected number of groups: {len(filename_vars)}")
        log_print(f"      Expected vars: {filename_vars}")
        log_print(f"    Actual number of groups: {sum([tp.groups for tp in target_patterns])}")
        for tp in target_patterns:
            log_print(f"      {tp.groups} groups in {tp}")
        assert sum([tp.groups for tp in target_patterns]) == len(filename_vars)

        log_print()
        temp_filename_vars = filename_vars.copy()
        for target_pattern in target_patterns:
            vars_for_this_pattern = temp_filename_vars[0:target_pattern.groups]
            temp_filename_vars = temp_filename_vars[target_pattern.groups:]
            log_print(f"    Pattern {target_pattern} has vars: {vars_for_this_pattern}")
            for group_idx, var_str in enumerate(vars_for_this_pattern):
                assert var_str in vars_dict
                regex_match = all_matches[target_pattern]
                vars_dict[var_str] = regex_match.group(group_idx + 1)
                log_print(f"        '{var_str}' = '{vars_dict[var_str]}'")
    log_print()

    # Change some variables so output names will be consistent.
    if vars_dict['limb'] == 'r':
        vars_dict['limb'] = 'rising'
        log_print(f"Changing 'limb' from 'r' to 'rising'")
    elif vars_dict['limb'] == 'f':
        vars_dict['limb'] = 'falling'
        log_print(f"Changing 'limb' from 'f' to 'falling'")
    log_print()

    log_print(f"Variables found in tree levels and old filename: \n    {vars_dict}")
    log_print()

    final_filename = codestr2filename(renaming_info['dest_filename'], **vars_dict)
    return final_filename


def double_check_uniqueness(all_moves):
    # Check that all the new filepaths will be unique.
    # I think just flipping the dictionary should be enough?
    
    log_print("", True)
    log_print(f"Checking that new paths are unique...", True)
    flipped_dict = {}
    all_unique = True
    for old_path, new_path in all_moves.items():
        if new_path not in flipped_dict:
            flipped_dict[new_path] = old_path
        else:
            all_unique = False
            log_print(f"{new_path} is not unique!")

    log_print(f"{'    Passed!' if all_unique else '    Failed!'}", True)
    log_print("", True)
    assert all_unique


def move_files(all_moves):
    # Check that all the new filepaths will be unique.
    # I think just flipping the dictionary should be enough?
    
    print_to_terminal = False
    
    log_print("Moving files...", True)
    log_print(f"Root dir:  {os.path.commonpath([input_dir, output_dir])}", print_to_terminal)
    for old_relative_path, new_relative_path in all_moves.items():
        old_path = p_join(input_dir, old_relative_path)
        new_path = p_join(output_dir, new_relative_path)

        common_path = os.path.commonpath([old_path, new_path])
        old_path_branch = os.path.relpath(old_path, common_path)
        new_path_branch = os.path.relpath(new_path, common_path)

        log_print(f"From: {old_path_branch}", print_to_terminal)
        log_print(f"  To: {new_path_branch}", print_to_terminal)

        isfile =  os.path.isfile(old_path)
        isdir =  os.path.isdir(old_path)
        assert isfile or isdir # Make sure the object exists!
        
        os.rename(old_path, new_path)

    log_print("", True)
    log_print(f"Finished moving files!", True)
    log_print("", True)


def setup_log_file(is_practice):
    intro_msg = ""
    if is_practice:
        intro_msg = '\n'.join(["Executing a practice run",
            "    Use '--real-run' to actually move and rename files.",
            "".join([
                "    However, before executing a real run, please check the ",
                "'Skipping' and 'Moving' sections at the end of the log file ",
                "to make sure everything is correct."
                ]),
            "",
            ])
        log_filename = f'collapse_data_tree_log_practice.txt'
        # Note practice log files overwrite

    else:
        intro_msg = "Executing a real run"
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S.%f")
        log_filename = f'collapse_data_tree_log_{timestamp_str}.txt'

    # Create/clear the log file
    # Currently puts the log file in './logs/[log_filename].txt'
    global log_filepath
    log_filepath = p_join('logs', log_filename)
    with open(log_filepath, 'w') as f:
        f.write('')
    log_print_breaks(2)
    log_print(intro_msg, True)
    log_print(f"    Saving to log file:", True)
    log_print(f"      {os.path.realpath(log_filepath)}", True)
    log_print('', True)
    log_print(f"Moving data from:  {input_dir}", True)
    log_print(f"Moving data to  :  {output_dir}", True)

def log_print(msg='', terminal_print=False):
    assert log_filepath is not None
    assert os.path.isfile(log_filepath)
    with open(log_filepath, 'a') as f:
        f.write(msg + '\n')

    if terminal_print:
        print(msg)

def log_print_breaks(n=1):
    log_print()
    log_print('\n'.join(['#'*80]*n))
    log_print()

def log_print_collapse_info(collapse_info):
    tree_root = collapse_info['tree_root']
    tree_levels = collapse_info['tree_levels']
    renaming_info = collapse_info['renaming_info']

    log_print("Collapse info:")
    log_print(f"    Root: {tree_root}")
    log_print(f"    Tree levels: {tree_levels}")
    for target_str, target_info in renaming_info.items():
        log_print(f"    Target string: '{target_str}'")
        if 're_groups' in target_info:
            log_print(f"        Target regex groups: {target_info['re_groups']}")
        log_print(f"        Dest subdir: {target_info['dest_subdir']}")
        log_print(f"        Dest filename: '{target_info['dest_filename']}'")
    log_print()


if __name__ == '__main__':
    timestamp = datetime.datetime.now()
    is_practice = not args['--real-run']
    setup_log_file(is_practice)

    all_collapse_info = [
            lighttable_collapse_info,
            cart_collapse_info,
            manual_collapse_info,
            ]

    # Find all the relevant files and identify where they will go
    all_moves = {}
    all_skipped = []
    for collapse_info in all_collapse_info:
        valid_moves, skipped_list = identify_moves(collapse_info)
        all_moves.update(valid_moves)
        all_skipped.extend(skipped_list)

    log_print_breaks(2)
    log_print("Skipping:")
    for skipped_path in all_skipped:
        log_print(f"  {skipped_path}")

    log_print_breaks(2)
    log_print("Planning to move:")
    for key, val in all_moves.items():
        log_print(f"{key}  ==>  {val}")

    log_print_breaks(2)
    double_check_uniqueness(all_moves)
    log_print_breaks(2)

    if is_practice:
        log_print("Finished with practice run. Please check log file.", True)

    else:
        if input("Are you sure you want to move data files? (y/n) ") == 'y':
            # For a real run, actually move the files...
            move_files(all_moves)
        else:
            log_print("On second thought.... aborting!", True)
