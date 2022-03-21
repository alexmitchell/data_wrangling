#!/usr/bin/env python3

import pytest
import os.path
from io import StringIO
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data_wrangling import DataCollectorBase

class FakeCollector(DataCollectorBase):

    def __init__(self, data_root_dir, output_dir):
        super().__init__(data_root_dir, output_dir)

    def is_my_file(self, filepath):
        return 'fake_data' in filepath

    def load_data_file(self, filepath):
        # returns data as a pandas dataframe

        assert self.is_my_file(filepath)
        return pd.read_csv(filepath, index_col=0,)

    def combine_data(self, data_dict):
        # Returns a pandas dataframe containing all the entries in data_dict 
        # {filename:pd_data}.

        concat_data = pd.concat(data_dict.values())
        concat_data.sort_index(inplace=True)

        return concat_data

    def save_file(self, data, extension):
        # save the data to a file with the extension
        if extension == 'csv':
            fpath = self.output_dir / "fake_output_data.csv"
            data.to_csv(fpath)
            return fpath
        else:
            raise NotImplementedError

def generate_fake_data_files(dir, data_dict):
    # Data_dict is {'data_name' : 'data string'}
    # data_name will be used as the filename
    # data string should include commas and newlines.
    # (e.g. "1,2,3\n4,5,6"
    filepaths = []
    for name, data in data_dict.items():
        if os.path.splitext(name)[1] == '':
            # Missing an extension
            filename = f"{name}.csv"
        else:
            filename = name
        filepath = os.path.join(dir, filename)

        with open(filepath, 'w') as f:
            f.writelines(data)

        filepaths.append(filepath)

    return filepaths


def test_testing_functions(tmp_path):
    data_dict = {
            'data_A' : '\n'.join([
                "1,2,3",
                "4,5,6",
                "7,8,9",
                ]),
            'data_B' : '\n'.join([
                "11,12,13",
                "14,15,16",
                "17,18,19",
                ]),
            }
    filepaths = generate_fake_data_files(tmp_path, data_dict)

    for filepath in filepaths:
        filename = os.path.split(filepath)[-1]
        name = filename[:-4]
        with open(filepath, 'r') as f:
            #print(f"File {filepath}")
            #for line in f.read().splitlines():
            #    print(line)
            file_text = f.read()

        assert data_dict[name] == file_text
        #print()

def test_is_my_file(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    data_dict = {
            'fake_data_A' : '\n'.join([
                "1,2,3",
                "4,5,6",
                "7,8,9",
                ]),
            'fake_data_B' : '\n'.join([
                "11,12,13",
                "14,15,16",
                "17,18,19",
                ]),
            }
    filepaths = generate_fake_data_files(input_dir, data_dict)

    fake_collector = FakeCollector(input_dir, output_dir)
    
    for filepath in filepaths:
        assert fake_collector.is_my_file(filepath)

    assert not(fake_collector.is_my_file('other_data'))

def test_gather_filepaths(tmp_path):
    answer_filepaths = [
            tmp_path.joinpath("A", "B", "C", "fake_data_1.txt"),
            tmp_path.joinpath("A", "B", "C", "fake_data_2.txt"),
            tmp_path.joinpath("A", "B", "D", "fake_data_3.txt"),
            tmp_path.joinpath("A", "E", "C", "fake_data_4.txt"),
            ]
    incorrect_filepaths = [
            tmp_path.joinpath("A", "E", "C", "incorrect_data.txt"),
            ]
    # Create the directory tree and empty files
    for fpath in answer_filepaths + incorrect_filepaths:
        # Make the directory
        fpath.parent.mkdir(parents=True, exist_ok=True)

        # Make an empty file
        fpath.touch()

    fake_collector = FakeCollector(tmp_path, tmp_path)
    gathered_paths = fake_collector.gather_filepaths()

    # Check that each answer path is somewhere in the gathered_paths
    # Not the most efficient method.... But there are only four files here, so 
    # only 16 comparisons.
    for apath in answer_filepaths:
        assert any([apath.samefile(gpath) for gpath in gathered_paths])

    for ipath in incorrect_filepaths:
        assert not any([ipath.samefile(gpath) for gpath in gathered_paths])
def test_load_data_file(tmp_path):
    # Test that files are loaded correctly individually
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    data_dict = {
            'fake_data_A' : '\n'.join([
                " ,A,B,C",
                "a,1,2,3",
                "b,4,5,6",
                "c,7,8,9",
                ]),
            'fake_data_B' : '\n'.join([
                " ,A,B,C",
                "d,11,12,13",
                "e,14,15,16",
                "f,17,18,19",
                ]),
            }
    filepaths = generate_fake_data_files(input_dir, data_dict)

    fake_collector = FakeCollector(input_dir, output_dir)
    
    # Test loading data individually
    for filepath in filepaths:
        pd_data = fake_collector.load_data_file(filepath)

        cols = pd_data.columns.values
        index = pd_data.index.values
        np_data = pd_data.values

        assert np.array_equal(cols, ['A','B','C'])
        if 'fake_data_A' in filepath:
            assert np.array_equal(index, ['a','b','c'])
            assert np.array_equal(np_data, [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                ])
        if 'fake_data_B' in filepath:
            assert np.array_equal(index, ['d','e','f'])
            assert np.array_equal(np_data, [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
                ])

def test_load_data_files(tmp_path):
    # Check that the dictionary from load_data_files is correct
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    data_dict = {
            'fake_data_A.csv' : '\n'.join([
                " ,A,B,C",
                "a,1,2,3",
                "b,4,5,6",
                "c,7,8,9",
                ]),
            'fake_data_B.csv' : '\n'.join([
                " ,A,B,C",
                "d,11,12,13",
                "e,14,15,16",
                "f,17,18,19",
                ]),
            }
    filepaths = generate_fake_data_files(input_dir, data_dict)

    fake_collector = FakeCollector(input_dir, output_dir)
    loaded_data_dict = fake_collector.load_data_files(filepaths)

    # Check that the entry names are the exact same
    assert all([fname in loaded_data_dict for fname in data_dict])
    assert all([fname in data_dict for fname in loaded_data_dict])
    
    # Check that the data loaded correctly
    for fname, str_data in data_dict.items():
        pd_answer = pd.read_csv(StringIO(str_data), index_col=0)
        pd_data = loaded_data_dict[fname]
        assert_frame_equal(pd_answer, pd_data)

def test_combine_data(tmp_path):

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    data_dict = {
            'fake_data_A' : '\n'.join([
                ",A,B,C",
                "a,1,2,3",
                "b,4,5,6",
                "c,7,8,9",
                ]),
            'fake_data_B' : '\n'.join([
                ",A,B,C",
                "d,10,11,12",
                "e,13,14,15",
                "f,16,17,18",
                ]),
            }
    pd_answer = pd.DataFrame(
            np.arange(1,19).reshape(6,3),
            columns=["A","B","C"],
            index=['a', 'b', 'c', 'd', 'e', 'f'],
            )
    filepaths = generate_fake_data_files(input_dir, data_dict)

    fake_collector = FakeCollector(input_dir, output_dir)
    loaded_data_dict = fake_collector.load_data_files(filepaths)
    combined_data = fake_collector.combine_data(loaded_data_dict)

    assert_frame_equal(pd_answer, combined_data)

def test_run_collector(tmp_path):

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    data_file_dict = {
            input_dir.joinpath("A", "B", "C", "fake_data_A.csv") : pd.DataFrame(
                np.arange(1,10).reshape(3,3),
                columns=["A","B","C"],
                index=['a', 'b', 'c'],
                ),
            input_dir.joinpath("A", "B", "C", "fake_data_B.csv"): pd.DataFrame(
                np.arange(10,19).reshape(3,3),
                columns=["A","B","C"],
                index=['d', 'e', 'f'],
                ),
            input_dir.joinpath("A", "B", "D", "fake_data_C.csv"): pd.DataFrame(
                np.arange(19,28).reshape(3,3),
                columns=["A","B","C"],
                index=['g', 'h', 'i'],
                ),

            input_dir.joinpath("A", "E", "C", "fake_data_D.csv"): pd.DataFrame(
                np.arange(28,37).reshape(3,3),
                columns=["A","B","C"],
                index=['j', 'k', 'l'],
                ),
            }
    answer_data = pd.DataFrame(
            np.arange(1,37).reshape(12,3),
            columns=["A","B","C"],
            index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            )
    # Create the directory tree and data files
    for fpath, pd_data in data_file_dict.items():
        # Make the directory
        fpath.parent.mkdir(parents=True, exist_ok=True)

        # Make the data file
        pd_data.to_csv(fpath)

    fake_collector = FakeCollector(input_dir, output_dir)

    collector_data = fake_collector.run_collector()

    print("Answer data:")
    print(answer_data)
    print()
    print("Collector data:")
    print(collector_data)
    print()
    assert_frame_equal(answer_data, collector_data)

    print("Saved collector data:")
    saved_filepath = fake_collector.run_collector('csv')
    print(saved_filepath)
    saved_data = pd.read_csv(saved_filepath, index_col=0)
    print(saved_data)
    assert_frame_equal(answer_data, saved_data)


@pytest.mark.parametrize("outcome,filepath,pattern",[
    # Matching
    (True, '/a/b/c/matching_text_A.txt', 'matching_text_?.txt'),
    (True, '/a/b/c/matching_text_A.txt', 'matching_text_A.txt'),
    (True, '/a/d/matching_text_B.txt'  , 'matching_text_?.txt'),
    (True, 'matching_text_C.txt'       , 'matching_text_?.txt'),
    (True, 'matching_text_D.txt'       , ['matching_text_?.txt', 'other_pattern']),
    (True, 'matching_text_D.txt'       , ['matching_text_?.txt', '*.txt']),
    (True, 'matching_text_D.txt'       , ['*.txt', '*.tex']),
    (True, '/a/b/c/other_text.tex'     , ['*.txt', '*.tex']),

    # Not matching
    (False, '/a/b/c/other_text_A.txt'   , 'matching_text_?.txt'),
    (False, '/a/d/other_text_B.txt'     , 'matching_text_?.txt'),
    (False, 'other_text_C.txt'          , 'matching_text_?.txt'),
    (False, '/a/b/c/matching_text_A.tex', 'matching_text_?.txt'),
    (False, '/a/d/matching_text_B.tex'  , 'matching_text_?.txt'),
    (False, 'matching_text_C.tex'       , 'matching_text_?.txt'),
    (False, 'some_text_C.tex'           , ['spam.txt', 'eggs.txt']),
    ])

def test_file_match_by_name(tmp_path, filepath, pattern, outcome):
    fake_collector = FakeCollector(tmp_path, tmp_path)

    assert outcome == fake_collector.file_match_by_name(filepath, pattern)

