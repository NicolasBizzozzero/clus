import os

import numpy as np
import pandas as pd

from clus.src.utils.decorator import remove_unexpected_arguments

ALIASES_GUESS = ("guess",)
ALIASES_CSV = ("csv",)
ALIASES_NPY = ("npy",)
ALIASES_NPZ = ("npz",)

EXTENSION_CSV = ("csv", "tsv", "dsv")
EXTENSION_NPY = ("npy",)
EXTENSION_NPZ = ("npz",)


class UnknownDataLoadingMethod(Exception):
    def __init__(self, method_name):
        Exception.__init__(self, "The data loading method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


class CannotGuessFileType(Exception):
    def __init__(self, file_name):
        Exception.__init__(self, "The file type of \"{file_name}\" cannot be guessed.".format(
            file_name=file_name
        ))


def load_data(path_file, file_type, delimiter, header, array_name):
    strategy = _str_to_dataloading(file_type)
    return strategy(path_file=path_file, array_name=array_name, delimiter=delimiter, header=header)


@remove_unexpected_arguments
def guess(path_file, array_name, delimiter, header):
    file_name, extension = os.path.splitext(path_file)
    extension = extension.lower()

    if extension in EXTENSION_CSV:
        return csv(path_file=path_file, delimiter=delimiter, header=header)
    if extension in EXTENSION_NPY:
        return npy(path_file=path_file)
    if extension in EXTENSION_NPZ:
        return npz(path_file=path_file, array_name=array_name)
    raise CannotGuessFileType(file_name=file_name)


@remove_unexpected_arguments
def csv(path_file, delimiter, header):
    return pd.read_csv(path_file, delimiter=delimiter, header=0 if header else None).values


@remove_unexpected_arguments
def npy(path_file):
    return np.load(path_file)


@remove_unexpected_arguments
def npz(path_file, array_name):
    assert array_name is not None, "You need to pass the --array-name option for the NPZ file type."

    return np.load(path_file)[array_name]


def _str_to_dataloading(string):
    global ALIASES_GUESS, ALIASES_CSV, ALIASES_NPY, ALIASES_NPZ

    string = string.lower()
    if string in ALIASES_GUESS:
        return guess
    if string in ALIASES_CSV:
        return csv
    if string in ALIASES_NPY:
        return npy
    if string in ALIASES_NPZ:
        return npz
    raise UnknownDataLoadingMethod(string)


if __name__ == "__main__":
    pass
