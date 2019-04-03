import os
from typing import Callable

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
    def __init__(self, method_name: str):
        Exception.__init__(self, "The data loading method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


class CannotGuessFileType(Exception):
    def __init__(self, file_name: str):
        Exception.__init__(self, "The file type of \"{file_name}\" cannot be guessed.".format(
            file_name=file_name
        ))


def data_loading(path_file: str, strategy: str, delimiter: str, header: bool, array_name: str) -> np.ndarray:
    strategy = _str_to_dataloading(strategy)
    return strategy(path_file=path_file, array_name=array_name, delimiter=delimiter, header=header)


@remove_unexpected_arguments
def guess(path_file: str, array_name: str, delimiter: str, header: bool) -> np.ndarray:
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
def csv(path_file: str, delimiter: str, header: bool) -> np.ndarray:
    return pd.read_csv(path_file, delimiter=delimiter, header=0 if header else None).values


@remove_unexpected_arguments
def npy(path_file: str) -> np.ndarray:
    return np.load(path_file)


@remove_unexpected_arguments
def npz(path_file: str, array_name: str) -> np.ndarray:
    return np.load(path_file)[array_name]


def random_gaussian(data: np.ndarray, components: int) -> np.ndarray:
    return np.random.normal(loc=data.mean(axis=0), scale=data.std(axis=0),
                            size=(components, data.shape[0])).astype(np.float64)


def random_choice(data: np.ndarray, components: int) -> np.ndarray:
    assert data.shape[0] >= components, ("Cannot take a number of components larger than the number of samples with thi"
                                         "s initialization method")

    idx = np.random.choice(np.arange(data.shape[0]), size=components, replace=False)
    return data[idx, :]


def _str_to_dataloading(string: str) -> Callable:
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
