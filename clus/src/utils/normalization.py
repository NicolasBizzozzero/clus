"""
Read :
https://pdfs.semanticscholar.org/f211/dbbe5d8c00a004a286b8274e210dfea51a70.pdf

Other lecture :
https://en.wikipedia.org/wiki/Feature_scaling
https://stackoverflow.com/a/8717248
https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
http://www.optimaldesign.com/AMHelp/Other/zeroonemeannorm.htm
http://joelouismarino.github.io/blog_posts/blog_whitening.html
https://en.wikipedia.org/wiki/Whitening_transformation
"""

from typing import Callable

import numpy as np


class UnknownNormalization(Exception):
    def __init__(self, name: str):
        Exception.__init__(self, "The normalization : \"{method_name}\" doesn't exists".format(
            method_name=name
        ))


def rescaling(array: np.ndarray, floor: float = 0, ceil: float = 1) -> np.ndarray:
    """ Rescale the range of features to a given range. """
    arr_min = array.min(axis=0)
    array = (array - arr_min) / (array.max(axis=0) - arr_min)
    return array * (ceil - floor) + floor


def rescaling_(array: np.ndarray, floor: float = 0, ceil: float = 1) -> None:
    """ Rescale the range of features to a given range inplace. """
    arr_min = array.min(axis=0)
    arr_max = array.max(axis=0)
    array -= arr_min
    array *= ceil - floor
    np.divide(array, arr_max - arr_min, out=array)
    array += floor


def mean_normalization(array: np.ndarray) -> np.ndarray:
    """ Normalize the features to zero mean. """
    return (array - array.mean(axis=0)) / (array.max(axis=0) - array.min(axis=0))


def mean_normalization_(array: np.ndarray) -> None:
    """ Normalize the features inplace to zero mean. """
    arr_mean = array.mean(axis=0)
    arr_min = array.min(axis=0)
    arr_max = array.max(axis=0)
    array -= arr_mean
    np.divide(array.astype(np.float64), arr_max - arr_min, out=array)


def standardization(array: np.ndarray) -> np.ndarray:
    """ Normalize the features to zero mean and unit std. """
    return (array - array.mean(axis=0)) / array.std(axis=0)


def standardization_(array: np.ndarray) -> None:
    """ Normalize the features inplace to zero mean and unit std. """
    arr_mean = array.mean(axis=0)
    arr_std = array.std(axis=0)
    array -= arr_mean
    np.divide(array.astype(np.float64), arr_std, out=array)


def scaling_to_unit_length(array: np.ndarray, norm_p: int = 2) -> np.ndarray:
    """ Scale the components by dividing each features by their p-norm. """
    return array / np.sum(np.abs(array) ** norm_p, axis=0) ** 1 / norm_p


def scaling_to_unit_length_(array: np.ndarray, norm_p: int = 2) -> None:
    """ Scale the components inplace by dividing each features by their p-norm. """
    np.divide(array.astype(np.float64), np.sum(np.abs(array) ** norm_p, axis=0) ** 1 / norm_p, out=array)


def whitening(array: np.ndarray) -> np.ndarray:
    return array / array.std(axis=0)


def whitening_(array: np.ndarray) -> None:
    arr_std = array.std(axis=0)
    np.divide(array.astype(np.float64), arr_std, out=array)


def _str_to_normalization(string: str, inplace=True) -> Callable:
    string = string.lower()
    if string in ("rescaling",):
        return rescaling_ if inplace else rescaling
    elif string in ("mean", "mean_normalization",):
        return mean_normalization_ if inplace else mean_normalization
    elif string in ("standardization",):
        return standardization_ if inplace else standardization
    elif string in ("unit_length",):
        return scaling_to_unit_length_ if inplace else scaling_to_unit_length
    elif string in ("whitening",):
        return whitening_ if inplace else whitening
    else:
        raise UnknownNormalization(string)


if __name__ == "__main__":
    pass
