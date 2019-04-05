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

import numpy as np


ALIASES_RESCALING = ("rescaling",)
ALIASES_MEAN_NORMALIZATION = ("mean", "mean_normalization",)
ALIASES_STANDARDIZATION = ("standardization",)
ALIASES_SCALE_TO_UNIT_LENGTH = ("unit_length",)
ALIASES_WHITENING_ZCA = ("whitening_zca", "zca")
ALIASES_WHITENING_PCA = ("whitening_pca", "pca")
ALIASES_WHITENING_CHOLESKY = ("whitening_cholesky", "cholesky")
ALIASES_WHITENING_ZCA_COR = ("whitening_zca_cor", "zca_cor")
ALIASES_WHITENING_PCA_COR = ("whitening_pca_cor", "pca_cor")


class UnknownNormalization(Exception):
    def __init__(self, name):
        Exception.__init__(self, "The normalization : \"{method_name}\" doesn't exists".format(
            method_name=name
        ))


def normalization(array, strategy):
    strategy = _str_to_normalization(strategy, inplace=True)
    strategy(array)


def rescaling(array, floor=0, ceil=1):
    """ Rescale the range of features to a given range. """
    arr_min = array.min(axis=0)
    array = (array - arr_min) / (array.max(axis=0) - arr_min)
    return array * (ceil - floor) + floor


def rescaling_(array, floor=0, ceil=1):
    """ Rescale the range of features to a given range inplace. """
    arr_min = array.min(axis=0)
    arr_max = array.max(axis=0)
    array -= arr_min
    array *= ceil - floor
    np.divide(array, arr_max - arr_min, out=array)
    array += floor


def mean_normalization(array):
    """ Normalize the features to zero mean. """
    return (array - array.mean(axis=0)) / (array.max(axis=0) - array.min(axis=0))


def mean_normalization_(array):
    """ Normalize the features inplace to zero mean. """
    arr_mean = array.mean(axis=0)
    arr_min = array.min(axis=0)
    arr_max = array.max(axis=0)
    array -= arr_mean
    np.divide(array.astype(np.float64), arr_max - arr_min, out=array)


def standardization(array):
    """ Normalize the features to zero mean and unit std. """
    return (array - array.mean(axis=0)) / array.std(axis=0)


def standardization_(array):
    """ Normalize the features inplace to zero mean and unit std. """
    arr_mean = array.mean(axis=0)
    arr_std = array.std(axis=0)
    array -= arr_mean
    np.divide(array.astype(np.float64), arr_std, out=array)


def scaling_to_unit_length(array, norm_p=2):
    """ Scale the components by dividing each features by their p-norm. """
    return array / np.sum(np.abs(array) ** norm_p, axis=0) ** 1 / norm_p


def scaling_to_unit_length_(array, norm_p=2):
    """ Scale the components inplace by dividing each features by their p-norm. """
    np.divide(array.astype(np.float64), np.sum(np.abs(array) ** norm_p, axis=0) ** 1 / norm_p, out=array)


def whitening_zca(array):
    """ Maximizes the average cross-covariance between each dimension of the whitened and original data, and uniquely
    produces a symmetric cross-covariance matrix.
    """
    pass


def whitening_zca_cor(array):
    """ Maximizes the average cross-correlation between each dimension of the whitened and original data, and uniquely
    produces a symmetric cross-correlation matrix.
    """
    pass


def whitening_pca(array):
    """ Maximally compresses all dimensions of the original data into each dimension of the whitened data using the
    cross-covariance matrix as the compression metric.
    """
    pass


def whitening_pca_cor(array):
    """ Maximally compresses all dimensions of the original data into each dimension of the whitened data using the
    cross-correlation matrix as the compression metric.
    """
    pass


def whitening_cholesky(array):
    """ Uniquely results in lower triangular positive diagonal cross-covariance and cross-correlation matrices. """
    pass


def _str_to_normalization(string, inplace=True):
    global ALIASES_RESCALING, ALIASES_MEAN_NORMALIZATION, ALIASES_STANDARDIZATION, ALIASES_SCALE_TO_UNIT_LENGTH,\
        ALIASES_WHITENING_ZCA, ALIASES_WHITENING_PCA, ALIASES_WHITENING_CHOLESKY, ALIASES_WHITENING_ZCA_COR,\
        ALIASES_WHITENING_PCA_COR

    string = string.lower()
    if string in ALIASES_RESCALING:
        return rescaling_ if inplace else rescaling
    elif string in ALIASES_MEAN_NORMALIZATION:
        return mean_normalization_ if inplace else mean_normalization
    elif string in ALIASES_STANDARDIZATION:
        return standardization_ if inplace else standardization
    elif string in ALIASES_SCALE_TO_UNIT_LENGTH:
        return scaling_to_unit_length_ if inplace else scaling_to_unit_length
    elif string in ALIASES_WHITENING_ZCA:
        raise NotImplementedError()
    elif string in ALIASES_WHITENING_PCA:
        raise NotImplementedError()
    elif string in ALIASES_WHITENING_CHOLESKY:
        raise NotImplementedError()
    elif string in ALIASES_WHITENING_ZCA_COR:
        raise NotImplementedError()
    elif string in ALIASES_WHITENING_PCA_COR:
        raise NotImplementedError()
    else:
        raise UnknownNormalization(string)


if __name__ == "__main__":
    pass
