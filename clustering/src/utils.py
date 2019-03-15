import random
import inspect
import functools
import sys
import time
import warnings

import numpy as np


# Used for stocking time delta between each iterations
from scipy._lib._util import _asarray_validated
from sklearn.preprocessing import MinMaxScaler

_TIME_LAST_ITERATION = None
_TIME_DELTAS = []


def remove_unexpected_arguments(func):
    """ The decorated function sliently ignore unexpected parameters without
    raising any error.

    Authors :
    * BIZZOZZERO Nicolas
    * POUYET Adrien
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        possible_parameters = inspect.getfullargspec(func).args
        new_kwargs = dict(filter(lambda a: a[0] in possible_parameters, kwargs.items()))

        return func(*args, **new_kwargs)
    return wrapper


def set_manual_seed(seed):
    """ Set a manual seed to the following packages :
    * Python's default random library
    * numpy
    * scikit-learn (use the same seed as numpy)
    This function attempts to make all results produced by these librairies as
    deterministic as possible.
    :param seed: The manual seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)


def retrieve_decimals(number: float, number_of_decimals_wanted=2) -> str:
    return str(round(number % 1, number_of_decimals_wanted)).split(".")[-1]


def pretty_time_delta(seconds):
    """ Pretty print a time delta in days, hours, minutes and seconds.

    Source :
    * https://gist.github.com/thatalextaylor/7408395
    """

    milliseconds = retrieve_decimals(seconds, number_of_decimals_wanted=2)
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds, seconds, days, hours, minutes = \
        milliseconds.zfill(2), str(seconds).zfill(2), str(days), str(hours).zfill(2), str(minutes).zfill(2)
    if int(days) > 0:
        return "{days}d{hours}h{minutes}m{seconds}s{milliseconds}ms".format(
            days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    elif int(hours) > 0:
        return "{hours}h{minutes}m{seconds}s{milliseconds}ms".format(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    elif int(minutes) > 0:
        return "{minutes}m{seconds}s{milliseconds}ms".format(
            minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    else:
        return "{seconds}s{milliseconds}ms".format(
            seconds=seconds, milliseconds=milliseconds
        )


def print_progression(iteration, loss, start_time):
    global _TIME_DELTAS, _TIME_LAST_ITERATION

    if _TIME_LAST_ITERATION is None:
        _TIME_DELTAS.append(time.time() - start_time)
        _TIME_LAST_ITERATION = _TIME_DELTAS[-1] + start_time
    else:
        _TIME_DELTAS.append(time.time() - _TIME_LAST_ITERATION)
        _TIME_LAST_ITERATION += _TIME_DELTAS[-1]

    extended_time = time.time() - start_time

    sys.stdout.write('\r')
    sys.stdout.write(("Iteration {iteration}\t"
                      "Loss {loss}\t"
                      "Extended_time {extended_time}\t"
                      "Mean_iter_time {mean_iter_time} (std {std_iter_time})").format(
        iteration=iteration,
        loss=round(loss, 4),
        extended_time=pretty_time_delta(extended_time),
        mean_iter_time=pretty_time_delta(np.mean(_TIME_DELTAS)),
        std_iter_time=pretty_time_delta(np.std(_TIME_DELTAS))
    ))
    sys.stdout.flush()


def normalization_mean_std(array):
    """ Normalize `array` to zero mean and unit variance.

    Source :
    * https://stackoverflow.com/a/31153050
    """
    return (array - array.mean(axis=0)) / array.std(axis=0)


def normalize_range(array, floor=0, ceil=1):
    """ Normalise an array between a given range.
    :param array: The array to normalize. Also works for pandas.DataFrame with
    numeric values.
    :param floor: The minimal value of the normalized range.
    :param ceil: The maximal value of the normalized range.
    """
    scaler = MinMaxScaler(feature_range=(floor, ceil), copy=True)
    return scaler.fit_transform(array)


def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.
    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set with whitening. Each feature is
    divided by its standard deviation across all observations to give
    it unit variance.
    Parameters
    ----------
    obs : ndarray
        Each row of the array is an observation.  The
        columns are the features seen during each observation.
        >>> #         f0    f1    f2
        >>> obs = [[  1.,   1.,   1.],  #o0
        ...        [  2.,   2.,   2.],  #o1
        ...        [  3.,   3.,   3.],  #o2
        ...        [  4.,   4.,   4.]]  #o3
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation
        of each column.
    Examples
    --------
    >>> from scipy.cluster.vq import whiten
    >>> features  = np.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7,]])
    >>> whiten(features)
    array([[ 4.17944278,  2.69811351,  7.21248917],
           [ 3.29956009,  2.93273208,  9.33380951],
           [ 1.75976538,  0.7038557 ,  7.21248917]])
    Source
    ------
    * https://github.com/scipy/scipy/blob/master/scipy/cluster/vq.py#L87
    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = obs.std(axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return obs / std_dev


if __name__ == '__main__':
    pass
