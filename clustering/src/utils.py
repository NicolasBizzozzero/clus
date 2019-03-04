import random
import inspect
import functools
import sys
import time

import numpy as np


# Used for stocking time delta between each iterations
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


if __name__ == '__main__':
    pass
