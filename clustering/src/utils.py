import random
import inspect

import numpy as np


def remove_unexpected_arguments(func):
    """ The decorated function sliently ignore unexpected parameters without
    raising any error.

    Authors :
    * BIZZOZZERO Nicolas
    * POUYET Adrien
    """
    def wrapper(*arg, **kwargs):
        possible_parameters = inspect.getfullargspec(func).args
        new_kwargs = dict(filter(lambda a: a[0] in possible_parameters, kwargs.items()))

        return func(*arg, **new_kwargs)
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


if __name__ == '__main__':
    pass
