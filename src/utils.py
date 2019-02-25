import random

import numpy as np


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
