""" This module contains multiple methods to handle empty clusters at each iteration of a clustering algorithm.

If you want to add another method, you need to update the `_str_to_emptyclustermethod` function to
return your new method, along with its aliases.
All methods must take as input :
* The data
* Each clusters' center
* The memberships matrix
* The indexes of all empty clusters
All methods do not return anything, they handle the empty clusters in-place.
"""

from typing import Callable

import numpy as np


ALIASES_NOTHING = ("nothing",)
ALIASES_RANDOM_EXAMPLE = ("random_example",)
ALIASES_FURTHEST_EXAMPLE_FROM_ITS_CENTROID = ("furthest_example_from_its_centroid",)


class UnknownEmptyClustersMethod(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The empty clusters method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def handle_empty_clusters(data: np.ndarray, clusters_center: np.ndarray, memberships: np.ndarray,
                          strategy: str = "nothing") -> None:
    global ALIASES_NOTHING

    if strategy in ALIASES_NOTHING:
        return

    empty_clusters_idx = np.where(np.sum(memberships, axis=0) == 0)[0]
    if empty_clusters_idx.size != 0:
        strategy = _str_to_emptyclustermethod(strategy)
        strategy(data, clusters_center, memberships, empty_clusters_idx)


def random_example(data: np.ndarray, clusters_center: np.ndarray, memberships: np.ndarray,
                   empty_clusters_idx: np.ndarray) -> None:
    new_examples_idx = np.random.choice(np.arange(data.shape[0]),
                                        size=empty_clusters_idx.size, replace=False)
    memberships[new_examples_idx, :] = 0
    memberships[new_examples_idx, empty_clusters_idx] = 1


def furthest_example_from_its_centroid(data: np.ndarray, clusters_center: np.ndarray,
                                       memberships: np.ndarray,
                                       empty_clusters_idx: np.ndarray) -> None:
    pass


def _str_to_emptyclustermethod(string: str) -> Callable:
    global ALIASES_RANDOM_EXAMPLE, ALIASES_FURTHEST_EXAMPLE_FROM_ITS_CENTROID

    string = string.lower()
    if string in ALIASES_RANDOM_EXAMPLE:
        return random_example
    if string in ALIASES_FURTHEST_EXAMPLE_FROM_ITS_CENTROID:
        raise NotImplementedError()
    raise UnknownEmptyClustersMethod(string)


if __name__ == "__main__":
    pass
