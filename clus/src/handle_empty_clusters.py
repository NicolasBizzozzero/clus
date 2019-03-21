from typing import Callable

import numpy as np


class UnknownEmptyClustersMethod(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The empty clusters method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def handle_empty_clusters(data: np.ndarray, clusters_center: np.ndarray, memberships: np.ndarray,
                          strategy: str = "nothing") -> None:
    if strategy == "nothing":
        return

    empty_clusters_idx = np.where(np.sum(memberships, axis=0) == 0)[0]
    if empty_clusters_idx.size != 0:
        strategy = _str_to_emptyclustermethod(strategy)
        strategy(data, clusters_center, memberships, empty_clusters_idx)


def random_example(data: np.ndarray, clusters_center: np.ndarray, memberships: np.ndarray,
                   empty_clusters_idx: np.ndarray):
    new_examples_idx = np.random.choice(np.arange(data.shape[0]),
                                        size=empty_clusters_idx.size, replace=False)
    memberships[new_examples_idx, :] = 0
    memberships[new_examples_idx, empty_clusters_idx] = 1


def furthest_example_from_its_centroid(data: np.ndarray, clusters_center: np.ndarray,
                                       memberships: np.ndarray,
                                       empty_clusters_idx: np.ndarray):
    pass


def _str_to_emptyclustermethod(string: str) -> Callable:
    string = string.lower()
    if string in ("random_example",):
        return random_example
    if string in ("furthest_example_from_its_centroid",):
        raise NotImplementedError()
    raise UnknownEmptyClustersMethod(string)


if __name__ == "__main__":
    pass
