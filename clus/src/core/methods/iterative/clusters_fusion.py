from typing import Callable, List

import numpy as np


ALIASES_ITERATIVE = ("iterative",)
ALIASES_END_CLUSTERING = ("end", "end_clustering")


class UnknownFusionMethod(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The fusion method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def clusters_fusion(clusters: List[np.ndarray], strategy: str) -> np.ndarray:
    strategy = _str_to_fusionmethod(strategy)
    return strategy(clusters)


def iterative(clusters: List[np.ndarray]) -> np.ndarray:
    pass


def end_clustering(clusters: List[np.ndarray]) -> np.ndarray:
    pass


def _str_to_fusionmethod(string: str) -> Callable:
    global ALIASES_ITERATIVE, ALIASES_END_CLUSTERING

    string = string.lower()
    if string in ALIASES_ITERATIVE:
        return iterative
    elif string in ALIASES_END_CLUSTERING:
        return end_clustering
    else:
        raise UnknownFusionMethod(string)


if __name__ == "__main__":
    pass
