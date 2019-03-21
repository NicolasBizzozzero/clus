from typing import Callable

import numpy as np


class UnknownClusterInitialization(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The cluster initialization method \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


class ClusterInitializationCantReturnIndexes(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The cluster initialization method \"{method_name}\" can't return data indexes".format(
            method_name=method_name
        ))


def cluster_initialization(data: np.ndarray, components: int, strategy: str, need_idx: bool) -> np.ndarray:
    strategy = _str_to_clusterinitialization(strategy, need_idx=need_idx)
    return strategy(data, components)


def random_uniform(data: np.ndarray, components: int) -> np.ndarray:
    return np.random.uniform(low=data.min(axis=0), high=data.max(axis=0),
                             size=(components, data.shape[0])).astype(np.float64)


def random_gaussian(data: np.ndarray, components: int) -> np.ndarray:
    return np.random.normal(loc=data.mean(axis=0), scale=data.std(axis=0),
                            size=(components, data.shape[0])).astype(np.float64)


def random_choice(data: np.ndarray, components: int) -> np.ndarray:
    assert data.shape[0] >= components, ("Cannot take a number of components larger than the number of samples with thi"
                                         "s initialization method")

    idx = np.random.choice(np.arange(data.shape[0]), size=components, replace=False)
    return data[idx, :]


def random_choice_idx(data: np.ndarray, components: int) -> np.ndarray:
    assert data.shape[0] >= components, ("Cannot take a number of components larger than the number of samples with thi"
                                         "s initialization method")

    return np.random.choice(np.arange(data.shape[0]), size=components, replace=False)


def _str_to_clusterinitialization(string: str, need_idx: bool) -> Callable:
    string = string.lower()
    if string in ("random_uniform", "uniform"):
        if need_idx:
            raise ClusterInitializationCantReturnIndexes(string)
        return random_uniform
    if string in ("random_gaussian", "gaussian"):
        if need_idx:
            raise ClusterInitializationCantReturnIndexes(string)
        return random_gaussian
    if string in ("random_choice", "choice"):
        if need_idx:
            return random_choice_idx
        return random_choice
    if string in ("central_dissimilar_medoids",):
        if need_idx:
            raise NotImplementedError()
        raise NotImplementedError()
    if string in ("central_dissimilar_random_medoids",):
        if need_idx:
            raise NotImplementedError()
        raise NotImplementedError()
    raise UnknownClusterInitialization(string)


if __name__ == "__main__":
    pass
