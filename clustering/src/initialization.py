import enum

import numpy as np


@enum.unique
class ClusterInitialization(enum.IntEnum):
    UNKNOWN = -1
    RANDOM_UNIFORM = 0
    RANDOM_GAUSSIAN = 1
    RANDOM_CHOICE = 2
    CENTRAL_DISSIMILAR_MEDOIDS = 3
    CENTRAL_DISSIMILAR_RANDOM_MEDOIDS = 4


class UnknownClusterInitialization(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The cluster initialization method : \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def cluster_initialization(data, components, strategy: int, need_idx: bool):
    if need_idx:
        strategy = int_to_clusterinitialization_idx_function(strategy)
    else:
        strategy = int_to_clusterinitialization_function(strategy)
    return strategy(data, components)


def random_uniform(data, components):
    return np.random.uniform(low=data.min(axis=0), high=data.max(axis=0),
                             size=(components, data.shape[0])).astype(np.float64)


def random_gaussian(data, components):
    return np.random.normal(loc=data.mean(axis=0), scale=data.std(axis=0),
                            size=(components, data.shape[0])).astype(np.float64)


def random_choice(data, components):
    assert data.shape[0] >= components, ("Cannot take a number of components larger than the number of samples with thi"
                                         "s initialization method")

    idx = np.random.choice(np.arange(data.shape[0]), size=components, replace=False)
    return data[idx, :]


def random_choice_idx(data, components):
    assert data.shape[0] >= components, ("Cannot take a number of components larger than the number of samples with thi"
                                         "s initialization method")

    return np.random.choice(np.arange(data.shape[0]), size=components, replace=False)


def int_to_clusterinitialization(integer):
    if integer in [init_method.value for init_method in ClusterInitialization]:
        return ClusterInitialization(integer)
    else:
        raise UnknownClusterInitialization(str(integer))


def str_to_clusterinitialization(string):
    string = string.lower()
    try:
        string = int(string)
    except ValueError:
        raise UnknownClusterInitialization(string)

    return int_to_clusterinitialization(string)


def clusterinitialization_to_str(init_method):
    return init_method.name.lower()


def clusterinitialization_to_function(init_method):
    if init_method is ClusterInitialization.RANDOM_UNIFORM:
        return random_uniform
    elif init_method is ClusterInitialization.RANDOM_GAUSSIAN:
        return random_gaussian
    elif init_method is ClusterInitialization.RANDOM_CHOICE:
        return random_choice
    elif init_method is ClusterInitialization.CENTRAL_DISSIMILAR_MEDOIDS:
        raise NotImplementedError(clusterinitialization_to_str(init_method))
    elif init_method is ClusterInitialization.CENTRAL_DISSIMILAR_RANDOM_MEDOIDS:
        raise NotImplementedError(clusterinitialization_to_str(init_method))


def clusterinitialization_to_idx_function(init_method):
    if init_method is ClusterInitialization.RANDOM_CHOICE:
        return random_choice_idx
    elif init_method is ClusterInitialization.CENTRAL_DISSIMILAR_MEDOIDS:
        raise NotImplementedError(clusterinitialization_to_str(init_method))
    elif init_method is ClusterInitialization.CENTRAL_DISSIMILAR_RANDOM_MEDOIDS:
        raise NotImplementedError(clusterinitialization_to_str(init_method))


def int_to_clusterinitialization_function(integer):
    method = int_to_clusterinitialization(integer)
    return clusterinitialization_to_function(method)


def int_to_clusterinitialization_idx_function(integer):
    method = int_to_clusterinitialization(integer)
    return clusterinitialization_to_idx_function(method)


def str_to_clusterinitialization_function(string):
    method = str_to_clusterinitialization(string)
    return clusterinitialization_to_function(method)


def str_to_clusterinitialization_idx_function(string):
    method = str_to_clusterinitialization(string)
    return clusterinitialization_to_idx_function(method)


if __name__ == "__main__":
    pass
