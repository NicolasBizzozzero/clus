import enum

import numpy as np


@enum.unique
class EmptyClustersMethod(enum.IntEnum):
    UNKNOWN = -1
    NOTHING = 1
    RANDOM_EXAMPLE = 2
    FURTHEST_EXAMPLE_FROM_ITS_CENTROID = 3


class UnknownEmptyClustersMethod(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The empty clusters method : \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def handle_empty_clusters(data, clusters_center, memberships, strategy):
    if strategy == EmptyClustersMethod.NOTHING:
        return

    empty_clusters_idx = np.where(np.sum(memberships, axis=0) == 0)[0]
    if empty_clusters_idx.size != 0:
        strategy = int_to_emptyclustermethod_function(strategy)
        strategy(data, clusters_center, memberships, empty_clusters_idx)


def nothing(data, clusters_center, memberships, empty_clusters_idx):
    pass


def random_example(data, clusters_center, memberships, empty_clusters_idx):
    new_examples_idx = np.random.choice(
        np.arange(data.shape[0]), size=empty_clusters_idx.size, replace=False)
    memberships[new_examples_idx, :] = 0
    memberships[new_examples_idx, empty_clusters_idx] = 1


def furthest_example_from_its_centroid(data, clusters_center, memberships, empty_clusters_idx):
    pass


def int_to_emptyclustermethod(integer):
    if integer in [init_method.value for init_method in EmptyClustersMethod]:
        return EmptyClustersMethod(integer)
    else:
        raise UnknownEmptyClustersMethod(str(integer))


def str_to_emptyclustermethod(string):
    string = string.lower()
    try:
        string = int(string)
    except ValueError:
        raise UnknownEmptyClustersMethod(string)

    return int_to_emptyclustermethod(string)


def emptyclustermethod_to_str(init_method):
    return init_method.name.lower()


def emptyclustermethod_to_function(init_method):
    if init_method is EmptyClustersMethod.NOTHING:
        return nothing
    elif init_method is EmptyClustersMethod.RANDOM_EXAMPLE:
        return random_example
    elif init_method is EmptyClustersMethod.FURTHEST_EXAMPLE_FROM_ITS_CENTROID:
        raise NotImplementedError(emptyclustermethod_to_str(init_method))


def int_to_emptyclustermethod_function(integer):
    method = int_to_emptyclustermethod(integer)
    return emptyclustermethod_to_function(method)


def str_to_emptyclustermethod_function(string):
    method = str_to_emptyclustermethod(string)
    return emptyclustermethod_to_function(method)


if __name__ == "__main__":
    pass
