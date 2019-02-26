import enum
from typing import Callable

from clustering.src.methods import kmeans, fuzzy_c_means
from clustering.src.methods.fuzzy_c_medoids import fuzzy_c_medoids

ALIAS_CM_KMEANS = ("kmeans",)
ALIAS_CM_FUZZY_C_MEANS = ("fuzzy_c_means", "fcm")
ALIAS_CM_POSSIBILISTIC_C_MEANS = ("possibilistic_c_means", "pcm")
ALIAS_CM_FUZZY_C_MEDOIDS = ("fuzzy_c_medoids", "fcmdd")
ALIAS_CM_HARD_C_MEDOIDS = ("hard_c_medoids", "hcmdd")
ALIAS_CM_LINEARIZED_FUZZY_C_MEDOIDS = ("linearized_fuzzy_c_medoids", "lfcmdd", "l_fc_med")
ALIAS_CM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT = ("linearized_fuzzy_c_medoids_select", "l_fcmed_select")
ALIAS_CM_DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT = ("datastream_linearized_fuzzy_c_medoids_select",
                                                         "ds_lfcmed_select")


@enum.unique
class ClusteringMethod(enum.IntEnum):
    UNKNOWN = -1
    KMEANS = 0
    FUZZY_C_MEANS = 1
    POSSIBILISTIC_C_MEANS = 2
    FUZZY_C_MEDOIDS = 3
    HARD_C_MEDOIDS = 4
    LINEARIZED_FUZZY_C_MEDOIDS = 5
    LINEARIZED_FUZZY_C_MEDOIDS_SELECT = 6
    DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT = 7


class UnknownClusteringMethods(Exception):
    def __init__(self, method_name: str):
        Exception.__init__(self, "The clustering method : \"{method_name}\" doesn't exists".format(
            method_name=method_name
        ))


def get_clustering_function(string: str) -> Callable:
    clustering_method = str_to_clusteringmethod(string)
    function = clusteringmethod_to_function(clustering_method)
    return function


def str_to_clusteringmethod(string):
    string = string.lower()

    if string in ALIAS_CM_KMEANS:
        return ClusteringMethod.KMEANS
    elif string in ALIAS_CM_FUZZY_C_MEANS:
        return ClusteringMethod.FUZZY_C_MEANS
    elif string in ALIAS_CM_POSSIBILISTIC_C_MEANS:
        return ClusteringMethod.POSSIBILISTIC_C_MEANS
    elif string in ALIAS_CM_FUZZY_C_MEDOIDS:
        return ClusteringMethod.FUZZY_C_MEDOIDS
    elif string in ALIAS_CM_HARD_C_MEDOIDS:
        return ClusteringMethod.HARD_C_MEDOIDS
    elif string in ALIAS_CM_LINEARIZED_FUZZY_C_MEDOIDS:
        return ClusteringMethod.LINEARIZED_FUZZY_C_MEDOIDS
    elif string in ALIAS_CM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT:
        return ClusteringMethod.LINEARIZED_FUZZY_C_MEDOIDS_SELECT
    elif string in ALIAS_CM_DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT:
        return ClusteringMethod.DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT
    else:
        raise UnknownClusteringMethods(string)


def clusteringmethod_to_str(clustering_method):
    return clustering_method.name.lower()


def clusteringmethod_to_function(clustering_method):
    if clustering_method is ClusteringMethod.KMEANS:
        return kmeans
    elif clustering_method is ClusteringMethod.FUZZY_C_MEANS:
        return fuzzy_c_means
    elif clustering_method is ClusteringMethod.POSSIBILISTIC_C_MEANS:
        raise NotImplementedError()
    elif clustering_method is ClusteringMethod.FUZZY_C_MEDOIDS:
        return fuzzy_c_medoids
    elif clustering_method is ClusteringMethod.HARD_C_MEDOIDS:
        raise NotImplementedError()
    elif clustering_method is ClusteringMethod.LINEARIZED_FUZZY_C_MEDOIDS:
        raise NotImplementedError()
    elif clustering_method is ClusteringMethod.LINEARIZED_FUZZY_C_MEDOIDS_SELECT:
        raise NotImplementedError()
    elif clustering_method is ClusteringMethod.DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT:
        raise NotImplementedError()


if __name__ == "__main__":
    pass
