import time

import numpy as np
from clus.src.handle_empty_clusters import handle_empty_clusters
from clus.src.initialization import cluster_initialization
from clus.src.utils.decorator import remove_unexpected_arguments, time_this
from clus.src.visualisation import print_progression


# TODO: Improve _compute_medoids
#  Tested with :
#  clus data/S-sets/s2.csv fcmdd -k 10 --visualise --normalization rescaling --fuzzifier 2 --seed 1
#  res in : results/fcmdd


@remove_unexpected_arguments
def fuzzy_c_medoids(data, distance_matrix, components, fuzzifier, eps, max_iter, initialization_method,
                    empty_clusters_method, medoids_idx=None):
    assert (len(distance_matrix.shape) == 2) and distance_matrix.shape[0] == distance_matrix.shape[1],\
        "The distance matrix is not squared"
    assert initialization_method in (2, 3, 4), "Your initialization method must be based on example selection"
    assert (medoids_idx is None) or \
           ((medoids_idx.shape == (components, data.shape[1])) and (all(medoids_idx < data.shape[0])))

    # Initialisation
    if medoids_idx is None:
        medoids_idx = cluster_initialization(distance_matrix, components, initialization_method, need_idx=True)

    memberships = None
    current_iter = 0
    losses = []
    medoids_idx_old = None
    start_time = time.time()
    while (current_iter <= max_iter) and \
            ((current_iter < 1) or (not all(medoids_idx == medoids_idx_old))) and \
            ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        medoids_idx_old = medoids_idx
        memberships = _compute_memberships(distance_matrix, medoids_idx, fuzzifier)
        handle_empty_clusters(distance_matrix, medoids_idx, memberships, strategy=empty_clusters_method)

        medoids_idx = _compute_medoids(distance_matrix, memberships, fuzzifier)

        loss = _compute_loss(distance_matrix, medoids_idx, memberships, fuzzifier)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter, loss=loss, start_time=start_time)
    return memberships, data[medoids_idx, :], np.array(losses)


def _compute_memberships(distance_matrix, medoids_idx, fuzzifier):
    dist_data_medoids = distance_matrix[:, medoids_idx]

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    dist_data_medoids += np.fmax(dist_data_medoids, np.finfo(distance_matrix.dtype).eps)

    tmp = (1 / dist_data_medoids) ** (1 / (fuzzifier - 1))
    memberships = tmp / tmp.sum(axis=1, keepdims=True)

    # TODO: Optimisable
    for index_medoid, medoid in enumerate(medoids_idx):
        memberships[medoid, :] = 0.
        memberships[medoid, index_medoid] = 1.

    return memberships


def _compute_medoids(distance_matrix, memberships, fuzzifier):
    fuzzified_memberships = memberships ** fuzzifier
    iterable = ((distance_matrix * fuzzified_memberships[:, i]).sum(axis=1).argmin(axis=0) for i in range(memberships.shape[1]))
    return np.fromiter(iterable, count=memberships.shape[1], dtype=np.int64)


def _compute_loss(distance_matrix, medoids_idx, memberships, fuzzifier):
    return ((memberships ** fuzzifier) * distance_matrix[:, medoids_idx]).sum(axis=(1, 0))


def __compute_medoids(distance_matrix, memberships, fuzzifier):
    """ DEPRECATED: old method used to compute the medoids.
    Very memory-heavy and slower than the existing method.
    """
    fuzzified_memberships = memberships ** fuzzifier
    return (distance_matrix[..., np.newaxis] * fuzzified_memberships).sum(axis=1).argmin(axis=0)


if __name__ == '__main__':
    pass
