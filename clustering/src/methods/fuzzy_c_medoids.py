import sys
import time

import numpy as np
from clustering.src.handle_empty_clusters import handle_empty_clusters
from sklearn.neighbors.dist_metrics import DistanceMetric

from clustering.src.initialization import cluster_initialization
from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def fuzzy_c_medoids(distance_matrix, components, fuzzifier, eps, max_iter, initialization_method, empty_clusters_method):
    assert (len(distance_matrix.shape) == 2) and distance_matrix.shape[0] == distance_matrix.shape[1], "The distance matrix is not squared"
    assert initialization_method in (2, 3, 4), "Your initialization method must be based on example selection"

    # Initialisation
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
    return memberships, distance_matrix[medoids_idx, :], np.array(losses)


def _compute_memberships(data, medoids_idx, fuzzifier):
    dist_data_medoids = data[:, medoids_idx]

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    dist_data_medoids += np.fmax(dist_data_medoids, np.finfo(data.dtype).eps)

    tmp = (1 / dist_data_medoids) ** (1 / (fuzzifier - 1))
    memberships = tmp / tmp.sum(axis=1, keepdims=True)

    for index_medoid, medoid in enumerate(medoids_idx):
        memberships[medoid, :] = 0.
        memberships[medoid, index_medoid] = 1.

    return memberships


def _compute_medoids(data, memberships, fuzzifier):
    return (data[..., np.newaxis] * (memberships ** fuzzifier)).sum(axis=1).argmin(axis=0)


def _compute_loss(data, medoids_idx, memberships, fuzzifier):
    return ((memberships ** fuzzifier) * data[:, medoids_idx]).sum(axis=(1, 0))


if __name__ == '__main__':
    pass
