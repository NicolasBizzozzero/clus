import time

import numpy as np
from clustering.src.handle_empty_clusters import handle_empty_clusters

from clustering.src.initialization import cluster_initialization
from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def fuzzy_c_means(data, components, fuzzifier, eps, max_iter, initialization_method, empty_clusters_method,
                  centroids=None):
    assert fuzzifier > 1
    assert (centroids is None) or (centroids.shape == (components, data.shape[1]))

    # Initialisation
    if centroids is not None:
        centroids = cluster_initialization(data, components, initialization_method, need_idx=False)

    memberships = None
    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        memberships = _compute_memberships(data, centroids, fuzzifier)
        handle_empty_clusters(data, centroids, memberships,
                              strategy=empty_clusters_method)

        centroids = _compute_centroids(data, memberships, fuzzifier)

        loss = _compute_loss(data, memberships, centroids, fuzzifier)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter,
                          loss=loss, start_time=start_time)
    return memberships, centroids, np.array(losses)


def _compute_memberships(data, centroids, fuzzifier):
    dist_data_centroids = np.linalg.norm(
        data - centroids[:, np.newaxis], ord=2, axis=-1)

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    try:
        dist_data_centroids = np.fmax(
            dist_data_centroids, np.finfo(data.dtype).eps)
    except ValueError:
        # Data is of type int, adding float64 by default
        dist_data_centroids = np.fmax(
            dist_data_centroids, np.finfo(np.float64).eps)

    tmp = dist_data_centroids ** (-2. / (fuzzifier - 1))
    return (tmp / tmp.sum(axis=0, keepdims=True)).T


def _compute_centroids(data, memberships, fuzzifier):
    # TODO : Compare with old formula.
    # TODO: Apply to other algorithms
    fuzzified_memberships = memberships ** fuzzifier
    sum_memberships_by_centroid = np.sum(fuzzified_memberships, axis=0)
    return np.divide(np.dot(data.T, fuzzified_memberships),
                     sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T

    # return (np.dot(data.T, np.power(memberships, fuzzifier)) /
    #         np.sum(np.power(memberships, fuzzifier), axis=0)).T


def _compute_loss(data, affectations, centroids, fuzzifier):
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (np.power(affectations, fuzzifier) *
            np.linalg.norm(dist_data_centroids, axis=-1, ord=2).T).sum()


if __name__ == '__main__':
    pass
