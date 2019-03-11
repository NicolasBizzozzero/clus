import time

import numpy as np

from clustering.src.handle_empty_clusters import handle_empty_clusters
from clustering.src.initialization import cluster_initialization
from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def kmeans(data, components, eps, max_iter, initialization_method, empty_clusters_method, centroids=None):
    assert (centroids is None) or (centroids.shape == (components, data.shape[1]))

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, strategy=initialization_method, need_idx=False)

    memberships = None
    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        memberships = _optim_memberships(data, centroids)
        handle_empty_clusters(data, centroids, memberships,
                              strategy=empty_clusters_method)

        centroids = _optim_centroids(data, memberships)

        loss = _compute_loss(data, memberships, centroids)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter,
                          loss=loss, start_time=start_time)
    return memberships, centroids, np.array(losses)


def _optim_memberships(data, centroids):
    """

    Source :
    * https://codereview.stackexchange.com/questions/61598/k-mean-with-numpy
    """
    # Compute euclidean distance between data and centroids
    # dist_data_centroids = np.array([np.linalg.norm(data - c, ord=2, axis=1) for c in centroids]).T
    # dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1).T
    dist_data_centroids = np.linalg.norm(np.expand_dims(
        data, 2) - np.expand_dims(centroids.T, 0), axis=1)

    # Set all binary affectations
    mask_closest_centroid = (
        np.arange(data.shape[0]), dist_data_centroids.argmin(axis=1))
    affectations = np.zeros(shape=dist_data_centroids.shape, dtype=np.int32)
    affectations[mask_closest_centroid] = 1

    return affectations


def _optim_centroids(data, memberships):
    # We compute the division only to with non-empty. Indeed, a cluster may be
    # empty in some rare cases. See [2]
    sum_memberships_by_centroid = np.sum(memberships, axis=0)
    return np.divide(np.dot(data.T, memberships),
                     sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids):
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (memberships *
            np.power(np.linalg.norm(dist_data_centroids, axis=-1, ord=2),
                     2).T).sum()


if __name__ == '__main__':
    pass
