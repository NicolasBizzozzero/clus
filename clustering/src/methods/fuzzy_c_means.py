import time

import numpy as np

from clustering.src.handle_empty_clusters import handle_empty_clusters
from clustering.src.initialization import cluster_initialization
from clustering.src.utils import remove_unexpected_arguments

# TODO: If an example is at the exact same coordinates than a centroid (euclidean distance == 0), set its membership to
 # 1, and the memberships of others to 0.
 # See [3]

# TODO: Why is sometime the loss increasing ? Is it normal ? Try FCM s4.csv, seed=1, m=1.5 and see iter 27-28


@remove_unexpected_arguments
def fuzzy_c_means(data, components, fuzzifier, eps, max_iter, initialization_method, empty_clusters_method,
                  centroids=None):
    assert fuzzifier > 1
    assert (centroids is None) or (centroids.shape == (components, data.shape[1]))
    assert 1 <= components <= data.shape[0]

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, initialization_method, need_idx=False)

    memberships = None
    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter):# and \
          #((current_iter < 2) or (abs(losses[-2] - losses[-1] > eps))):
        memberships = _compute_memberships(data, centroids, fuzzifier)
        handle_empty_clusters(data, centroids, memberships,
                              strategy=empty_clusters_method)

        centroids = _compute_centroids(data, memberships, fuzzifier)

        loss = _compute_loss(data, memberships, centroids, fuzzifier)
        losses.append(loss)

        grow = False if (current_iter < 2) else (losses[-2] - losses[-1]) < 0

        current_iter += 1
        print("Iter :", current_iter, "Loss :", loss, "\tgrow? :", grow)
        # print_progression(iteration=current_iter,
        #                  loss=loss, start_time=start_time)

    return memberships, centroids, np.array(losses)


def _compute_memberships(data, centroids, fuzzifier):
    dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1) ** 2

    tmp = np.power(dist_data_centroids, -2 / (fuzzifier - 1), where=dist_data_centroids != 0)
    big_sum = tmp.sum(axis=0, keepdims=True)
    res = np.divide(tmp, big_sum, where=big_sum != 0).T
    res = np.fmax(res, 0.)  # Float manipulation sometimes cause a 0. to be set to -0.
    return res


def _compute_centroids(data, memberships, fuzzifier):
    fuzzified_memberships = memberships ** fuzzifier
    sum_memberships_by_centroid = np.sum(fuzzified_memberships, axis=0)
    return np.divide(np.dot(data.T, fuzzified_memberships),
                     sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids, fuzzifier):
    dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1) ** 2
    return ((memberships ** fuzzifier) * dist_data_centroids.T).sum()


if __name__ == '__main__':
    pass
