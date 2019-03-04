import time

import numpy as np
from scipy.spatial.distance import cdist

from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def fuzzy_c_means(data, components, fuzzifier, eps, max_iter):
    # Initialisation
    nb_examples, dim = data.shape
    centroids = np.random.rand(components, dim).astype(np.float64)
    affectations = np.zeros(shape=(nb_examples, components), dtype=np.float64)

    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        affectations = _optim_affectations(data, centroids, fuzzifier)
        centroids = _optim_centroids(data, affectations, fuzzifier)

        loss = _compute_loss(data, affectations, centroids, fuzzifier)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter, loss=loss, start_time=start_time)
    return affectations, centroids, np.array(losses)


def _optim_affectations(data, centroids, fuzzifier):
    dist_data_centroids = cdist(data, centroids, metric="euclidean").T

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    dist_data_centroids = np.fmax(dist_data_centroids, np.finfo(data.dtype).eps)

    tmp = dist_data_centroids ** (-2. / (fuzzifier - 1))
    return (tmp / np.sum(tmp, axis=0, keepdims=True)).T


def _optim_centroids(data, affectations, fuzzifier):
    return (np.dot(data.T, np.power(affectations, fuzzifier)) /
            np.sum(np.power(affectations, fuzzifier), axis=0)).T


def _compute_loss(data, affectations, centroids, fuzzifier):
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (np.power(affectations, fuzzifier) *
            np.power(np.linalg.norm(dist_data_centroids, axis=-1,
                                    ord=2), 2).T).sum()


if __name__ == '__main__':
    pass
