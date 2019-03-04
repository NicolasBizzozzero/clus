import time

import numpy as np

from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def kmeans(data, components, eps, max_iter):
    # Initialisation
    nb_examples, dim = data.shape
    centroids = np.random.rand(components, dim).astype(np.float64)
    affectations = np.zeros(shape=(nb_examples, components), dtype=np.float64)

    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        affectations = _optim_affectations(data, centroids)
        centroids = _optim_centroids(data, affectations)

        loss = _compute_loss(data, affectations, centroids)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter, loss=loss, start_time=start_time)
    return affectations, centroids, np.array(losses)


def _compute_loss(data, affectations, centroids):
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (affectations *
            np.power(np.linalg.norm(dist_data_centroids, axis=-1, ord=2),
                     2).T).sum()


def _optim_affectations(data, centroids):
    # Compute euclidean distance over all data
    dist_data_centroids = data - centroids[:, np.newaxis]
    affectations = np.linalg.norm(dist_data_centroids, ord=2, axis=-1).T

    # Set all affectations
    mask_closest_centroid = (np.arange(len(affectations)),
                             affectations.argmin(1))
    affectations[mask_closest_centroid] = 1
    affectations[affectations != 1] = 0
    return affectations


def _optim_centroids(data, affectations):
    return (np.dot(data.T, affectations) / np.sum(affectations, axis=0)).T


if __name__ == '__main__':
    pass
