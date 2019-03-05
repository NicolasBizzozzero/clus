import time

import numpy as np

from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def kmeans(data, components, eps, max_iter):
    # Initialisation
    nb_examples, dim = data.shape
    centroids = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0),
                                  size=(components, dim)).astype(np.float64)
    print(centroids)
    centroids = np.random.rand(components, dim).astype(np.float64)
    print(centroids)
    centroids = np.random.normal(loc=data.mean(axis=0), scale=data.std(axis=0),
                                 size=(components, dim)).astype(np.float64)
    print(centroids)
    affectations = None

    current_iter = 0
    losses = []
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        affectations = _optim_affectations(data, centroids)

        # TODO: Corriger ce bug
        if 0 in np.sum(affectations, axis=0):
            print(np.sum(affectations, axis=0))
            return affectations, centroids, np.array(losses)

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
    """

    Source :
    * https://codereview.stackexchange.com/questions/61598/k-mean-with-numpy
    """
    # Compute euclidean distance between data and centroids
    # dist_data_centroids = np.array([np.linalg.norm(data - c, ord=2, axis=1) for c in centroids]).T
    # dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1).T
    dist_data_centroids = np.linalg.norm(np.expand_dims(data, 2) - np.expand_dims(centroids.T, 0), axis=1)

    # Set all binary affectations
    mask_closest_centroid = (np.arange(data.shape[0]), dist_data_centroids.argmin(axis=1))
    affectations = np.zeros(shape=dist_data_centroids.shape, dtype=np.int32)
    affectations[mask_closest_centroid] = 1

    return affectations


def _optim_centroids(data, affectations):
    # TODO: np.sum(affectations, axis=0) sometimes contains 0, bug appearing when there is too many clusters and one do
    #  not contains any example
    print(np.sum(affectations, axis=0))
    return (np.dot(data.T, affectations) / np.sum(affectations, axis=0)).T


def _handle_empty_clusters(data, affectations, centroids, strategy):
    if strategy == "nothing":
        pass
    elif strategy == "random_example":
        pass
    elif strategy == "furthest_example_from_its_centroid":
        pass


if __name__ == '__main__':
    pass
