import numpy as np


def kmeans(data, k, eps=None, max_iter=1000):
    if eps is None:
        eps = 0.001
    if max_iter is None:
        max_iter = 1000

    # Initialisation
    nb_examples, dim = data.shape
    centroids = np.random.rand(k, dim).astype(np.float64)
    affectations = np.zeros(shape=(nb_examples, k), dtype=np.float64)

    current_iter = 0
    losses = []
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        affectations = _optim_affectations(data, centroids)
        centroids = _optim_centroids(data, affectations)
        loss = _compute_loss(data, affectations, centroids)
        losses.append(loss)
        current_iter += 1
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
