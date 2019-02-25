import numpy as np


def fuzzy_c_means(data, c, fuzzifier=2, eps=None, max_iter=1000):
    if eps is None:
        eps = 0.001
    if max_iter is None:
        max_iter = 1000

    # Initialisation
    nb_examples, dim = data.shape
    centroids = np.random.rand(c, dim).astype(np.float64)
    affectations = np.zeros(shape=(nb_examples, c), dtype=np.float64)

    current_iter = 0
    losses = []
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        print(affectations.shape)
        affectations = _optim_affectations(data, centroids, fuzzifier)
        centroids = _optim_centroids(data, affectations, fuzzifier)
        loss = _compute_loss(data, affectations, centroids, fuzzifier)
        losses.append(loss)
        current_iter += 1
    return affectations, centroids, np.array(losses)


def _compute_loss(data, affectations, centroids, fuzzifier):
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (np.power(affectations, fuzzifier) *
            np.power(np.linalg.norm(dist_data_centroids, axis=-1,
                                    ord=2), 2).T).sum()


def _optim_affectations(data, centroids, fuzzifier):
    for i in range(data.shape[0]):
        for r in range(centroids.shape[-1]):
            bidule = sum([])
            truc = bidule
            data[i][r] = 1 / truc


def _optim_centroids(data, affectations, fuzzifier):
    return (np.dot(data.T, np.power(affectations, fuzzifier)) /
            np.sum(np.power(affectations, fuzzifier), axis=0)).T


if __name__ == '__main__':
    pass
