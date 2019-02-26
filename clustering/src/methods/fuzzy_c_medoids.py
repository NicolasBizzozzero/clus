"""
https://github.com/agartland/utils/blob/e200d7e41039ca0053bd817c1d1857aab33bd503/kmedoids.py#L243

"""
import numpy as np
from sklearn.neighbors import DistanceMetric

from clustering.src.utils import remove_unexpected_arguments


@remove_unexpected_arguments
def fuzzy_c_medoids(data, components, fuzzifier, eps, max_iter):
    # Initialisation
    nb_examples, dim = data.shape
    medoids_idx = np.random.choice(range(nb_examples), size=components)
    medoids = data[medoids_idx, :]
    distance_matrix = DistanceMetric.get_metric('euclidean').pairwise(data)
    print("Data shape            :", data.shape)
    print("Medoids shape         :", medoids.shape)
    print("Medoids idx shape     :", medoids_idx.shape)
    print("Distance matrix shape :", distance_matrix.shape)

    # memberships = np.zeros(shape=(nb_examples, components), dtype=np.float64)

    current_iter = 0
    losses = []
    while (current_iter <= max_iter) and \
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):
        memberships = _compute_memberships(distance_matrix, medoids_idx, fuzzifier)
        print(memberships.shape)
        print(memberships)
        exit(0)
        _optim_medoids(data, medoids, memberships, fuzzifier)

        loss = _compute_loss(data, memberships, medoids, fuzzifier)
        losses.append(loss)
        current_iter += 1
    return memberships, medoids, np.array(losses)


def _compute_loss(data, memberships, medoids, fuzzifier):
    pass


def _compute_memberships(distance_matrix, medoids_idx, fuzzifier):
    r = distance_matrix[:, medoids_idx]
    print(r)
    print(1 / r)  # TODO: Comment gérer les 0 (petit epsilon partout ou juste sur les 0) ou même supprimer les doublons
                  # TODO: Est-ce que je normalise mes données ?
                  # TODO: La matrice de dissimilarité, on la fournie ou je dois la calculer ? La dissimilarité est symétrique ?
    exit(0)

    tmp = (1 / r) ** (1 / (fuzzifier - 1))

    memberships = tmp / tmp.sum(axis=1, keepdims=True)
    for index_medoid, medoid in enumerate(medoids_idx):
        memberships[medoid, :] = 0.
        memberships[medoid, index_medoid] = 1.
    #memberships[np.isnan(memberships)] = 1
    return memberships


def _optim_medoids(data, medoids, memberships, fuzzifier):
    pass


if __name__ == '__main__':
    pass
