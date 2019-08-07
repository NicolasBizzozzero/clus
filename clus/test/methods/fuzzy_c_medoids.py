import scipy

import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.methods.partition_based.fuzzy_c_medoids import _compute_medoids_square, _compute_loss_square, \
    _compute_memberships_square, __compute_medoids_square, __compute_loss_square, __compute_memberships_square, \
    _compute_memberships_condensed, _compute_medoids_condensed, _compute_loss_condensed
from clus.src.core.cluster_initialization import random_choice_idx
from clus.src.utils.distance_matrix import square_row_idx_to_condensed_row, square_rows_idx_to_condensed_rows
from clus.src.utils.random import set_manual_seed


def test_compute_memberships_square(data, distance_matrix, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships_square(distance_matrix, medoids0, fuzzifier)
    memberships = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)

    assert np.all(np.isclose(true_memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships, true_memberships))


def test_compute_medoids_square(data, distance_matrix, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships_square(distance_matrix, medoids0, fuzzifier)

    true_medoids = __compute_medoids_square(distance_matrix, true_memberships, fuzzifier)
    medoids = _compute_medoids_square(distance_matrix, true_memberships, fuzzifier)

    assert np.all(np.isclose(medoids, true_medoids))


def test_compute_loss_square(data, distance_matrix, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships_square(distance_matrix, medoids0, fuzzifier)
    true_medoids = __compute_medoids_square(distance_matrix, true_memberships, fuzzifier)

    true_loss = __compute_loss_square(distance_matrix, true_medoids, true_memberships, fuzzifier)
    loss = _compute_loss_square(distance_matrix, true_medoids, true_memberships, fuzzifier)

    assert np.isclose(true_loss, loss)


def test_compute_memberships_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier):
    medoids0_idx = random_choice_idx(data, components=components)

    memberships_condensed = _compute_memberships_condensed(distance_matrix_condensed, medoids0_idx, fuzzifier,
                                                           n=data.shape[0])
    memberships_square = _compute_memberships_square(distance_matrix, medoids0_idx, fuzzifier)

    assert np.all(np.isclose(memberships_condensed.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships_square, memberships_condensed))


def test_compute_medoids_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)
    true_memberships = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)

    true_medoids = _compute_medoids_square(distance_matrix, true_memberships, fuzzifier)
    medoids = _compute_medoids_condensed(distance_matrix_condensed, true_memberships, fuzzifier, n=data.shape[0])

    print(true_medoids.shape)
    print(medoids.shape)
    exit(0)

    assert np.all(np.isclose(medoids, true_medoids))


def test_compute_loss_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)
    true_memberships = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)
    true_medoids = _compute_medoids_square(distance_matrix, true_memberships, fuzzifier)

    true_loss = _compute_loss_square(distance_matrix, true_medoids, true_memberships, fuzzifier)
    loss = _compute_loss_condensed(distance_matrix_condensed, true_medoids, true_memberships, fuzzifier,
                                   n=data.shape[0])

    assert np.isclose(true_loss, loss)


if __name__ == "__main__":
    components = 3
    fuzzifier = 2.0
    data = load_iris().data[:5, :]
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)
    distance_matrix_condensed = scipy.spatial.distance.pdist(data, "euclidean")

    for seed in range(1000):
        print("Seed", seed)
        set_manual_seed(seed)
        test_compute_memberships_square(data, distance_matrix, components, fuzzifier)
        test_compute_medoids_square(data, distance_matrix, components, fuzzifier)
        test_compute_loss_square(data, distance_matrix, components, fuzzifier)
        test_compute_memberships_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)
        test_compute_medoids_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)
        test_compute_loss_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)
