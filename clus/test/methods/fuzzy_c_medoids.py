import scipy

import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.methods.fuzzy_c_medoids import _compute_medoids_square, _compute_loss_square, \
    _compute_memberships_square, __compute_medoids_square, __compute_loss_square, __compute_memberships_square, \
    _compute_memberships_condensed, _compute_medoids_condensed, _compute_loss_condensed
from clus.src.core.cluster_initialization import random_choice_idx
from clus.src.utils.random import set_manual_seed
from clus.src.utils.array import square_idx_to_condensed_idx


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
    medoids0 = random_choice_idx(data, components=components)

    memberships_condensed = _compute_memberships_condensed(distance_matrix_condensed, medoids0, fuzzifier)
    memberships_square = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)

    print(memberships_condensed.shape)
    print(memberships_square.shape)
    print(memberships_condensed)
    print(memberships_square)

    assert np.all(np.isclose(memberships_condensed.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships_square, memberships_condensed))


def test_compute_medoids_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)
    true_memberships = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)

    true_medoids = _compute_medoids_square(distance_matrix, true_memberships, fuzzifier)
    medoids = _compute_medoids_condensed(distance_matrix_condensed, true_memberships, fuzzifier)

    assert np.all(np.isclose(medoids, true_medoids))


def test_compute_loss_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier):
    medoids0 = random_choice_idx(data, components=components)
    true_memberships = _compute_memberships_square(distance_matrix, medoids0, fuzzifier)
    true_medoids = _compute_medoids_square(distance_matrix, true_memberships, fuzzifier)

    true_loss = _compute_loss_square(distance_matrix, true_medoids, true_memberships, fuzzifier)
    loss = _compute_loss_condensed(distance_matrix_condensed, true_medoids, true_memberships, fuzzifier)

    assert np.isclose(true_loss, loss)


if __name__ == "__main__":
    components = 3
    fuzzifier = 2.0
    data = load_iris().data
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)
    distance_matrix_condensed = scipy.spatial.distance.pdist(data, "euclidean")

    for seed in range(1000):
        print(seed)
        set_manual_seed(seed)
        # test_compute_memberships_square(data, distance_matrix, components, fuzzifier)
        # test_compute_medoids_square(data, distance_matrix, components, fuzzifier)
        # test_compute_loss_square(data, distance_matrix, components, fuzzifier)
        test_compute_memberships_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)
        # test_compute_medoids_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)
        # test_compute_loss_condensed(data, distance_matrix, distance_matrix_condensed, components, fuzzifier)

