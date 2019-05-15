import numpy as np
from scipy.spatial.distance import cdist

from sklearn.datasets import load_iris
from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.methods.fuzzy_c_medoids import fuzzy_c_medoids, _compute_medoids, _compute_loss, _compute_memberships, __compute_medoids, __compute_loss, __compute_memberships
from clus.src.core.cluster_initialization import random_choice_idx
from clus.src.utils.random import set_manual_seed


def test_compute_memberships():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)

    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships(distance_matrix, medoids0, fuzzifier)
    memberships = _compute_memberships(distance_matrix, medoids0, fuzzifier)

    assert np.all(np.isclose(true_memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships, true_memberships))


def test_compute_medoids():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)

    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships(distance_matrix, medoids0, fuzzifier)

    true_medoids = __compute_medoids(distance_matrix, true_memberships, fuzzifier)
    medoids = _compute_medoids(distance_matrix, true_memberships, fuzzifier)

    assert np.all(np.isclose(medoids, true_medoids))


def test_compute_loss():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)

    medoids0 = random_choice_idx(data, components=components)

    true_memberships = __compute_memberships(distance_matrix, medoids0, fuzzifier)
    true_medoids = __compute_medoids(distance_matrix, true_memberships, fuzzifier)

    true_loss = __compute_loss(distance_matrix, true_memberships, true_medoids, fuzzifier)
    loss = _compute_loss(distance_matrix, true_memberships, true_medoids, fuzzifier)

    print("loss true    :", true_loss)
    print("loss clus    :", loss)

    assert np.isclose(true_loss, loss)


if __name__ == "__main__":
    for seed in range(10000):
        print(seed)
        set_manual_seed(seed)
        test_compute_medoids()
