import numpy as np
from scipy.spatial.distance import cdist

from sklearn.datasets import load_iris

from clus.src.core.methods.partition_based.fuzzy_c_means import _compute_centroids, _compute_loss, _compute_memberships, __compute_centroids, __compute_loss, __compute_memberships
from clus.src.core.cluster_initialization import random_choice


def test_compute_memberships():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data

    centroids0 = random_choice(data, components=components)

    true_memberships = __compute_memberships(data, centroids0, fuzzifier)
    memberships = _compute_memberships(data, centroids0, fuzzifier)

    assert np.all(np.isclose(true_memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships.sum(axis=1), np.ones((1, data.shape[0]))))
    assert np.all(np.isclose(memberships, true_memberships))


def test_compute_centroids():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data

    centroids0 = random_choice(data, components=components)

    true_memberships = __compute_memberships(data, centroids0, fuzzifier)

    true_centroids = __compute_centroids(data, true_memberships, fuzzifier)
    centroids = _compute_centroids(data, true_memberships, fuzzifier)

    assert np.all(np.isclose(centroids, true_centroids))


def test_compute_loss():
    components = 3
    fuzzifier = 2.0

    data = load_iris().data

    centroids0 = random_choice(data, components=components)

    true_memberships = __compute_memberships(data, centroids0, fuzzifier)
    true_centroids = __compute_centroids(data, true_memberships, fuzzifier)

    true_loss = __compute_loss(data, true_memberships, true_centroids, fuzzifier)
    loss = _compute_loss(data, true_memberships, true_centroids, fuzzifier)
    loss_skfuzzy = compute_loss_skfuzzy(data.T, true_memberships.T, components, fuzzifier)

    print("loss true    :", true_loss)
    print("loss clus    :", compute_loss_clus(data, true_memberships, true_centroids, fuzzifier))
    print("loss skfuzzy :", loss_skfuzzy)

    assert loss_skfuzzy == 104.12778851148951

    assert np.isclose(true_loss, loss)


def compute_loss_clus(data, memberships, centroids, fuzzifier):
    dist_data_centroids = cdist(data, centroids, metric="euclidean") ** 2
    return ((memberships ** fuzzifier) * dist_data_centroids).sum()


def compute_loss_skfuzzy(data, memberships, components, fuzzifier):
    # Normalizing, then eliminating any potential zero values.
    memberships /= np.ones((components, 1)).dot(np.atleast_2d(memberships.sum(axis=0)))
    memberships = np.fmax(memberships, np.finfo(np.float64).eps)

    um = memberships ** fuzzifier

    # Calculate cluster centers
    data = data.T
    centers = um.dot(data) / np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T

    d = cdist(data, centers).T

    return (um * d ** 2).sum()


if __name__ == "__main__":
    test_compute_loss()
