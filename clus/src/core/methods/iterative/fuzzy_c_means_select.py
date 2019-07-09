from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.utils.array import mini_batches
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_DEFAULT_MEMBERSHIP_SUBSET_SIZE_PERCENT = 0.1


@remove_unexpected_arguments
def fuzzy_c_means_select(data, components=10, eps=1e-4, max_iter=1000, fuzzifier=2, weights=None,
                         initialization_method="random_choice", empty_clusters_method="nothing",
                         centroids=None):
    assert len(data.shape) == 2, "The data must be a 2D array"
    assert data.shape[0] > 0, "The data must have at least one example"
    assert data.shape[1] > 0, "The data must have at least one feature"
    assert 1 <= components <= data.shape[0], "The number of components wanted must be between 1 and %s" % data.shape[0]
    assert 0 <= max_iter, "The number of max iterations must be positive"
    assert fuzzifier > 1, "The fuzzifier must be greater than 1"
    assert (weights is None) or (len(weights) == data.shape[1]),\
        "The number of weights given must be the same as the number of features. Expected size : %s, given size : %s" %\
        (data.shape[1], len(weights))
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )

    # Locally import clustering method to prevent cyclic dependencies
    from clus.src.core.methods import fuzzy_c_means

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, initialization_method)


    MAX_EPOCHS = 1
    MIN_CENTROID_SIZE = 10
    MAX_CENTROID_DIAMETER = 100
    for epoch in range(MAX_EPOCHS):
        trashcan_data = []
        trashcan_distance = []
        for batch_data in mini_batches(data, batch_size=batch_size, allow_dynamic_batch_size=True, shuffle=True):
            clus_result = fuzzy_c_means(
                data=data, components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier, weights=weights,
                initialization_method=initialization_method, empty_clusters_method=empty_clusters_method,
                centroids=centroids
            )

            affectations = clus_result["affectations"]
            least_common = reversed(Counter(affectations).most_common())
            for i_cluster, n_of_elements in least_common:
                if n_of_elements > MIN_CENTROID_SIZE:
                    break
                if n_of_elements <= MIN_CENTROID_SIZE:
                    batch_data = _delete_cluster(batch_data, batch_distance, i_cluster, closest_cluster,
                                                 trashcan_data, trashcan_distance)
            # Maximal diameter filtering
            closest_cluster = cdist(data, clusters_center, metric='euclidean').argmin(axis=-1)
            for i_cluster in np.unique(closest_cluster):
                cluster_diameter = _compute_cluster_diameter(batch_data, batch_distance, i_cluster)
                if cluster_diameter > MAX_MEDOID_DIAMETER:
                    batch_data, batch_distance = _delete_cluster(batch_data, batch_distance, i_cluster, closest_cluster,
                                                                 trashcan_data, trashcan_distance)

    # Perform hierarchical clustering on final medoids
    linkage_mtx = linkage(data[medoids_idx, :])
    return linkage_mtx


def _delete_cluster(data, distance_mtx, i_cluster, closest_cluster, trashcan_data, trashcan_distance):
    idx_cluster = np.array(closest_cluster == i_cluster, dtype=np.uint8)

    trashcan_data.append(data[idx_cluster])
    trashcan_distance.append(distance_mtx[idx_cluster][:, idx_cluster])

    data = np.delete(data, idx_cluster, axis=0)
    distance_mtx = np.delete(distance_mtx, idx_cluster, axis=0)
    distance_mtx = np.delete(distance_mtx, idx_cluster, axis=1)
    return data, distance_mtx


def _compute_cluster_diameter(batch_data, batch_distance, i_cluster):
    pass


if __name__ == '__main__':
    pass
