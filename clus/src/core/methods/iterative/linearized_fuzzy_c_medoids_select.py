from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage

from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.utils.array import mini_batches_dist
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"


@remove_unexpected_arguments
def linearized_fuzzy_c_medoids_select(data, distance_matrix, components=1000,
                                      eps=1e-4, max_iter=100, fuzzifier=2,
                                      batch_size=64, membership_subset_size=None,
                                      initialization_method="random_choice",
                                      empty_clusters_method="nothing",
                                      medoids_idx=None):
    """ Performs the linearized fuzzy c-medoids select clustering algorithm on a dataset.

    :param data: The dataset into which the clustering will be performed. The dataset must be 2D np.array with rows as
    examples and columns as features.
    :param distance_matrix: The pairwise distance matrix applied across all examples from the data matrix. The distance
    matrix must be a square matrix.
    :param components: The number of components (clusters) wanted.
    :param eps: Criterion used to define convergence. If the absolute differences between two consecutive losses is
    lower than `eps`, the clustering stop.
    :param max_iter: Criterion used to stop the clustering if the number of iterations exceeds `max_iter`.
    :param fuzzifier: Membership fuzzification coefficient.
    :param batch_size: Number of examples randomly sampled at each iteration of the iterative step.
    :param membership_subset_size: Size of subset to inspect during the memberships matrix computation. Reduce
    computations length.
    :param initialization_method: Method used to initialise the centroids. Can take one of the following values :
    * "random_uniform" or "uniform", samples values between the min and max across each dimension.
    * "random_gaussian" or "gaussian", samples values from a gaussian with the same mean and std as each data's
    dimension.
    * "random_choice" or "choice", samples random examples from the data without replacement.
    * "central_dissimilar_medoids", sample the first medoid as the most central point of the dataset, then sample all
    successive medoids as the most dissimilar to all medoids that have already been picked.
    * "central_dissimilar_random_medoids", same as "central_dissimilar_medoids", but the first medoid is sampled
    randomly.
    :param empty_clusters_method: Method used at each iteration to handle empty clusters. Can take one of the following
    values :
    * "nothing", do absolutely nothing and ignore empty clusters.
    * "random_example", assign a random example to all empty clusters.
    * "furthest_example_from_its_centroid", assign the furthest example from its centroid to each empty cluster.
    :param medoids_idx: Initials medoids indexes to use instead of randomly initialize them.
    :return: A tuple containing :
    * The memberships matrix.
    * The medoids matrix.
    * An array with all losses at each iteration.
    """
    assert len(data.shape) == 2, "The data must be a 2D array"
    assert data.shape[0] > 0, "The data must have at least one example"
    assert data.shape[1] > 0, "The data must have at least one feature"
    assert (len(distance_matrix.shape) == 2) and (distance_matrix.shape[0] == distance_matrix.shape[1]), \
        "The distance matrix is not a square matrix"
    assert 1 <= components <= data.shape[0], "The number of components wanted must be between 1 and %s" % data.shape[0]
    assert 0 <= max_iter, "The number of max iterations must be positive"
    assert fuzzifier > 1, "The fuzzifier must be greater than 1"
    assert 1 <= batch_size <= data.shape[0], "The batch size muste be between 1 and %d" % data.shape[0]
    assert (membership_subset_size is None) or (1 <= membership_subset_size <= data.shape[0]), \
        "The membership subset size wanted must be between 1 and %s" % data.shape[0]
    assert (medoids_idx is None) or (medoids_idx.shape == components), \
        "The given medoids indexes do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components,), medoids_idx.shape
        )
    assert (medoids_idx is None) or np.all(medoids_idx < data.shape[0]), \
        "The provided medoid indexes array contains unreachable indexes"

    # Locally import clustering method to prevent cyclic dependencies
    from clus.src.core.methods import linearized_fuzzy_c_medoids

    # If no `membership_subset_size` is specified, [1] suggest to use a value much smaller than the average of points
    # in a cluster
    if membership_subset_size is None:
        membership_subset_size = distance_matrix.shape[0] // components

    # Initialisation
    if medoids_idx is None:
        medoids_idx = cluster_initialization(distance_matrix, components, initialization_method, need_idx=True)

    MAX_EPOCHS = 1
    MIN_MEDOID_SIZE = 4
    MAX_MEDOID_DIAMETER = 1000
    for epoch in range(MAX_EPOCHS):
        trashcan_data = []
        trashcan_distance = []
        for batch_data, batch_distance in mini_batches_dist(data, distance_matrix, batch_size=batch_size,
                                                            allow_dynamic_batch_size=True, shuffle=True):
            memberships, medoids_idx, losses = linearized_fuzzy_c_medoids(
                data=batch_data, distance_matrix=batch_distance, components=components, eps=eps, max_iter=max_iter,
                fuzzifier=fuzzifier, membership_subset_size=membership_subset_size,
                initialization_method=initialization_method, empty_clusters_method=empty_clusters_method,
                medoids_idx=medoids_idx
            )

            # TODO: I previously used data points as clusters_center, I updated theses values to data indexes.
            #  So I need to updated all the code below, and probably not use the "closest_cluster"
            #  computation but the distance matrix.
            # Minimal cardinal filtering
            closest_cluster = np.linalg.norm(data - clusters_center[:, np.newaxis], axis=-1, ord=2).argmin(axis=0)
            least_common = reversed(Counter(closest_cluster).most_common())
            for i_cluster, n_of_elements in least_common:
                if n_of_elements > MIN_MEDOID_SIZE:
                    break
                if n_of_elements <= MIN_MEDOID_SIZE:
                    batch_data, batch_distance = _delete_cluster(batch_data, batch_distance, i_cluster, closest_cluster,
                                                                 trashcan_data, trashcan_distance)
            # Maximal diameter filtering
            closest_cluster = np.linalg.norm(data - clusters_center[:, np.newaxis], axis=-1, ord=2).argmin(axis=0)
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
