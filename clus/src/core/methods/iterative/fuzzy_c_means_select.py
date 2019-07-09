from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.utils.array import mini_batches, mini_batches_idx
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_DEFAULT_MEMBERSHIP_SUBSET_SIZE_PERCENT = 0.1


@remove_unexpected_arguments
def fuzzy_c_means_select(data, components=10, eps=1e-4, max_iter=1000, fuzzifier=2, batch_size=64, weights=None,
                         max_epochs=10, min_centroid_size=3, max_centroid_diameter=1.0,
                         initialization_method="random_choice", empty_clusters_method="nothing",
                         centroids=None):
    assert len(data.shape) == 2, "The data must be a 2D array"
    assert data.shape[0] > 0, "The data must have at least one example"
    assert data.shape[1] > 0, "The data must have at least one feature"
    assert 1 <= components <= data.shape[0], "The number of components wanted must be between 1 and %s" % data.shape[0]
    assert 0 <= max_iter, "The number of max iterations must be positive"
    assert fuzzifier > 1, "The fuzzifier must be greater than 1"
    assert 1 <= batch_size <= data.shape[0], "The batch size muste be between 1 and %d" % data.shape[0]
    assert (weights is None) or (len(weights) == data.shape[1]),\
        "The number of weights given must be the same as the number of features. Expected size : %s, given size : %s" %\
        (data.shape[1], len(weights))
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )
    # TODO: assert for max_epoch, min_centroid_size et max_centroid_diameter

    # Locally import clustering method to prevent cyclic dependencies
    from clus.src.core.methods import fuzzy_c_means

    if weights is not None:
        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
        # weighted by the square root of the weights, see [5]
        data = data * np.sqrt(weights)

    losses = []
    data_to_affect_idx = np.arange(0, data.shape[0])
    good_data_idx = []
    for epoch in range(max_epochs):
        if data_to_affect_idx.shape[0] < batch_size:
            # No more data to process
            break

        trashcan = []

        # Sample a random batch of new data (or previously discarded data)
        batch_data_idx = np.random.choice(data_to_affect_idx, size=batch_size, replace=False)
        data_to_affect_idx = data_to_affect_idx[~np.isin(data_to_affect_idx, batch_data_idx)]

        clus_result = fuzzy_c_means(
            data=data[batch_data_idx, :], components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier,
            weights=None, initialization_method=initialization_method,
            empty_clusters_method=empty_clusters_method
        )

        # First filter criterion, filter by cluster size
        affectations = clus_result["affectations"]
        for i_cluster, n_of_elements in reversed(Counter(affectations).most_common()):
            if n_of_elements >= min_centroid_size:
                break
            if n_of_elements < min_centroid_size:
                _delete_data(batch_data_idx, i_cluster, affectations, trashcan)

        # Update data structures
        affectations = affectations[batch_data_idx != -1]
        batch_data_idx = batch_data_idx[batch_data_idx != -1]

        # Second filter criterion, filter by cluster diameter
        for i_cluster in np.unique(affectations):
            cluster = data[batch_data_idx[affectations == i_cluster]]
            if _compute_cluster_diameter(cluster) > max_centroid_diameter:
                _delete_data(batch_data_idx, i_cluster, affectations, trashcan)

        # Update data structures
        affectations = affectations[batch_data_idx != -1]
        batch_data_idx = batch_data_idx[batch_data_idx != -1]

        # Keep good clusters for HC
        for idx in batch_data_idx:
            good_data_idx.append(idx)

        # Put back bad discarded samples into the pool for the next epoch
        for idx in trashcan:
            data_to_affect_idx = np.append(data_to_affect_idx, idx)

        losses.append(clus_result["losses"][-1])

    # Perform hierarchical clustering on good data
    linkage_mtx = linkage(data[good_data_idx, :])

    return {
        "linkage_mtx": linkage_mtx,
        "good_data_idx": np.sort(good_data_idx),
        "losses": np.array(losses),
    }


def _delete_data(batch_data_idx, i_cluster, affectations, trashcan):
    mask_data_idx_to_delete = affectations == i_cluster
    data_idx_to_delete = batch_data_idx[mask_data_idx_to_delete]
    batch_data_idx[mask_data_idx_to_delete] = -1

    for idx in data_idx_to_delete:
        trashcan.append(idx)
    # batch_data_idx = np.delete(batch_data_idx, trashcan_data, axis=0)


def _compute_cluster_diameter(cluster):
    return max(pdist(cluster, metric="euclidean"))


if __name__ == '__main__':
    pass
