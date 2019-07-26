import scipy
import sys
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, cdist

from clus.src.utils.array import idx_to_r_elements
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_LABEL_UNASSIGNED = -1
_CLUSTER_ID_DELETED = -1


@remove_unexpected_arguments
def fuzzy_c_means_select(data, components=1000, eps=1e-4, max_iter=100, fuzzifier=2, batch_size=16_384, weights=None,
                         max_epochs=32, min_centroid_size=10, max_centroid_diameter=1.0,
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

    # affectations: Current affectations of the data to the clusters.
    # .. TODO: remplir

    losses = []
    affectations = np.ones(shape=(data.shape[0]), dtype=np.int64) * _LABEL_UNASSIGNED
    clusters_centers = []
    for epoch in range(max_epochs):
        min_cluster_id = epoch * components
        not_affected_data_idx = np.where(affectations == _LABEL_UNASSIGNED)[0]

        if not_affected_data_idx.size < batch_size:
            # No more data (or not enough) to process
            break

        # Sample a random batch of new data (or previously discarded data)
        batch_data_idx = np.random.choice(not_affected_data_idx, size=batch_size, replace=False)

        clus_result = fuzzy_c_means(
            data=data[batch_data_idx, :], components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier,
            weights=None, initialization_method=initialization_method,
            empty_clusters_method=empty_clusters_method, progress_bar=False
        )
        losses.append(clus_result["losses"][-1])

        batch_affectations = clus_result["affectations"] + min_cluster_id
        batch_clusters_centers = clus_result["clusters_center"]
        batch_clusters_id = clus_result["clusters_id"] + min_cluster_id

        # TODO: verifier que les clusters_cardinal et clusters_diameter sont dans le meme ordre que cluster_id dans la
        #   fonction des fcm. Sinon trier.
        # First filter criterion, filter by cluster size
        for i, (cluster_id, cluster_cardinal) in enumerate(zip(batch_clusters_id, clus_result["clusters_cardinal"])):
            if cluster_cardinal < min_centroid_size:
                # Delete the cluster and its data
                batch_affectations[batch_affectations == cluster_id] = _LABEL_UNASSIGNED
                batch_clusters_id[i] = _CLUSTER_ID_DELETED

        # Second filter criterion, filter by cluster diameter
        for i, (cluster_id, cluster_diameter) in enumerate(zip(batch_clusters_id, clus_result["clusters_diameter"])):
            if cluster_id == _CLUSTER_ID_DELETED:
                continue
            if cluster_diameter > max_centroid_diameter:
                # Delete the cluster and its data
                batch_affectations[batch_affectations == cluster_id] = _LABEL_UNASSIGNED
                batch_clusters_id[i] = _CLUSTER_ID_DELETED

        # Keep good clusters for HC
        batch_good_clusters_id = batch_clusters_id[batch_clusters_id != _CLUSTER_ID_DELETED]
        batch_good_clusters_centers = batch_clusters_centers[batch_good_clusters_id - min_cluster_id]
        for cluster_center in batch_good_clusters_centers:
            clusters_centers.append(cluster_center)

        # Put back bad discarded samples into the pool for the next epoch
        affectations[batch_data_idx[np.where(batch_affectations != _LABEL_UNASSIGNED)[0]]] = _LABEL_UNASSIGNED

        # Unaffected Fraud Allocation
        unassigned_data_idx = np.where(affectations == _LABEL_UNASSIGNED)[0]
        unassigned_data = data[unassigned_data_idx]
        distance_data_centroids = cdist(unassigned_data, batch_good_clusters_centers, metric="euclidean")

        batch_good_clusters_radius = clus_result["clusters_diameter"][batch_good_clusters_id - min_cluster_id] / 2
        for i in range(len(unassigned_data_idx)):
            # Retrieve centroids matching their radius condition wrt the data
            mask_centroids = distance_data_centroids[i, :] < batch_good_clusters_radius
            idx_centroids = np.where(mask_centroids)[0]

            if idx_centroids.size >= 1:
                # Find in them the closest to the data point
                idx_closest_centroid = idx_centroids[distance_data_centroids[i, idx_centroids].argmin()]
                idx_closest_centroid = batch_good_clusters_id[idx_closest_centroid]

                # Assign data point to this centroid
                affectations[i] = idx_closest_centroid

    clusters_centers = np.array(clusters_centers)
    if len(clusters_centers) == 0:
        print("No good clusters centers found after filtering. Try lowering the restrictions on the parameters "
              "`min_centroid_size` and `max_centroid_diameter`.", file=sys.stderr)
        linkage_mtx = None
    else:
        # Perform hierarchical clustering on clusters' centers
        condensed_distance_matrix = pdist(clusters_centers, metric="euclidean")
        linkage_mtx = linkage(condensed_distance_matrix, method="single")

    return {
        "linkage_mtx": linkage_mtx,
        "affectations": affectations,
        "clusters_centers": clusters_centers,
        "noise_data_idx": np.where(affectations == _LABEL_UNASSIGNED),
        "losses": np.array(losses)
    }


def _compute_cluster_diameter(cluster):
    return max(pdist(cluster, metric="euclidean"))


if __name__ == '__main__':
    pass
