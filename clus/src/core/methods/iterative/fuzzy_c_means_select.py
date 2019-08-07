import sys

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, cdist
from tqdm import tqdm

from clus.src.utils.array import flatten_id
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_epochs, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_LABEL_UNASSIGNED = -1
_CLUSTER_ID_DELETED = -1


@remove_unexpected_arguments
def fuzzy_c_means_select(data, components=1000, eps=1e-4, max_iter=100, fuzzifier=2, batch_size=16_384, weights=None,
                         max_epochs=32, min_centroid_size=10, max_centroid_diameter=1.0, linkage_method="simple",
                         initialization_method="random_choice", empty_clusters_method="nothing",
                         centroids=None, progress_bar=True):
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
    assert 1 <= max_epochs, "The maximal number of epochs must be of at least 1"
    assert (min_centroid_size is None) or (0 <= min_centroid_size), "`min_centroid_size` must be positive"
    assert 0 < max_centroid_diameter, "`max_centroid_diameter` must be strictly positive"
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )

    # Locally import clustering method to prevent cyclic dependencies
    from clus.src.core.methods import fuzzy_c_means

    if weights is not None:
        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
        # weighted by the square root of the weights, see [5]
        data = data * np.sqrt(weights)

    if min_centroid_size is None:
        min_centroid_size = int(np.floor(batch_size / components))

    # affectations (N,) : Current affectations of the data to the clusters. An affectation of `-1` means the data have
    #   not yet been affected to a cluster. If a data point is still affected to the `-1` cluster after all epochs, it
    #   is interpreted as an outlier.
    # clusters_centers (C, D) : Data points matching good clusters found during each epoch.
    affectations = np.ones(shape=(data.shape[0]), dtype=np.int64) * _LABEL_UNASSIGNED
    clusters_centers = []
    stats_epoch = {
        "clusters_found_per_epoch": [],
        "affected_data_per_epoch": []
    }
    with tqdm(total=max_epochs, bar_format=_FORMAT_PROGRESS_BAR, disable=not progress_bar) as progress_bar:
        # min_cluster_id int : Minimal ID a cluster can have for this epoch. We don't want clusters' ID to overlap after
        #   each epoch.
        # not_affected_data_idx (C, D) : Indexes of data points not assigned to a cluster.
        # batch_data_idx (B,) : Indexes of the current batch data points'.
        # batch_affectations (B,) : Affectations of the current batch data points'.
        # batch_clusters_centers (<<C, D) : Data points matching clusters found for this epoch.
        # batch_clusters_id (<<C,) : ID of the clusters for this current batch. If an ID is set to `-1`, this cluster
        #   has been deleted.
        # batch_good_clusters_id (<<C,) : ID of the clusters not deleted for this current batch.
        # batch_good_clusters_centers (<<C, D) : Data points matching good clusters found for this epoch.
        for epoch in range(max_epochs):
            min_cluster_id = epoch * components
            not_affected_data_idx = np.where(affectations == _LABEL_UNASSIGNED)[0]

            if not_affected_data_idx.size < components:
                # No more data to process with this number of components
                break
            if not_affected_data_idx.size < batch_size:
                # No more data to process for a batch
                batch_size = not_affected_data_idx.size

            # Sample a random batch of new data (or previously discarded data)
            batch_data_idx = np.random.choice(not_affected_data_idx, size=batch_size, replace=False)

            clus_result = fuzzy_c_means(
                data=data[batch_data_idx, :], components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier,
                weights=None, initialization_method=initialization_method,
                empty_clusters_method=empty_clusters_method, progress_bar=False
            )

            batch_affectations = clus_result["affectations"] + min_cluster_id
            batch_clusters_centers = clus_result["clusters_center"]
            batch_clusters_id = clus_result["clusters_id"] + min_cluster_id

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
            if batch_good_clusters_id.size == 0:
                # No good clusters found for this epoch
                continue
            batch_good_clusters_centers = batch_clusters_centers[batch_good_clusters_id - min_cluster_id]

            for cluster_center in batch_good_clusters_centers:
                clusters_centers.append(cluster_center)

            # Update affectations found for the current batch
            not_deleted_batch_data_mask = batch_affectations != _LABEL_UNASSIGNED
            affectations[batch_data_idx[not_deleted_batch_data_mask]] = batch_affectations[not_deleted_batch_data_mask]

            _unaffected_data_allocation(data=data, affectations=affectations,
                                        batch_good_clusters_centers=batch_good_clusters_centers,
                                        batch_good_clusters_id=batch_good_clusters_id, min_cluster_id=min_cluster_id,
                                        clus_result=clus_result)

            # Update clustering info per epoch
            stats_epoch["clusters_found_per_epoch"].append(np.array(clusters_centers).shape[0])
            stats_epoch["affected_data_per_epoch"].append((affectations != _LABEL_UNASSIGNED).sum())

            # Update the progress bar
            progress_bar.update()
            progress_bar.set_postfix({
                "clusters_found": stats_epoch["clusters_found_per_epoch"][-1],
                "affected_data": "{}/{}".format(stats_epoch["affected_data_per_epoch"][-1], affectations.shape[0])
            })

    affectations = flatten_id(affectations)

    clusters_centers = np.array(clusters_centers)
    if len(clusters_centers) == 0:
        print("No good clusters centers found after filtering. Try lowering the restrictions on the parameters "
              "`min_centroid_size` and `max_centroid_diameter`.", file=sys.stderr)
        linkage_matrix = None
    else:
        # Perform hierarchical clustering on clusters' centers
        distance_matrix = pdist(clusters_centers, metric="euclidean")
        linkage_matrix = linkage(distance_matrix, method=linkage_method)

    return {
        "linkage_matrix": linkage_matrix,
        "affectations": affectations,
        "clusters_centers": clusters_centers,
        "noise_data_idx": np.where(affectations == _LABEL_UNASSIGNED),
        "clusters_found": clusters_centers.shape[0],
        "affected_data": (affectations != _LABEL_UNASSIGNED).sum(),
        **stats_epoch
    }


def _unaffected_data_allocation(data, affectations, batch_good_clusters_centers, batch_good_clusters_id, min_cluster_id,
                                clus_result):
    """ Assign unaffected data to an existing cluster center if it does not increase its diameter. """
    global _LABEL_UNASSIGNED

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


if __name__ == '__main__':
    pass
