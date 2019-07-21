import scipy
import sys
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, cdist

from clus.src.utils.array import idx_to_r_elements
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_DEFAULT_MEMBERSHIP_SUBSET_SIZE_PERCENT = 0.1

_LABEL_UNASSIGNED = -2
_LABEL_NOISE = -1
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
    affectations = np.ones(shape=(data.shape[0])) * _LABEL_UNASSIGNED
    clusters_centers = []
    for epoch in range(max_epochs):
        not_affected_data_idx = np.where(affectations == _LABEL_UNASSIGNED)[0]
        if not_affected_data_idx.size < batch_size:
            # No more data (or not enough) to process
            break

        trashcan_mask = np.zeros(shape=(batch_size,), dtype=np.bool)

        # Sample a random batch of new data (or previously discarded data)
        batch_data_idx = np.random.choice(not_affected_data_idx, size=batch_size, replace=False)

        clus_result = fuzzy_c_means(
            data=data[batch_data_idx, :], components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier,
            weights=None, initialization_method=initialization_method,
            empty_clusters_method=empty_clusters_method, progress_bar=False
        )
        losses.append(clus_result["losses"][-1])

        batch_affectations = clus_result["affectations"]
        batch_clusters_centers = clus_result["clusters_center"]
        batch_clusters_id = clus_result["clusters_id"]

        # TODO: verifier que les clusters_cardinal et clusters_diameter sont dans le meme ordre que cluster_id.
        #  Sinon trier.
        # First filter criterion, filter by cluster size
        for i, (cluster_id, cluster_cardinal) in enumerate(zip(batch_clusters_id, clus_result["clusters_cardinal"])):
            if cluster_cardinal < min_centroid_size:
                # Delete the cluster and its data
                trashcan_mask |= batch_affectations == cluster_id
                batch_clusters_id[i] = _CLUSTER_ID_DELETED

        # Second filter criterion, filter by cluster diameter
        for i, (cluster_id, cluster_diameter) in enumerate(zip(batch_clusters_id, clus_result["clusters_diameter"])):
            if cluster_id == _CLUSTER_ID_DELETED:
                continue
            if cluster_diameter > max_centroid_diameter:
                # Delete the cluster and its data
                trashcan_mask |= batch_affectations == cluster_id
                batch_clusters_id[i] = _CLUSTER_ID_DELETED

        # Keep good clusters for HC
        for idx in batch_data_idx:
            good_data_idx.append(idx)

        # Put back bad discarded samples into the pool for the next epoch
        for idx in trashcan:
            affectations = np.append(affectations, idx)

        # Unaffected Fraud Allocation
        # TODO: Itérer sur les données, regarder si chaque donnée non traitée est dans le RAYON (diametre/2) du
        #  centroide d'un des clusters. Si oui, alors l'ajouter.
        clusters_center = clus_result["clusters_center"]

        distance_data_centroids = cdist(data[affectations], clusters_center, metric="euclidean") ** 2
        data_argmax = distance_data_centroids.argmax(axis=1)
        data_max = distance_data_centroids.max(axis=1)

        clusters_radius = clus_result["clusters_diameter"] / 2

        todelete = []
        for i_vector, vector in enumerate(data[affectations]):
            # Retrieve centroids matching their radius condition wrt the data
            mask_centroids = distance_data_centroids[i_vector, :] < clusters_radius
            idx_centroids = np.where(mask_centroids)[0]

            if idx_centroids.size >= 1:
                # Find in them the closest to the data point
                idx_closest_centroid = idx_centroids[distance_data_centroids[i_vector, idx_centroids].argmin()]

                # Assign data point to this centroid
                good_data_idx.append(i_vector)
                affectations[i_vector] = -1
                todelete.append(idx_closest_centroid)

        deleted_data = ~np.ma.masked_equal(affectations, -1).mask
        affectations = affectations[deleted_data]
        print(affectations.shape)
        print(Counter(todelete).most_common())
        exit(0)

    if len(good_data_idx) == 0:
        print("No more good data after filtering. Try lowering the restrictions on the parameters `min_centroid_size` "
              "and `max_centroid_diameter`.", file=sys.stderr)
        linkage_mtx = None
    else:
        # Perform hierarchical clustering on good data
        linkage_mtx = linkage(data[good_data_idx], method="single")

    return {
        "linkage_mtx": linkage_mtx,
        "good_data_idx": good_data_idx,
        "noise_data_idx": affectations,
        "losses": np.array(losses)
    }


def _compute_cluster_diameter(cluster):
    return max(pdist(cluster, metric="euclidean"))


if __name__ == '__main__':
    pass
