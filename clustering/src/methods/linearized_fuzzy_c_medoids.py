import time

import numpy as np
from clustering.src.handle_empty_clusters import handle_empty_clusters
from scipy.sparse import csr_matrix

from clustering.src.initialization import cluster_initialization
from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def linearized_fuzzy_c_medoids(distance_matrix, components, fuzzifier, membership_subset_size, eps, max_iter,
                               initialization_method, empty_clusters_method):
    assert (len(distance_matrix.shape) == 2) and distance_matrix.shape[0] == distance_matrix.shape[1], "The distance matrix is not squared"
    assert initialization_method in (2, 3, 4), "Your initialization method must be based on example selection"

    # If no `membership_subset_size` is specified, [1] suggest to use a value much smaller than the average of points
    # in a cluster
    if membership_subset_size is None:
        membership_subset_size = distance_matrix.shape[0] // components

    # Initialisation
    medoids_idx = cluster_initialization(distance_matrix, components, initialization_method, need_idx=True)

    memberships = None
    current_iter = 0
    losses = []
    medoids_idx_old = None
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 1) or (not all(medoids_idx == medoids_idx_old))) and\
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):

        medoids_idx_old = medoids_idx
        memberships = _compute_memberships(distance_matrix, medoids_idx, fuzzifier)
        handle_empty_clusters(distance_matrix, medoids_idx, memberships, strategy=empty_clusters_method)

        top_memberships_mask = _compute_top_membership_subset(memberships, membership_subset_size)
        medoids_idx = _compute_medoids(distance_matrix, memberships, fuzzifier, top_memberships_mask)

        loss = _compute_loss(distance_matrix, medoids_idx, memberships, fuzzifier)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter, loss=loss, start_time=start_time)
    return memberships, distance_matrix[medoids_idx, :], np.array(losses)


def _compute_memberships(data, medoids_idx, fuzzifier):
    r = data[:, medoids_idx]

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    r += np.finfo(data.dtype).eps

    tmp = (1 / r) ** (1 / (fuzzifier - 1))
    memberships = tmp / tmp.sum(axis=1, keepdims=True)

    for index_medoid, medoid in enumerate(medoids_idx):
        memberships[medoid, :] = 0.
        memberships[medoid, index_medoid] = 1.

    return memberships


def _compute_top_membership_subset(memberships, membership_subset_size):
    """ Compute a mask of the `memberships` matrix. The mask is True `membership_subset_size` times in each column for
    all indexes where the membership has one of the `membership_subset_size` highest value for this column.

    Many thanks to RISSER-MAROIX Olivier and POUYET Adrien for their help.
    """
    topk_idx = np.argpartition(memberships, -membership_subset_size, axis=0)[-membership_subset_size:]

    # TODO: Sparse matrix may be faster, but it could not be used for the medoids computation because the other matrix
    # has 3 dimensions, and it is currently not possible to do matrix multiplication with a sparse matrix and a
    # more-than-2-dimensions matrix. See : https://github.com/scipy/scipy/blob/master/scipy/sparse/base.py#L527
    # Thus we convert it to a traditional matrix with the `.toarray()` operation.
    top_memberships_mask = \
        csr_matrix((np.ones((topk_idx.shape[0] * topk_idx.shape[1],)),
                    (topk_idx.flatten(), np.nonzero(abs(topk_idx) + 1)[1])),
                   shape=memberships.shape,
                   dtype=bool).toarray()

    # Return the subset of the top memberships (the result of the application of the mask).
    # Not used here but might be useful. Many thanks to POUYET Adrien for its help.
    # top_memberships_subset = memberships[topk_idx, np.arange(memberships.shape[1])]

    return top_memberships_mask


def _compute_medoids(data, memberships, fuzzifier, top_memberships_mask):
    return (data[..., np.newaxis] * top_memberships_mask * (memberships ** fuzzifier)).sum(axis=1).argmin(axis=0)


def _compute_loss(data, medoids_idx, memberships, fuzzifier):
    return ((memberships ** fuzzifier) * data[:, medoids_idx]).sum(axis=(1, 0))


if __name__ == '__main__':
    pass
