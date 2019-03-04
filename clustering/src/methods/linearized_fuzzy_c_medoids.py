import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

from clustering.src.utils import remove_unexpected_arguments, print_progression


@remove_unexpected_arguments
def linearized_fuzzy_c_medoids(data, components, fuzzifier,
                               membership_subset_size, eps, max_iter):
    # The data matrix must be a square matrix
    assert (len(data.shape) == 2) and data.shape[0] == data.shape[1]

    # If no `membership_subset_size` is specified, [1] suggest to use a value much smaller than the average of points
    # in a cluster
    if membership_subset_size is None:
        membership_subset_size = data.shape[0] // components

    # Initialisation
    nb_examples, dim = data.shape
    memberships = np.zeros(shape=(nb_examples, components), dtype=np.float64)
    medoids_idx = _init_medoids(data, nb_examples, components, selection_method="random")

    current_iter = 0
    losses = []
    medoids_idx_old = None
    start_time = time.time()
    while (current_iter <= max_iter) and \
          ((current_iter < 1) or (not all(medoids_idx == medoids_idx_old))) and\
          ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):

        medoids_idx_old = medoids_idx
        memberships = _compute_memberships(data, medoids_idx, fuzzifier)
        top_memberships_mask = _compute_top_membership_subset(memberships, membership_subset_size)
        medoids_idx = _compute_medoids(data, memberships, fuzzifier, top_memberships_mask)

        loss = _compute_loss(data, medoids_idx, memberships, fuzzifier)
        losses.append(loss)

        current_iter += 1
        print_progression(iteration=current_iter, loss=loss, start_time=start_time)
    return memberships, data[medoids_idx, :], np.array(losses)


def _init_medoids(data, nb_examples, components, selection_method="random"):
    if selection_method == "random":
        return np.random.choice(range(nb_examples), size=components)
    elif selection_method == "most_dissimilar":
        return None
    elif selection_method == "most_dissimilar_randomized":
        return None
    else:
        return None


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

    # TODO: tester la rapidité sans le toarray
    top_memberships_mask = \
        csr_matrix(([1] * len(topk_idx.flatten()),
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
