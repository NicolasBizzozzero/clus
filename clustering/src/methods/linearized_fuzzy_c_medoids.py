import sys

import numpy as np

from clustering.src.utils import remove_unexpected_arguments


@remove_unexpected_arguments
def linearized_fuzzy_c_medoids(data, components, fuzzifier, membership_subset_size, eps, max_iter):
    assert (len(data.shape) == 2) and data.shape[0] == data.shape[1]  # The data matrix must be a square matrix

    # Initialisation
    nb_examples, dim = data.shape
    memberships = np.zeros(shape=(nb_examples, components), dtype=np.float64)
    medoids_idx = _init_medoids(data, nb_examples, components, selection_method="random")

    current_iter = 0
    losses = []
    medoids_idx_old = None
    while (current_iter <= max_iter) and \
            ((current_iter < 1) or (not all(medoids_idx == medoids_idx_old))) and \
            ((current_iter < 2) or not (abs(losses[-1] - losses[-2]) <= eps)):

        medoids_idx_old = medoids_idx
        memberships = _compute_memberships(data, medoids_idx, fuzzifier)
        top_membership_subset = _compute_top_membership_subset(memberships, membership_subset_size)
        exit(0)
        medoids_idx = _compute_medoids(data, memberships, fuzzifier, top_membership_subset)
        print(medoids_idx)
        exit(0)

        loss = _compute_loss(data, medoids_idx, memberships, fuzzifier)
        losses.append(loss)
        current_iter += 1

        sys.stdout.write('\r')
        sys.stdout.write("Iteration {}, Loss : {}".format(
            current_iter, loss
        ))
        sys.stdout.flush()
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


def _compute_loss(data, medoids_idx, memberships, fuzzifier):
    return ((memberships ** fuzzifier) * data[:, medoids_idx]).sum(axis=(1, 0))


def _compute_memberships(data, medoids_idx, fuzzifier):
    r = data[:, medoids_idx]

    # If two examples are of equals distance, the computation will make divisions by zero. We add this
    # small coefficient to not divide by zero while keeping our distances as correct as possible
    r += 1e-7

    tmp = (1 / r) ** (1 / (fuzzifier - 1))
    memberships = tmp / tmp.sum(axis=1, keepdims=True)

    for index_medoid, medoid in enumerate(medoids_idx):
        memberships[medoid, :] = 0.
        memberships[medoid, index_medoid] = 1.

    return memberships


def _compute_medoids(data, memberships, fuzzifier, top_membership_subset):
    # TODO: Comprendre les dimensions en entrée et en retour
    # TODO: COmprendre ce que je fais à chaque calcul et ce que je veux
    # TODO:
    C = 5
    N = 5000

    for i in range(C):
        for i_xk in top_membership_subset[:, i]:
            print((memberships[:, i] ** fuzzifier) * data[i_xk, :].sum())
        exit(0)



    r = data[..., np.newaxis]

    print(r)
    print(r.shape)
    exit(0)
    u = (memberships.take(top_membership_subset) ** fuzzifier)
    print(r.shape)
    print(u.shape)
    print((r * u).shape)
    print((r * u).sum(axis=1))

    exit(0)
    return (r * u).sum(axis=1).argmin(axis=0)


def _compute_top_membership_subset(memberships, membership_subset_size):
    topk_idx = np.argpartition(memberships, -membership_subset_size, axis=0)[-membership_subset_size:]
    top_memberships_subset = memberships[topk_idx, np.arange(memberships.shape[1])]  # Adrien
    from scipy import sparse
    mask = sparse.coo_matrix((np.ones(len(topk_idx[0])), topk_idx), shape=top_memberships_subset.shape, dtype=bool)
    print(mask.shape)
    print(mask)
    return top_memberships_subset

    # return memberships.argsort(axis=0)[-membership_subset_size:][::-1]


if __name__ == '__main__':
    pass
