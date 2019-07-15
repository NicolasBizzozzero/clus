from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist


def ambiguity(memberships):
    """ Compute the ambiguity of a memberships matrix.
    The ambiguity of a memberships matrix is defined as the vector containing for each sample the differences of the two
    highest memberships he has.
    """
    partition = -np.partition(-memberships, [0, 1], axis=1)
    top1 = partition[:, 0]
    top2 = partition[:, 1]
    return top1 - top2


def partition_coefficient(memberships):
    """ Compute the partition coefficient of a memberships matrix.

    The partition coefficient is defined in [6]. The value $F_c$ it returns is
    contained between $$\frac{1}{c} \leq F_c \leq 0$$
    """
    return (np.power(memberships, 2) / memberships.shape[0]).sum()


def partition_entropy(memberships):
    """ Compute the partition entropy of a memberships matrix.

    The partition entropy is defined in [6]. The value $H_c$ it returns is
    contained between $$0 \leq H_c \leq log_a(c)$$
    """
    return -(memberships * np.log(memberships, where=memberships != 0) / memberships.shape[0]).sum()


def entropy():
    pass


def cluster_diameter(cluster):
    """ Computer the diameter of a cluster, defined as the maximal pairwise distance between all its points. """
    pairwise_distances = pdist(cluster, metric="euclidean")
    if pairwise_distances.size == 0:
        return 0
    return max(pairwise_distances)


def clusters_diameter(data, affectations, clusters_id):
    diameters = np.empty_like(clusters_id, dtype=np.float32)
    for i, id_cluster in enumerate(clusters_id):
        cluster = data[affectations == id_cluster]
        diameters[i] = cluster_diameter(cluster)
    return diameters


def clusters_cardinal(affectations, clusters_id):
    counter = Counter(affectations)

    cardinals = np.empty_like(clusters_id, dtype=np.int32)
    for i, id_cluster in enumerate(clusters_id):
        cardinals[i] = counter[id_cluster]
    return cardinals


if __name__ == "__main__":
    pass
