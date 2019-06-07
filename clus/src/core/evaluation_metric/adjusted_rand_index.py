import numpy as np

from scipy.special import comb

from clus.src.utils.array import contingency_matrix
from clus.src.utils.decorator import remove_unexpected_arguments


@remove_unexpected_arguments
def adjusted_rand_index(affectations_ground_truth, affectations_prediction):
    """ Rand index adjusted for chance.
    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.
    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::
        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).
    ARI is a symmetric measure::
        adjusted_rand_index(a, b) == adjusted_rand_index(b, a)

    Parameters
    ----------
    affectations_ground_truth : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    affectations_prediction : array, shape = [n_samples]
        Cluster labels to evaluate
    Returns
    -------
    ari : float
       Similarity score between -1.0 and 1.0. Random labellings have an ARI
       close to 0.0. 1.0 stands for perfect match.
    Examples
    --------
    Perfectly matching labelings have a score of 1 even
      >>> adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Labellings that assign all classes members to the same clusters
    are complete be not always pure, hence penalized::
      >>> adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1])  # doctest: +ELLIPSIS
      0.57...
    ARI is symmetric, so labellings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::
      >>> adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2])  # doctest: +ELLIPSIS
      0.57...
    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::
      >>> adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
    Source
    ------
    https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/metrics/cluster/supervised.py#L137
    """
    n_samples = affectations_ground_truth.shape[0]
    n_classes = np.unique(affectations_ground_truth).shape[0]
    n_clusters = np.unique(affectations_prediction).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1) or \
       (n_classes == n_clusters == 0) or \
       (n_classes == n_clusters == n_samples):
        return 1.0

    # Compute the ARI using the contingency data
    contingency = contingency_matrix(affectations_ground_truth, affectations_prediction, sparse=True)
    sum_comb_c = sum(comb(n_c, 2, exact=True) for n_c in np.ravel(contingency.sum(axis=1)))
    sum_comb_k = sum(comb(n_k, 2, exact=True) for n_k in np.ravel(contingency.sum(axis=0)))
    sum_comb = sum(comb(n_ij, 2, exact=True) for n_ij in contingency.data)

    prod_comb = (sum_comb_c * sum_comb_k) / comb(n_samples, 2, exact=True)
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return (sum_comb - prod_comb) / (mean_comb - prod_comb)
