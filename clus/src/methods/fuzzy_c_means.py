from typing import Optional, Tuple

import numpy as np

from tqdm import tqdm

from clus.src.handle_empty_clusters import handle_empty_clusters
from clus.src.cluster_initialization import cluster_initialization
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, Elapsed:{elapsed}, ETA:{remaining}{postfix}"


@remove_unexpected_arguments
def fuzzy_c_means(data: np.ndarray, components: int = 10, eps: float = 1e-4, max_iter: int = 1000, fuzzifier: float = 2,
                  initialization_method: str = "random_choice", empty_clusters_method: str = "nothing",
                  centroids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Performs the fuzzy c-means clustering algorithm on a dataset.

    :param data: The dataset into which the clustering will be performed. The dataset must be 2D np.array with rows as
    examples and columns as features.
    :param components: The number of components (clusters) wanted.
    :param eps: Criterion used to define convergence. If the absolute differences between two consecutive losses is
    lower than `eps`, the clustering stop.
    :param max_iter: Criterion used to stop the clustering if the number of iterations exceeds `max_iter`.
    :param fuzzifier: Membership fuzzification coefficient.
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
    :param centroids: Initials centroids to use instead of randomly initialize them.
    :return: A tuple containing :
    * The memberships matrix.
    * The centroids matrix.
    * An array with all losses at each iteration.
    """
    assert len(data.shape) == 2, "The data must be a 2D array"
    assert data.shape[0] > 0, "The data must have at least one example"
    assert data.shape[1] > 0, "The data must have at least one feature"
    assert 1 <= components <= data.shape[0], "The number of components wanted must be between 1 and %s" % data.shape[0]
    assert 0 <= max_iter, "The number of max iterations must be positive"
    assert fuzzifier > 1, "The fuzzifier must be greater than 1"
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, initialization_method, need_idx=False)

    with tqdm(total=max_iter, bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
        memberships = None
        current_iter = 0
        losses = []
        while (current_iter <= max_iter) and \
              ((current_iter < 2) or (abs(losses[-2] - losses[-1] > eps))):
            memberships = _compute_memberships(data, centroids, fuzzifier)
            handle_empty_clusters(data, centroids, memberships, strategy=empty_clusters_method)

            centroids = _compute_centroids(data, memberships, fuzzifier)

            loss = _compute_loss(data, memberships, centroids, fuzzifier)
            losses.append(loss)

            # Update the progress bar
            current_iter += 1
            progress_bar.update()
            progress_bar.set_postfix({
                "Loss": "{0:.6f}".format(loss)
            })

    return memberships, centroids, np.array(losses)


def _compute_memberships(data, centroids, fuzzifier):
    # TODO: If an example is at the exact same coordinates than a centroid (euclidean distance == 0), set its membership
    #  to 1, and the memberships of others to 0. See [3]
    dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1) ** 2

    tmp = np.power(dist_data_centroids, -2 / (fuzzifier - 1), where=dist_data_centroids != 0)
    big_sum = tmp.sum(axis=0, keepdims=True)
    res = np.divide(tmp, big_sum, where=big_sum != 0).T
    res = np.fmax(res, 0.)  # Float manipulation sometimes cause a 0. to be set to -0.
    return res


def _compute_centroids(data, memberships, fuzzifier):
    fuzzified_memberships = memberships ** fuzzifier
    sum_memberships_by_centroid = np.sum(fuzzified_memberships, axis=0)
    return np.divide(np.dot(data.T, fuzzified_memberships), sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids, fuzzifier):
    dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1) ** 2
    return ((memberships ** fuzzifier) * dist_data_centroids.T).sum()


def __compute_memberships(data, centroids, fuzzifier):
    """ DEPRECATED: old method used to compute the memberships matrix.
    Much slower than the existing method.
    """
    u_ir = np.zeros(shape=(data.shape[0], centroids.shape[0]))
    for i in range(data.shape[0]):
        for r in range(centroids.shape[0]):
            d_ir = np.linalg.norm(data[i] - centroids[r], ord=2) ** 2
            if d_ir == 0:
                for s in range(centroids.shape[0]):
                    u_ir[i][s] = 0
                u_ir[i][r] = 1
                break
            big_sum = sum((d_ir / (np.linalg.norm(data[i] - centroids[s], ord=2) ** 2)) ** (2 / (fuzzifier - 1))
                          for s in range(centroids.shape[0]))
            u_ir[i][r] = 1 / big_sum
    return u_ir


def __compute_loss(data, memberships, centroids, fuzzifier):
    """ DEPRECATED: old method used to compute the loss.
    Much slower than the existing method.
    """
    res = 0
    for i in range(centroids.shape[0]):
        for j in range(data.shape[0]):
            membership_fuzzified = memberships[j][i] ** fuzzifier
            dist_data_centroid = np.linalg.norm(data[j] - centroids[i], ord=2) ** 2
            res += membership_fuzzified * dist_data_centroid
    return res


if __name__ == '__main__':
    pass
