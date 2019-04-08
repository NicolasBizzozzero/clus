import numpy as np
from tqdm import tqdm

from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.core.handle_empty_clusters import handle_empty_clusters
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"


@remove_unexpected_arguments
def kmeans(data, components=10, eps=1e-4, max_iter=1000, weights=None,
           initialization_method="random_choice", empty_clusters_method="nothing",
           centroids=None):
    """ Performs the k-means clustering algorithm on a dataset.

    :param data: The dataset into which the clustering will be performed. The dataset must be 2D np.array with rows as
    examples and columns as features.
    :param components: The number of components (clusters) wanted.
    :param eps: Criterion used to define convergence. If the absolute differences between two consecutive losses is
    lower than `eps`, the clustering stop.
    :param max_iter: Criterion used to stop the clustering if the number of iterations exceeds `max_iter`.
    :param weights: Weighting of each features during clustering. Must be an Iterable of weights with the same size as
    the number of features.
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
    assert (weights is None) or (len(weights) == data.shape[1]),\
        "The number of weights given must be the same as the number of features. Expected size : %s, given size : %s" %\
        (data.shape[1], len(weights))
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, strategy=initialization_method, need_idx=False)

    if weights is not None:
        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
        # weighted by the square root of the weights, see [5]
        data = data * np.sqrt(weights)

    with tqdm(total=max_iter, bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
        best_memberships = None
        best_centroids = None
        best_loss = np.inf

        memberships = None
        losses = []
        current_iter = 0
        while (current_iter < max_iter) and \
              ((current_iter < 2) or (abs(losses[-2] - losses[-1]) > eps)):
            memberships = _optim_memberships(data, centroids)
            handle_empty_clusters(data, centroids, memberships, strategy=empty_clusters_method)

            centroids = _optim_centroids(data, memberships)

            loss = _compute_loss(data, memberships, centroids)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_memberships = memberships
                best_centroids = centroids

            # Update the progress bar
            current_iter += 1
            progress_bar.update()
            progress_bar.set_postfix({
                "Loss": "{0:.6f}".format(loss),
                "best_loss": "{0:.6f}".format(best_loss)
            })

    return best_memberships, best_centroids, np.array(losses)


def _optim_memberships(data, centroids):
    """ Compute the memberships matrix minimizing the distance across all data and the centroids.

    Source :
    * https://codereview.stackexchange.com/questions/61598/k-mean-with-numpy
    """
    # Compute euclidean distance between data and centroids
    # dist_data_centroids = np.array([np.linalg.norm(data - c, ord=2, axis=1) for c in centroids]).T
    # dist_data_centroids = np.linalg.norm(data - centroids[:, np.newaxis], ord=2, axis=-1).T
    dist_data_centroids = np.linalg.norm(np.expand_dims(data, 2) - np.expand_dims(centroids.T, 0), axis=1)

    # Set all binary affectations
    mask_closest_centroid = (np.arange(data.shape[0]), dist_data_centroids.argmin(axis=1))
    affectations = np.zeros(shape=dist_data_centroids.shape, dtype=np.int32)
    affectations[mask_closest_centroid] = 1

    return affectations


def _optim_centroids(data, memberships):
    """ Compute the centroids minimizing the distance between all data examples and their respective centroids. """
    # We compute the division only with non-empty clusters. Indeed, a cluster may be
    # empty in some rare cases. See [2]
    sum_memberships_by_centroid = np.sum(memberships, axis=0)
    return np.divide(np.dot(data.T, memberships), sum_memberships_by_centroid, where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids):
    """ Compute the loss of the clustering algorithm.
    This method do not have any purpose in the clustering algorithm. It is only invoked for result analysis.
    """
    dist_data_centroids = data - centroids[:, np.newaxis]
    return (memberships * np.power(np.linalg.norm(dist_data_centroids, axis=-1, ord=2), 2).T).sum()


if __name__ == '__main__':
    pass
