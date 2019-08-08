import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

from clus.src.core.analysis import ambiguity, partition_coefficient, partition_entropy, clusters_diameter
from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.core.handle_empty_clusters import handle_empty_clusters
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"


@remove_unexpected_arguments
def fuzzy_c_means(data, components=10, eps=1e-4, max_iter=1000, fuzzifier=2, weights=None,
                  initialization_method="random_choice", empty_clusters_method="nothing",
                  centroids=None, progress_bar=True):
    """ Performs the fuzzy c-means clustering algorithm on a dataset.

    :param data: The dataset into which the clustering will be performed. The dataset must be 2D np.array with rows as
    examples and columns as features.
    :param components: The number of components (clusters) wanted.
    :param eps: Criterion used to define convergence. If the absolute differences between two consecutive losses is
    lower than `eps`, the clustering stop.
    :param max_iter: Criterion used to stop the clustering if the number of iterations exceeds `max_iter`.
    :param fuzzifier: Membership fuzzification coefficient.
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
    :param progress_bar: If `False`, disable the progress bar.
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
    assert (weights is None) or (len(weights) == data.shape[1]),\
        "The number of weights given must be the same as the number of features. Expected size : %s, given size : %s" %\
        (data.shape[1], len(weights))
    assert (centroids is None) or (centroids.shape == (components, data.shape[1])), \
        "The given centroids do not have a correct shape. Expected shape : {}, given shape : {}".format(
            (components, data.shape[1]), centroids.shape
        )

    if weights is not None:
        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
        # weighted by the square root of the weights, see [5]
        data = data * np.sqrt(weights)

    # Initialisation
    if centroids is None:
        centroids = cluster_initialization(data, components, initialization_method, need_idx=False)

    with tqdm(total=max_iter, bar_format=_FORMAT_PROGRESS_BAR, disable=not progress_bar) as progress_bar:
        best_memberships = None
        best_centroids = None
        best_loss = np.inf

        memberships = None
        current_iter = 0
        losses = []
        while (current_iter < max_iter) and \
              ((current_iter < 2) or (abs(losses[-2] - losses[-1]) > eps)):
            memberships = _compute_memberships(data, centroids, fuzzifier)
            handle_empty_clusters(data, centroids, memberships, strategy=empty_clusters_method)
            centroids = _compute_centroids(data, memberships, fuzzifier)

            loss = _compute_loss(data, memberships, centroids, fuzzifier)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_memberships = memberships
                best_centroids = centroids

            # Update the progress bar
            current_iter += 1
            progress_bar.update()
            progress_bar.set_postfix({
                "loss": "{0:.6f}".format(loss),
                "best_loss": "{0:.6f}".format(best_loss)
            })

    affectations = best_memberships.argmax(axis=1)
    clusters_id, clusters_cardinal = np.unique(affectations, return_counts=True)
    return {
        # Clustering results
        "memberships": best_memberships,
        "affectations": affectations,
        "clusters_center": best_centroids,
        "clusters_id": clusters_id,
        "losses": np.array(losses),
        "extended_time": progress_bar.last_print_t - progress_bar.start_t,

        # Evaluation : Memberships matrix
        "ambiguity": ambiguity(best_memberships),
        "partition_coefficient": partition_coefficient(best_memberships),
        "partition_entropy": partition_entropy(best_memberships),

        # Evaluation : Clusters center
        "clusters_diameter": clusters_diameter(data, affectations, clusters_id),
        "clusters_cardinal": clusters_cardinal,

        # Evaluation : Affectations
        "silhouette_samples": silhouette_samples(data, affectations),
        "silhouette": silhouette_score(data, affectations),
        "variance_ratio": calinski_harabasz_score(data, affectations),
        "davies_bouldin": davies_bouldin_score(data, affectations)
    }


def _compute_memberships(data, centroids, fuzzifier):
    np.seterr(all='warn')
    import warnings
    warnings.filterwarnings('error')

    dist_data_centroids = cdist(data, centroids, metric="euclidean")
    tmp = np.power(dist_data_centroids, -2 / (fuzzifier - 1), where=~np.isclose(dist_data_centroids, 0))
    big_sum = tmp.sum(axis=1, keepdims=True)
    res = np.divide(tmp, big_sum, where=~np.isclose(big_sum, 0))
    """
    try:
        res = np.divide(tmp, big_sum, where=~np.isclose(big_sum, 0))
    except RuntimeWarning as w:
        print("\nWarning   : " + str(w))
        print("Data      :", data.shape)
        print("Centroids :", centroids.shape)
        print("Tmp       :", tmp.shape)
        print("Big_sum   :", big_sum.shape)

        print(data)
        print(centroids)
        print(dist_data_centroids)
        print(tmp)
        print(big_sum)
        exit(0)
    """
    # If an example is at the exact same coordinates than a centroid (euclidean distance == 0), set its membership to
    # 1, and the memberships of others to 0. See [3]
    # This is done by computing a mask of zeros elements' index of the `dist_data_centroids` matrix, then by performing
    # the operation cited above afterward.
    # These operations do nothing if `idx_rows_with_zero` is empty.
    idx_rows_with_zero = np.where(np.isclose(dist_data_centroids, 0))
    res[idx_rows_with_zero[0]] = 0
    res[idx_rows_with_zero] = 1

    res = np.fmax(res, 0.)  # Float manipulation sometimes cause a 0. to be set to -0.
    return res


def _compute_centroids(data, memberships, fuzzifier):
    fuzzified_memberships = memberships ** fuzzifier
    sum_memberships_by_centroid = np.sum(fuzzified_memberships, axis=0)
    return np.divide(np.dot(data.T, fuzzified_memberships), sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids, fuzzifier):
    dist_data_centroids = cdist(data, centroids, metric="euclidean") ** 2
    return ((memberships ** fuzzifier) * dist_data_centroids).sum()


def __compute_memberships(x, w, m):
    """ DEPRECATED: old method used to compute the memberships matrix.
    Much slower than the existing method.
    """
    u_ir = np.zeros(shape=(x.shape[0], w.shape[0]))
    for i in range(x.shape[0]):
        for r in range(w.shape[0]):
            d_ir = np.sqrt(((x[i] - w[r]) ** 2).sum())
            if d_ir == 0:
                for s in range(w.shape[0]):
                    u_ir[i][s] = 0
                u_ir[i][r] = 1
                break

            big_sum = 0
            for s in range(w.shape[0]):
                d_is = np.sqrt(((x[i] - w[s]) ** 2).sum())
                if d_is == 0:
                    # The point is at the same position of the centroids, set it's distance to 0
                    continue
                big_sum += (d_ir / np.sqrt(((x[i] - w[s]) ** 2).sum())) ** (2 / (m - 1))
            u_ir[i][r] = 1 / big_sum
    return u_ir


def __compute_centroids(x, u, m):
    """ DEPRECATED: old method used to compute the centroids.
    Much slower than the existing method.
    """
    w = np.zeros(shape=(u.shape[1], x.shape[1]))
    for r in range(w.shape[0]):
        # compute big top sum
        big_top_sum = np.zeros(shape=(1, x.shape[1]))
        for i in range(x.shape[0]):
            big_top_sum += (u[i][r] ** m) * x[i]

        # compute big bottom sum
        big_bot_sum = np.zeros(shape=(1, x.shape[1]))
        for i in range(x.shape[0]):
            big_bot_sum += u[i][r] ** m
        w[r] = big_top_sum / big_bot_sum
    return w


def __compute_loss(x, u, w, m):
    """ DEPRECATED: old method used to compute the loss.
    Much slower than the existing method.
    """
    res = 0
    c = w.shape[0]
    n = x.shape[0]

    for r in range(c):
        for i in range(n):
            res += (u[i][r] ** m) * (np.sqrt(((x[i] - w[r]) ** 2).sum()) ** 2)
    return res


if __name__ == '__main__':
    pass
