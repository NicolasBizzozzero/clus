import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from clus.src.core.analysis import ambiguity, partition_coefficient, partition_entropy
from clus.src.core.cluster_initialization import cluster_initialization
from clus.src.core.handle_empty_clusters import handle_empty_clusters
from clus.src.core.visualisation import visualise_clustering_3d, visualise_clustering_2d
from clus.src.utils.decorator import remove_unexpected_arguments

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} max_iter, elapsed:{elapsed}, ETA:{remaining}{postfix}"


@remove_unexpected_arguments
def fuzzy_c_means(data, components=10, eps=1e-4, max_iter=1000, fuzzifier=2, weights=None,
                  initialization_method="random_choice", empty_clusters_method="nothing",
                  centroids=None):
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

    with tqdm(total=max_iter, bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
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
            # TODO: Remove this, only for testing
            if current_iter == 0:
                visualise_clustering_3d(data[:, :-1], centroids[:, :-1], memberships.argmax(axis=1),
                                        clustering_method="fcm", dataset_name="rhocut", header=["x", "y", "v"],
                                        show=False, save=True, saving_path="test2/rhocut_" + str(current_iter) + ".png")

            centroids = _compute_centroids(data, memberships, fuzzifier)

            loss = _compute_loss(data, memberships, centroids, fuzzifier)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_memberships = memberships
                best_centroids = centroids

            """
            # TODO:
            print("\n________________\nmemberships :")
            print(memberships)
            print(__compute_memberships(data, centroids, fuzzifier))
            print("\n")

            print(centroids)
            print("\n")

            print(loss)
            print(__compute_loss(data, memberships, centroids, fuzzifier))
            print("\n")
            """
            # real_loss = __compute_loss(data, memberships, centroids, fuzzifier)
            real_real_loss = ___compute_loss(data, memberships, centroids, fuzzifier)
            print("\n\n", loss, real_loss, real_real_loss, "\n\n")

            # Update the progress bar
            current_iter += 1
            progress_bar.update()
            progress_bar.set_postfix({
                "loss": "{0:.6f}".format(loss),
                "best_loss": "{0:.6f}".format(best_loss)
            })
            # TODO: Remove this, only for testing
            visualise_clustering_3d(data[:, :-1], centroids[:, :-1], memberships.argmax(axis=1),
                                    clustering_method="fcm", dataset_name="rhocut", header=["x", "y", "v"],
                                    show=False, save=True, saving_path="test2/rhocut_" + str(current_iter) + ".png")

    visualise_clustering_3d(data[:, :-1], best_centroids[:, :-1], best_memberships.argmax(axis=1),
                            clustering_method="fcm", dataset_name="rhocut", header=["x", "y", "v"],
                            show=True, save=False, saving_path="test2/rhocut_" + str(current_iter) + ".png")

    return {
        "memberships": best_memberships,
        "clusters_center": best_centroids,
        "losses": np.array(losses),
        "affectations": best_memberships.argmax(axis=1),
        "ambiguity": ambiguity(best_memberships),
        "partition_coefficient": partition_coefficient(best_memberships),
        "partition_entropy": partition_entropy(best_memberships),
    }


def _compute_memberships(data, centroids, fuzzifier):
    # TODO: If an example is at the exact same coordinates than a centroid (euclidean distance == 0), set its membership
    #  to 1, and the memberships of others to 0. See [3]
    dist_data_centroids = cdist(data, centroids, metric="euclidean")

    tmp = np.power(dist_data_centroids, -2 / (fuzzifier - 1), where=dist_data_centroids != 0)
    big_sum = tmp.sum(axis=1, keepdims=True)
    res = np.divide(tmp, big_sum, where=big_sum != 0)
    res = np.fmax(res, 0.)  # Float manipulation sometimes cause a 0. to be set to -0.
    return res


"""
memberships, on devrait trouver ça :
[[2.14329233e-06 2.16417805e-06 1.75012183e-06 ... 1.00174269e-06
  9.24806978e-07 9.20941396e-07]
 [2.16349689e-06 2.18517620e-06 1.76360931e-06 ... 1.00637118e-06
  9.28596963e-07 9.24587933e-07]
 [2.18389428e-06 2.20638976e-06 1.77717758e-06 ... 1.01099706e-06
  9.32377983e-07 9.28223106e-07]
 ...
 [5.92730794e-07 5.56126034e-07 7.23210122e-07 ... 1.12070477e-06
  1.23302104e-06 1.35028562e-06]
 [6.00138287e-07 5.62944776e-07 7.33135364e-07 ... 1.14013592e-06
  1.25524884e-06 1.37478733e-06]
 [6.12203604e-07 5.73522007e-07 7.49499639e-07 ... 1.16957141e-06
  1.28906069e-06 1.41727407e-06]]
"""


def _compute_centroids(data, memberships, fuzzifier):
    fuzzified_memberships = memberships ** fuzzifier
    sum_memberships_by_centroid = np.sum(fuzzified_memberships, axis=0)
    return np.divide(np.dot(data.T, fuzzified_memberships), sum_memberships_by_centroid,
                     where=sum_memberships_by_centroid != 0).T


def _compute_loss(data, memberships, centroids, fuzzifier):
    dist_data_centroids = cdist(data, centroids, metric="euclidean")
    return ((memberships ** fuzzifier) * dist_data_centroids).sum()


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
            big_sum = sum((d_ir / (np.linalg.norm(data[i] - centroids[s], ord=2))) ** (2 / (fuzzifier - 1))
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
            dist_data_centroid = np.linalg.norm(data[j] - centroids[i], ord=2)
            res += membership_fuzzified * dist_data_centroid
    return res


def ___compute_loss(x, u, w, m):
    """ DEPRECATED: old method used to compute the loss.
    Much slower than the existing method.
    """
    res = 0
    c = w.shape[0]
    n = x.shape[0]
    print("C:", c)
    print("N:", n)
    exit(0)

    for r in range(c):
        for i in range(n):
            res += (u[i][r] ** m) * np.sqrt(((x[i] - w[r]) ** 2).sum())
    return res


if __name__ == '__main__':
    pass
