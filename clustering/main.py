""" Apply a clustering algorithm to a CSV dataset.

\b
The following clustering algorithms are supported :
* kmeans
* fuzzy_c_means (or fcm)
* possibilistic_c_means (or pcm)
* fuzzy_c_medoids (or fcmdd)
* hard_c_medoids (or hcmdd)
* linearized_fuzzy_c_medoids (or lfcmdd, l_fc_med)
* linearized_fuzzy_c_medoids_select (or l_fcmed_select)
* datastream_linearized_fuzzy_c_medoids_select (or ds_lfcmed_select)
"""

import ntpath
import os
import sys

import click

import pandas as pd

from sklearn.neighbors.dist_metrics import DistanceMetric

from clustering.src.methods.methods import get_clustering_function, use_distance_matrix
from clustering.src.utils import set_manual_seed, normalization_mean_std
from clustering.src.visualisation import visualise_clustering_2d, visualise_clustering_3d, visualise_clustering_loss


@click.command(help=__doc__)
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("clustering_algorithm", type=click.Choice([
    "kmeans",
    "fuzzy_c_means", "fcm",
    "possibilistic_c_means", "pcm",
    "fuzzy_c_medoids", "fcmdd",
    "hard_c_medoids", "hcmdd",
    "linearized_fuzzy_c_medoids", "lfcmdd", "l_fc_med",
    "linearized_fuzzy_c_medoids_select", "l_fcmed_select",
    "datastream_linearized_fuzzy_c_medoids_select", "ds_lfcmed_select",
]))
# CSV parsing options
@click.option("--delimiter", "--sep", type=str, default=",", show_default=True,
              help="Character or REGEX used for separating data in the CSV data file")
@click.option("--header", is_flag=True,
              help=("Set this flag if your dataset contains a header, it will then be ignored by the clustering algorit"
                    "hm. If you set this flag while not having a header, the first example of the dataset will be ignor"
                    "ed"))
# Clustering options
@click.option("--initialization-method", type=int, default=3, show_default=True,
              help=("Method used to initialize the clusters' center. The following method are available :\n"
                    "1 - random: Draw clusters from a uniform distribution between the min and the max of each attribu"
                    "te\n"
                    "2 - random_gaussian: Draw clusters from a gaussian distribution\n"
                    "3 - random_choice: Randomly choose clusters from the dataset\n"
                    "4 - central_dissimilar_medoids: TODO\n"
                    "5 - central_dissimilar_random_medoids: TODO\n"))
@click.option("--empty-clusters-method", type=int, default=1, show_default=True,
              help=("Method used to handle empty clusters. The following method are available :\n"
                    "1 - nothing: Do absolutely nothing\n"
                    "2 - random_example: Draw a random example and fully assign it to the cluster\n"
                    "3 - furthest_example_from_its_centroid: TODO\n"))
@click.option("-c", "-k", "--components", type=int, default=5, show_default=True,
              help="Number of clustering components")
@click.option("--eps", type=float, default=1e-4, show_default=True,
              help="Minimal threshold caracterizing an algorithm's convergence")
@click.option("--max-iter", type=int, default=1000, show_default=True,
              help="Maximal number of iteration to make before stopping an algorithm")
@click.option("-m", "--fuzzifier", type=float, default=2, show_default=True,
              help="Fuzzification exponent applied to the membership degrees")
@click.option("-p", "--membership-subset-size", type=int, default=None, show_default=True,
              help="Size of the highest membership subset examined during the medoids computation for LFCMdd.")
# Visualisation options
@click.option("--visualise", is_flag=True,
              help=("Set this flag if you want to visualise the clustering result. If your data's dimension is more tha"
                    "n 2, a 2-components PCA is applied to the data before visualising."))
@click.option("--visualise-3d", is_flag=True,
              help=("Set this flag if you want to visualise the clustering result in 3D. If your data's dimension is mo"
                    "re than 3, a 3-components PCA is applied to the data before visualising."))
@click.option("--save", is_flag=True,
              help=("Set this flag if you want to save the visualisation of the clustering result. If your data's dimen"
                    "sion is more than 2, a 2-components PCA is applied to the data before visualising."))
@click.option("--save-3d", is_flag=True,
              help=("Set this flag if you want to save the visualisation of the clustering result in 3D. If your data's"
                    " dimension is more than 3, a 3-components PCA is applied to the data before visualising."))
# Miscellaneous options
@click.option("--seed", type=int, default=None, show_default=True,
              help="Random seed to set")
@click.option("--normalize", is_flag=True,
              help="Set this flag if you want to normalize your data to zero mean and unit variance")
@click.option("--quiet", is_flag=True,
              help="Set this flag if you want to have absolutely no output during the execution")
def main(dataset, clustering_algorithm, delimiter, header, initialization_method,
         empty_clusters_method, components, eps, max_iter, fuzzifier,
         membership_subset_size, visualise, visualise_3d, save, save_3d, seed, normalize,
         quiet):
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    if seed is not None:
        set_manual_seed(seed)

    print("Starting clustering with the following parameters :", parameters)

    # Load the clustering algorithm
    clustering_function = get_clustering_function(clustering_algorithm)

    # Load data
    data = pd.read_csv(dataset, delimiter=delimiter, header=0 if header else None).values
    #TODO: TODELETE
    data = data / 100000
    print(data.min(), data.max())

    if normalize:
        # TODO: Which normalization ?
        data = normalization_mean_std(data)

    # Some methods need the data to be a pairwise distance matrix
    # If it is not the case, default to the euclidean distance
    distance_matrix = None
    if use_distance_matrix(clustering_algorithm):
        if data.shape[0] != data.shape[1]:
            print("The data need to be a pairwise distance matrix for the {} clustering "
                  "method.".format(clustering_algorithm), "Applying euclidean distance.")
            distance_matrix = DistanceMetric.get_metric(
                'euclidean').pairwise(data)
        else:
            distance_matrix = data

    # Perform the clustering method
    memberships, clusters_center, losses = clustering_function(
        data=data,
        distance_matrix=distance_matrix,
        components=components,
        eps=eps,
        max_iter=max_iter,
        fuzzifier=fuzzifier,
        membership_subset_size=membership_subset_size,
        initialization_method=initialization_method,
        empty_clusters_method=empty_clusters_method,
    )
    print("")

    if visualise:
        visualise_clustering_2d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist())

    if visualise_3d:
        visualise_clustering_3d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist())
    if save:
        visualise_clustering_2d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist(),
                                show=False,
                                saving_path=_compute_saving_path(dataset,
                                                                 clustering_algorithm,
                                                                 components,
                                                                 seed,
                                                                 dir_dest="results"))

    if save_3d:
        visualise_clustering_3d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist(),
                                show=True,
                                saving_path=_compute_saving_path(dataset,
                                                                 clustering_algorithm,
                                                                 components,
                                                                 seed,
                                                                 dir_dest="results"))


def _compute_saving_path(dataset, clustering_algorithm, components,
                         seed, dir_dest) -> str:
    os.makedirs(dir_dest, exist_ok=True)

    return os.path.join(dir_dest, "{}_{}_{}_{}.png".format(
        os.path.splitext(ntpath.basename(dataset))[0],
        clustering_algorithm,
        components,
        seed
    ))


if __name__ == '__main__':
    pass
