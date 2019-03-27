""" Apply a clustering algorithm to a CSV dataset.

Some algorithms need a pairwise distance matrix as a dataset. If the dataset you provide is not a pairwise distance
matrix (eg: with not the same number of examples and dimensions), the software will compute it itself with a pairwise
euclidean distance.

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
import numpy as np

from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.methods.methods import get_clustering_function, use_distance_matrix, use_medoids
from clus.src.core.normalization import normalization as normalize
from clus.src.utils.random import set_manual_seed
from clus.src.core.visualisation import visualise_clustering_2d, visualise_clustering_3d
from clus.src.utils.path import compute_visualisation_saving_path

_MAX_TEXT_OUTPUT_WIDTH = 120


@click.command(help=__doc__, context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
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
              help="Character or REGEX used for separating data in the CSV data file.")
@click.option("--header", is_flag=True,
              help="Set this flag if your dataset contains a header, it will then be ignored by the clustering "
                   "algorithm. If you set this flag while not having a header, the first example of the dataset will "
                   "be ignored.")
# Clustering options
@click.option("--initialization-method", type=str, default="random_choice", show_default=True,
              help="Method used to initialize the clusters' center. The following methods are available :\b\n"
                   "- 'random_uniform' or 'uniform', samples values between the min and max across each dimension.\n"
                   "- 'random_gaussian' or 'gaussian', samples values from a gaussian with the same mean and std as "
                   "each data's dimension.\n"
                   "- 'random_choice' or 'choice', samples random examples from the data without replacement.\n"
                   "- 'central_dissimilar_medoids', samples the first medoid as the most central point of the "
                   "dataset, then sample all successive medoids as the most dissimilar to all medoids that have "
                   "already been picked.\n"
                   "- 'central_dissimilar_random_medoids', same as 'central_dissimilar_medoids', but the first medoid "
                   "is sampled randomly.")
@click.option("--empty-clusters-method", type=str, default="nothing", show_default=True,
              help="Method used to handle empty clusters. The following methods are available :\n"
                   "- 'nothing', do absolutely nothing and ignore empty clusters.\n"
                   "- 'random_example', assign a random example to all empty clusters.\n"
                   "- 'furthest_example_from_its_centroid', assign the furthest example from its centroid to each "
                   "empty cluster.")
@click.option("-c", "-k", "--components", type=int, default=5, show_default=True,
              help="Number of clustering components.")
@click.option("--eps", type=float, default=1e-6, show_default=True,
              help="Minimal threshold characterizing an algorithm's convergence.")
@click.option("--max-iter", type=int, default=1000, show_default=True,
              help="Maximal number of iteration to make before stopping an algorithm.")
@click.option("-m", "--fuzzifier", type=float, default=2, show_default=True,
              help="Fuzzification exponent applied to the membership degrees.")
@click.option("--pairwise-distance", type=str, default="euclidean", show_default=True,
              help="Metric used to compute the distance matrix when the clustering algorithm need it. Set to "
                   "\"precomputed\" if your data is already a distance matrix. All possible metrics are described at "
                   "the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("-p", "--membership-subset-size", type=int, default=None, show_default=True,
              help="Size of the highest membership subset examined during the medoids computation for LFCMdd.")
# Visualisation options
@click.option("--visualise", is_flag=True,
              help="Set this flag if you want to visualise the clustering result. If your data's dimension is more "
                   "than 2, a 2-components PCA is applied to the data before visualising.")
@click.option("--visualise-3d", is_flag=True,
              help="Set this flag if you want to visualise the clustering result in 3D. If your data's dimension is "
                   "more than 3, a 3-components PCA is applied to the data before visualising.")
@click.option("--save", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result. If your data's "
                   "dimension is more than 2, a 2-components PCA is applied to the data before visualising.")
@click.option("--save-3d", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result in 3D. If your data's "
                   "dimension is more than 3, a 3-components PCA is applied to the data before visualising.")
# Miscellaneous options
@click.option("--seed", type=int, default=None, show_default=True,
              help="Random seed to set.")
@click.option("--normalization", type=str, default=None, show_default=True,
              help="Normalize your data with any of the proposed methods below :\n"
                   "- 'rescaling', rescale the range of features into the [0, 1] range.\n"
                   "- 'mean' or 'mean_normalization', normalize the features to zero mean.\n"
                   "- 'standardization', normalize the features to zero mean and unit std.\n"
                   "- 'unit_length', scale the components by dividing each features by their p-norm.\n"
                   "- 'whitening_zca' or 'zca', maximizes the average cross-covariance between each dimension of the "
                   "whitened and original data, and uniquely produces a symmetric cross-covariance matrix.\n"
                   "- 'whitening_pca' or 'pca', maximally compresses all dimensions of the original data into each "
                   "dimension of the whitened data using the cross-covariance matrix as the compression metric.\n"
                   "- 'whitening_zca_cor' or 'zca_cor', maximizes the average cross-correlation between each dimension "
                   "of the whitened and original data, and uniquely produces a symmetric cross-correlation matrix.\n"
                   "- 'whitening_pca_cor' or 'pca_cor', maximally compresses all dimensions of the original data into "
                   "each dimension of the whitened data using the cross-correlation matrix as the compression metric.\n"
                   "- 'whitening_cholesky' or 'cholesky', Uniquely results in lower triangular positive diagonal "
                   "cross-covariance and cross-correlation matrices.")
@click.option("--quiet", is_flag=True,
              help="Set this flag if you want to completely silence all outputs to stdout.")
@click.option("--path-dir-dest", type=str, default="results", show_default=True,
              help="Path to the directory containing all saved results (logs, plots, ...). Will be created if it does "
                   "not already exists.")
def main(dataset, clustering_algorithm, delimiter, header, initialization_method,
         empty_clusters_method, components, eps, max_iter, fuzzifier, pairwise_distance,
         membership_subset_size, visualise, visualise_3d, save, save_3d, seed, normalization,
         quiet, path_dir_dest):
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

    if normalization is not None:
        data = data.astype(np.float64)
        normalize(data, strategy=normalization)

    # Some methods need the data to be a pairwise distance matrix
    # If it is not the case, default to the euclidean distance
    distance_matrix = None
    if use_distance_matrix(clustering_algorithm):
        if pairwise_distance == "precomputed":
            assert data.shape[0] != data.shape[1], ("Your precomputed distance matrix is not square (shape: {})."
                                                    "").format(data.shape)
            distance_matrix = data
        else:
            distance_matrix = DistanceMetric.get_metric(pairwise_distance).pairwise(data)

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
    if use_medoids(clustering_algorithm):
        clusters_center = data[clusters_center, :]

    if visualise:
        visualise_clustering_2d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist(),
                                saving_path=compute_visualisation_saving_path(dataset,
                                                                              clustering_algorithm,
                                                                              components,
                                                                              seed,
                                                                              dir_dest=path_dir_dest),
                                show=True,
                                save=save)

    if visualise_3d:
        visualise_clustering_3d(data=data,
                                clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist(),
                                saving_path=compute_visualisation_saving_path(dataset,
                                                                              clustering_algorithm,
                                                                              components,
                                                                              seed,
                                                                              dir_dest=path_dir_dest),
                                show=True,
                                save=save_3d)


if __name__ == '__main__':
    pass
