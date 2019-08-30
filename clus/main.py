import ntpath
import os
import sys

import click
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, single as linkage_pairwise_single, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.data_loading import load_data
from clus.src.core.evaluation_metric.evaluation_metric import evaluate, ALIASES_ADJUSTED_RAND_INDEX, \
    ALIASES_ADJUSTED_MUTUAL_INFO, ALIASES_COMPLETENESS, ALIASES_CONTINGENCY_MATRIX, ALIASES_FOWLKES_MALLOWS_INDEX, \
    ALIASES_HOMOGENEITY, ALIASES_MUTUAL_INFO, ALIASES_NORMALIZED_MUTUAL_INFO, ALIASES_V_MEASURE, ALIASES_N10, \
    ALIASES_N01, ALIASES_N00, ALIASES_N11, ALIASES_PURITY, ALIASES_INVERSE_PURITY
from clus.src.core.methods.methods import ALIASES_KMEANS, ALIASES_OPTICS, ALIASES_DBSCAN, \
    ALIASES_DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT, ALIASES_FUZZY_C_MEANS, ALIASES_FUZZY_C_MEDOIDS, \
    ALIASES_HARD_C_MEDOIDS, ALIASES_LINEARIZED_FUZZY_C_MEDOIDS, ALIASES_LINEARIZED_FUZZY_C_MEDOIDS_SELECT, \
    ALIASES_MINI_BATCH_KMEANS, ALIASES_POSSIBILISTIC_C_MEANS, ALIASES_FUZZY_C_MEANS_SELECT
from clus.src.core.methods.methods import get_clustering_function, use_distance_matrix, is_hard_clustering
from clus.src.core.normalization import normalization as normalize
from clus.src.core.saving_path import compute_file_saving_path_clus
from clus.src.core.saving_path import compute_file_saving_path_dclus
from clus.src.core.visualisation import visualise_clustering_2d, visualise_clustering_3d, plot_dendrogram
from clus.src.utils.click import OptionInfiniteArgs
from clus.src.utils.common import str_to_number
from clus.src.utils.decorator import wrap_max_memory_consumption
from clus.src.utils.process import execute
from clus.src.utils.random import set_manual_seed

_MAX_TEXT_OUTPUT_WIDTH = 120


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("datasets", type=str, nargs=-1, required=True)
@click.argument("clustering_algorithm", type=click.Choice([
    *ALIASES_KMEANS,
    *ALIASES_MINI_BATCH_KMEANS,
    *ALIASES_FUZZY_C_MEANS,
    *ALIASES_POSSIBILISTIC_C_MEANS,
    *ALIASES_FUZZY_C_MEDOIDS,
    *ALIASES_HARD_C_MEDOIDS,
    *ALIASES_FUZZY_C_MEANS_SELECT,
    *ALIASES_LINEARIZED_FUZZY_C_MEDOIDS,
    *ALIASES_LINEARIZED_FUZZY_C_MEDOIDS_SELECT,
    *ALIASES_DATASTREAM_LINEARIZED_FUZZY_C_MEDOIDS_SELECT
]))
# Data loading options
@click.option("--file-type", type=str, default="guess", show_default=True,
              help="The type of file from which the data is read. Possible values are :\n"
                   "- 'guess', automatically guess the filetype from the file extension.\n"
                   "- 'csv', load the data with a call to the pandas.read_csv method.\n"
                   "- 'npy', load the only array contained in a numpy file.\n"
                   "- 'npz', load one of the array contained in a numpy file. The `--array-name` option needs to be "
                   "set.")
@click.option("--delimiter", "--sep", type=str, default=",", show_default=True,
              help="Character or REGEX used for separating data in the CSV data file.")
@click.option("--header", is_flag=True,
              help="Set this flag if your dataset contains a header, it will then be ignored by the clustering "
                   "algorithm. If you set this flag while not having a header, the first example of the dataset will "
                   "be ignored.")
@click.option("--array-name", type=str, default=None, show_default=True,
              help="Used to load a specific array from a numpy npz file.")
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
@click.option("-m", "--fuzzifier", type=float, default=2.0, show_default=True,
              help="Fuzzification exponent applied to the membership degrees.")
@click.option("--pairwise-distance", type=str, default="euclidean", show_default=True,
              help="Metric used to compute the distance matrix when the clustering algorithm need it. Set to "
                   "\"precomputed\" if your data is already a distance matrix. All possible metrics are described at "
                   "the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("--weights", cls=OptionInfiniteArgs,
              help="Weights used for the \"weighted_euclidean\" pairwise distance. You need as much weights as "
                   "you have features in your data.")
@click.option("--max-epochs", type=int, default=128, show_default=True,
              help="Number of time to repeat a clustering algorithm.")
@click.option("-b", "--batch-size", type=int, default=1000,
              help="Size of a batch for minibatch compatible algorithms.")
@click.option("-p", "--membership-subset-size", type=int, default=None, show_default=True,
              help="Size of the highest membership subset examined during the medoids computation for LFCMdd.")
@click.option("--min-centroid-size", type=int, default=None, show_default=True,
              help="Criterion used to remove clusters with a too small cardinal after each epoch.")
@click.option("--max-centroid-diameter", type=float, default=np.inf, show_default=True,
              help="Criterion used to remove clusters with a too big diameter after each epoch.")
@click.option("--linkage-method", type=str, default="single", show_default=True,
              help="The linkage algorithm to use for hierarchical clustering. Available methods are listed here : "
                   "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html")
@click.option("--save-clus", is_flag=True,
              help="Set this flag if you want to save the clustering result. A .npz file will be created, containing "
                   "the memberships matrix 'memberships', the clusters' center matrix 'clusters_center' and the "
                   "losses across all iterations 'losses'.")
@click.option("--keep-memberships", is_flag=True,
              help="Set this flag if you want to keep the memberships matrix in your results. It has been removed by "
                   "default because it is usually not used, take a large amount of disk space and can be resumed by "
                   "the 'ambiguity' and 'entropy' scalars.")
@click.option("--flat-clusters-criterion", default="maxclust", show_default=True, type=str,
              help="The criterion to use in forming flat clusters. Possible values can be found at :\n"
                   "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html")
@click.option("--flat-clusters-value", default=0.0, show_default=True, type=float,
              help="For criteria 'inconsistent', 'distance' or 'monocrit', this is the threshold to apply when forming "
                   "flat clusters. For 'maxclust' or 'maxclust_monocrit' criteria, this would be max number of "
                   "clusters requested.")
# Visualisation options
@click.option("--visualise", is_flag=True,
              help="Set this flag if you want to visualise the clustering result. If your data's dimension is more "
                   "than 2, a 2-components t-SNE is applied to the data before visualising.")
@click.option("--visualise-3d", is_flag=True,
              help="Set this flag if you want to visualise the clustering result in 3D. If your data's dimension is "
                   "more than 3, a 3-components t-SNE is applied to the data before visualising.")
@click.option("--save-visu", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result. If your data's "
                   "dimension is more than 2, a 2-components t-SNE is applied to the data before visualising.")
@click.option("--save-visu-3d", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result in 3D. If your data's "
                   "dimension is more than 3, a 3-components t-SNE is applied to the data before visualising.")
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
@click.option("--disable-progress-bar", is_flag=True,
              help="Set this flag if you want to completely disable the progress bar.")
@click.option("--path-dir-dest", default="results", show_default=True, type=str,
              help="Path to the directory containing all saved results (logs, plots, ...). Will be created if it does "
                   "not already exists.")
@click.option("--format-filename-dest-results", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{components}_{fuzzifier}_{seed}_{distance}",
              help="Format of the destination filename for the clustering results. Variables need to be enclosed in "
                   "brackets. Possible variables are : {dataset}, {clustering_algorithm}, {components}, {fuzzifier}, "
                   "{seed}, {distance}.")
@click.option("--format-filename-dest-visu", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{components}_{fuzzifier}_{seed}_{distance}",
              help="Format of the destination filename for the visualisation picture. Variables need to be enclosed in "
                   "brackets. Possible variables are : {dataset}, {clustering_algorithm}, {components}, {fuzzifier}, "
                   "{seed}, {distance}.")
@click.option("--format-filename-dest-visu-3d", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{components}_{fuzzifier}_{seed}_{distance}_3d",
              help="Format of the destination filename for the 3D-visualisation picture. Variables need to be enclosed "
                   "in brackets. Possible variables are : {dataset}, {clustering_algorithm}, {components}, "
                   "{fuzzifier}, {seed}, {distance}.")
@click.option("--zero-fill-components", default=3, show_default=True, type=int,
              help="The desired length of the 'number of components' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-seed", default=3, show_default=True, type=int,
              help="The desired length of the 'seed' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-weights", default=3, show_default=True, type=int,
              help="The desired length of the 'weights' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-fuzzifier", default=3, show_default=True, type=int,
              help="The desired length of the 'fuzzifier' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--url-scp", default=None, show_default=True, type=str,
              help="If given, any saved result will be sent to this ssh address by using the `scp` command. The file "
                   "destination will then be 'url_scp:path_dir_dest'. For it to works, you also need to set your "
                   "public key to the destination computer. You can easily do it with the `ssh-keygen` software.")
def clus(datasets, clustering_algorithm, file_type, delimiter, header, array_name, initialization_method,
         empty_clusters_method, components, eps, max_iter, fuzzifier, pairwise_distance, weights, max_epochs, batch_size,
         membership_subset_size, min_centroid_size, max_centroid_diameter, linkage_method,
         save_clus, keep_memberships, flat_clusters_criterion, flat_clusters_value, visualise, visualise_3d, save_visu,
         save_visu_3d, seed, normalization, quiet, disable_progress_bar, path_dir_dest, format_filename_dest_results,
         format_filename_dest_visu, format_filename_dest_visu_3d, zero_fill_components, zero_fill_seed,
         zero_fill_weights, zero_fill_fuzzifier, url_scp):
    """ Apply a clustering algorithm to a CSV dataset.

    Some algorithms need a pairwise distance matrix as a dataset. If the dataset you provide is not a pairwise distance
    matrix (eg: with not the same number of examples and dimensions), the software will compute it itself with a
    pairwise euclidean distance.

    \b
    The following clustering algorithms are supported :
    * kmeans
    * fuzzy_c_means (or fcm)
    * fuzzy_c_medoids (or fcmdd)
    * hard_c_medoids (or hcmdd)
    * linearized_fuzzy_c_medoids (or lfcmdd, l_fc_med)
    """
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    for dataset in datasets:
        print("Starting clustering with the following parameters :", parameters)

        if seed is not None:
            set_manual_seed(seed)

        # Load the clustering algorithm
        clustering_function = get_clustering_function(clustering_algorithm)

        # Load data
        data = load_data(dataset, file_type=file_type,
                         delimiter=delimiter, header=header, array_name=array_name)

        if normalization is not None:
            data = data.astype(np.float64)
            normalize(data, strategy=normalization)

        if weights is not None:
            # Sometimes weights are parse as a tuple, or as a string with space in them. Take both cases in
            # consideration
            if " " in weights[0]:
                weights = tuple(
                    map(lambda s: str_to_number(s), weights[0].split(" ")))
            else:
                weights = tuple(map(lambda s: str_to_number(s), weights))

        # Some methods need the data to be a pairwise distance matrix
        # If it is not the case, default to the euclidean distance
        distance_matrix = None
        if use_distance_matrix(clustering_algorithm):
            if pairwise_distance == "precomputed":
                assert data.shape[0] != data.shape[1], ("Your precomputed distance matrix is not square (shape: {})."
                                                        "").format(data.shape)
                distance_matrix = data
            elif pairwise_distance == "weighted_euclidean":
                distance_matrix = DistanceMetric.get_metric(
                    "euclidean").pairwise(data * np.sqrt(weights))
            else:
                distance_matrix = DistanceMetric.get_metric(
                    pairwise_distance).pairwise(data)

        # Perform the clustering method
        clustering_result = clustering_function(
            data=data,
            distance_matrix=distance_matrix,
            components=components,
            eps=eps,
            max_iter=max_iter,
            fuzzifier=fuzzifier,
            weights=weights,
            batch_size=batch_size,
            max_epochs=max_epochs,
            min_centroid_size=min_centroid_size,
            max_centroid_diameter=max_centroid_diameter,
            linkage_method=linkage_method,
            membership_subset_size=membership_subset_size,
            initialization_method=initialization_method,
            empty_clusters_method=empty_clusters_method,
            progress_bar=not disable_progress_bar,
            flat_clusters_criterion=flat_clusters_criterion,
            flat_clusters_value=flat_clusters_value
        )
        if (not keep_memberships) and ("memberships" in clustering_result):
            del clustering_result["memberships"]

        # Create destination directory if it does not already exists
        os.makedirs(path_dir_dest, exist_ok=True)

        if save_clus:
            file_path = compute_file_saving_path_clus(format_filename=format_filename_dest_results,
                                                      dataset=dataset,
                                                      clustering_algorithm=clustering_algorithm,
                                                      components=components,
                                                      fuzzifier=None if is_hard_clustering(
                                                          clustering_algorithm) else fuzzifier,
                                                      seed=seed,
                                                      distance=pairwise_distance,
                                                      weights=weights,
                                                      dir_dest=path_dir_dest,
                                                      extension="npz",
                                                      zero_fill_components=zero_fill_components,
                                                      zero_fill_fuzzifier=zero_fill_fuzzifier,
                                                      zero_fill_seed=zero_fill_seed,
                                                      zero_fill_weights=zero_fill_weights)

            np.savez_compressed(file_path, **clustering_result)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)

        if visualise or save_visu:
            file_path = compute_file_saving_path_clus(format_filename=format_filename_dest_visu,
                                                      dataset=dataset,
                                                      clustering_algorithm=clustering_algorithm,
                                                      components=components,
                                                      fuzzifier=None if is_hard_clustering(
                                                          clustering_algorithm) else fuzzifier,
                                                      seed=seed,
                                                      distance=pairwise_distance,
                                                      weights=weights,
                                                      dir_dest=path_dir_dest,
                                                      extension="png",
                                                      zero_fill_components=zero_fill_components,
                                                      zero_fill_fuzzifier=zero_fill_fuzzifier,
                                                      zero_fill_seed=zero_fill_seed,
                                                      zero_fill_weights=zero_fill_weights)
            visualise_clustering_2d(data=data,
                                    clusters_center=clustering_result["clusters_center"],
                                    affectations=clustering_result["affectations"],
                                    clustering_method=clustering_algorithm,
                                    dataset_name=ntpath.basename(dataset),
                                    header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                               header=0).columns.tolist(),
                                    saving_path=file_path,
                                    show=visualise,
                                    save=save_visu)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)

        if visualise_3d or save_visu_3d:
            file_path = compute_file_saving_path_clus(format_filename=format_filename_dest_visu_3d,
                                                      dataset=dataset,
                                                      clustering_algorithm=clustering_algorithm,
                                                      components=components,
                                                      fuzzifier=None if is_hard_clustering(
                                                          clustering_algorithm) else fuzzifier,
                                                      seed=seed,
                                                      distance=pairwise_distance,
                                                      weights=weights,
                                                      dir_dest=path_dir_dest,
                                                      extension="png",
                                                      zero_fill_components=zero_fill_components,
                                                      zero_fill_fuzzifier=zero_fill_fuzzifier,
                                                      zero_fill_seed=zero_fill_seed,
                                                      zero_fill_weights=zero_fill_weights)
            visualise_clustering_3d(data=data,
                                    clusters_center=clustering_result["clusters_center"],
                                    affectations=clustering_result["affectations"],
                                    clustering_method=clustering_algorithm,
                                    dataset_name=ntpath.basename(dataset),
                                    header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                               header=0).columns.tolist(),
                                    saving_path=file_path,
                                    show=visualise_3d,
                                    save=save_visu_3d)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("datasets", type=str, nargs=-1, required=True)
# Data loading options
@click.option("--file-type", type=str, default="guess", show_default=True,
              help="The type of file from which the data is read. Possible values are :\n"
                   "- 'guess', automatically guess the filetype from the file extension.\n"
                   "- 'csv', load the data with a call to the pandas.read_csv method.\n"
                   "- 'npy', load the only array contained in a numpy file.\n"
                   "- 'npz', load one of the array contained in a numpy file. The `--array-name` option needs to be "
                   "set.")
@click.option("--delimiter", "--sep", type=str, default=",", show_default=True,
              help="Character or REGEX used for separating data in the CSV data file.")
@click.option("--header", is_flag=True,
              help="Set this flag if your CSV dataset contains a header, it will then be ignored by the clustering "
                   "algorithm. If you set this flag while not having a header, the first example of the dataset will "
                   "be ignored.")
@click.option("--array-name", type=str, default=None, show_default=True,
              help="Used to load a specific array from a numpy npz file.")
@click.option("--is-linkage-mtx", is_flag=True,
              help="Set this flag if your dataset is already a linkage matrix.")
# Clustering option
@click.option("--distance-metric", type=str, default="euclidean", show_default=True,
              help="Metric used to compute distance between examples. If set to \"weighted_euclidean\", the --weights "
                   "parameter also needs to be set. All possible metrics are described at the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("--weights", cls=OptionInfiniteArgs,
              help="Weights used for the \"weighted_euclidean\" distance. You need as much weights as you have "
                   "features in your data.")
@click.option("--linkage-method", type=str, default="single", show_default=True,
              help="The linkage algorithm to use for hierarchical clustering. Available methods are listed here : "
                   "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html")
@click.option("--save-z", is_flag=True,
              help="Set this flag if you want to save the Z matrix containing the hierarchical clustering result. A "
                   "(n - 1) by 4 matrix Z is then saved. At the i-th iteration, clusters with indices Z[i, 0] and "
                   "Z[i, 1] are combined to form cluster n+1. A cluster with an index less than n corresponds to one "
                   "of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by "
                   "Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly "
                   "formed cluster.")
@click.option("--save-flat-clusters", is_flag=True,
              help="Set this flag if you want to save the clusters in a flattened form (as an affectation array). If "
                   "this flag has been set, you also need to user the `--flat-clusters-criterion` and "
                   "`--flat-clusters-value` parameters.")
@click.option("--flat-clusters-criterion", default="distance", show_default=True, type=str,
              help="The criterion to use in forming flat clusters. Possible values can be found at :\n"
                   "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html")
@click.option("--flat-clusters-value", default=0.0, show_default=True, type=float,
              help="For criteria 'inconsistent', 'distance' or 'monocrit', this is the threshold to apply when forming "
                   "flat clusters. For 'maxclust' or 'maxclust_monocrit' criteria, this would be max number of "
                   "clusters requested.")
# Visualisation options
@click.option("--visualise", is_flag=True,
              help="Set this flag if you want to visualise the resulting dendrogram.")
@click.option("--save-dendrogram", is_flag=True,
              help="Set this flag if you want to save the resulting dendrogram.")
@click.option("--depth-cut", type=int, default=8, show_default=True,
              help="Max size of the plotted dendrogram.")
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
@click.option("--path-dir-dest", default="results", show_default=True, type=str,
              help="Path to the directory containing all saved results (logs, plots, ...). Will be created if it does "
                   "not already exists.")
@click.option("--format-filename-dest-z", default="z_{dataset_name}_{linkage_method}", show_default=True, type=str,
              help="Format of the destination filename for the z matrix. Variables need to be enclosed in brackets. "
                   "Possible variables are : {dataset_name}, {linkage_method}.")
@click.option("--format-filename-dest-f", default="f_{dataset_name}_{linkage_method}", show_default=True, type=str,
              help="Format of the destination filename for the flat clustering vector. Variables need to be enclosed "
                   "in brackets. Possible variables are : {dataset_name}, {linkage_method}.")
def hclus(datasets, file_type, delimiter, header, array_name, is_linkage_mtx, distance_metric, weights, linkage_method,
          save_z, save_flat_clusters, flat_clusters_criterion, flat_clusters_value, visualise, save_dendrogram,
          depth_cut, seed, normalization, quiet, path_dir_dest, format_filename_dest_z, format_filename_dest_f):
    parameters = locals()
    del parameters["datasets"]

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    for dataset in datasets:
        parameters["dataset"] = dataset
        print(
            "Starting hierarchical clustering with the following parameters :", parameters)

        if seed is not None:
            set_manual_seed(seed)

        # Load data
        dataset_name = os.path.splitext(ntpath.basename(dataset))[0]
        data = load_data(dataset, file_type=file_type,
                         delimiter=delimiter, header=header, array_name=array_name)

        if is_linkage_mtx:
            linkage_mtx = data
        else:
            if normalization is not None:
                data = data.astype(np.float64)
                normalize(data, strategy=normalization)
            url_scp = "bizzozzero@gate.lip6.fr"
            flat_clusters_criterion = "maxclus"
            clus_results = hierarchical_clustering(data, distance_metric, weights, linkage_method,
                                                   flat_clusters_criterion, flat_clusters_value, dataset, url_scp,
                                                   path_dir_dest)

            file_path = ntpath.basename(dataset) + "_" + "hc_" + linkage_method
            visualise_clustering_2d(data=data,
                                    clusters_center=None,
                                    affectations=clus_results["affectations"],
                                    clustering_method="hc_" + linkage_method,
                                    dataset_name=ntpath.basename(dataset),
                                    header=None,
                                    saving_path=file_path + ".png",
                                    show=False,
                                    save=True)
            if url_scp is not None:
                retcode = execute("scp", file_path + ".png", url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path + ".png")

            np.savez_compressed(file_path + ".npz", **clus_results)
            if url_scp is not None:
                retcode = execute("scp", file_path + ".npz", url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path + ".npz")

        # TODO: remove this, temporarely
        exit(0)

        # Create destination directory if it does not already exists
        os.makedirs(path_dir_dest, exist_ok=True)

        if save_z:
            file_name = format_filename_dest_z.format(
                dataset_name=dataset_name,
                linkage_method=linkage_method
            )
            dir_file_linkage_mtx = os.path.join(path_dir_dest, file_name)
            np.save(dir_file_linkage_mtx, linkage_mtx)

        if save_flat_clusters:
            flat_clusters = fcluster(linkage_mtx, criterion=flat_clusters_criterion, t=flat_clusters_value)
            file_name = format_filename_dest_f.format(
                dataset_name=dataset_name,
                linkage_method=linkage_method
            )
            dir_file_linkage_mtx = os.path.join(path_dir_dest, file_name)
            np.save(dir_file_linkage_mtx, flat_clusters)

        if visualise or save_dendrogram:
            plot_dendrogram(linkage_mtx=linkage_mtx, depth_cut=depth_cut, dataset_name=dataset_name,
                            linkage_method=linkage_method, show=visualise, save=save_dendrogram)


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("datasets", type=str, nargs=-1, required=True)
@click.argument("clustering_algorithm", type=click.Choice([
    *ALIASES_DBSCAN,
    *ALIASES_OPTICS
]))
# Data loading options
@click.option("--file-type", type=str, default="guess", show_default=True,
              help="The type of file from which the data is read. Possible values are :\n"
                   "- 'guess', automatically guess the filetype from the file extension.\n"
                   "- 'csv', load the data with a call to the pandas.read_csv method.\n"
                   "- 'npy', load the only array contained in a numpy file.\n"
                   "- 'npz', load one of the array contained in a numpy file. The `--array-name` option needs to be "
                   "set.")
@click.option("--delimiter", "--sep", type=str, default=",", show_default=True,
              help="Character or REGEX used for separating data in the CSV data file.")
@click.option("--header", is_flag=True,
              help="Set this flag if your dataset contains a header, it will then be ignored by the clustering "
                   "algorithm. If you set this flag while not having a header, the first example of the dataset will "
                   "be ignored.")
@click.option("--array-name", type=str, default=None, show_default=True,
              help="Used to load a specific array from a numpy npz file.")
# Clustering options
@click.option("--eps", type=float, default=1e-6, show_default=True,
              help="The maximum distance between two samples for them to be considered as in the same neighborhood.")
@click.option("--min-samples", type=int, default=3, show_default=True,
              help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core "
                   "point. This includes the point itself.")
@click.option("--max-eps", type=float, default=np.inf, show_default=True,
              help="The maximum distance between two samples for them to be considered as in the same neighborhood. "
                   "Default value of np.inf will identify clusters across all scales; reducing max_eps will result in "
                   "shorter run times.")
@click.option("--pairwise-distance", type=str, default="euclidean", show_default=True,
              help="Metric used to compute the distance matrix when the clustering algorithm need it. Set to "
                   "\"precomputed\" if your data is already a distance matrix. All possible metrics are described at "
                   "the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("--weights", cls=OptionInfiniteArgs,
              help="Weights used for the \"weighted_euclidean\" pairwise distance. You need as much weights as "
                   "you have features in your data.")
@click.option("--save-clus", is_flag=True,
              help="Set this flag if you want to save the clustering result. A .npz file will be created, containing TODO")
# Visualisation options
@click.option("--visualise", is_flag=True,
              help="Set this flag if you want to visualise the clustering result. If your data's dimension is more "
                   "than 2, a 2-components t-SNE is applied to the data before visualising.")
@click.option("--visualise-3d", is_flag=True,
              help="Set this flag if you want to visualise the clustering result in 3D. If your data's dimension is "
                   "more than 3, a 3-components t-SNE is applied to the data before visualising.")
@click.option("--save-visu", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result. If your data's "
                   "dimension is more than 2, a 2-components t-SNE is applied to the data before visualising.")
@click.option("--save-visu-3d", is_flag=True,
              help="Set this flag if you want to save the visualisation of the clustering result in 3D. If your data's "
                   "dimension is more than 3, a 3-components t-SNE is applied to the data before visualising.")
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
@click.option("--path-dir-dest", default="results", show_default=True, type=str,
              help="Path to the directory containing all saved results (logs, plots, ...). Will be created if it does "
                   "not already exists.")
@click.option("--format-filename-dest-results", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{min_samples}_{eps}_{seed}_{distance}",
              help="Format of the destination filename for the clustering results. Variables need to be enclosed in "
                   "brackets. Possible variables are : {dataset}, {clustering_algorithm}, {min_samples}, {eps}, "
                   "{seed}, {distance}.")
@click.option("--format-filename-dest-visu", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{min_samples}_{eps}_{seed}_{distance}",
              help="Format of the destination filename for the clustering results. Variables need to be enclosed in "
                   "brackets. Possible variables are : {dataset}, {clustering_algorithm}, {min_samples}, {eps}, "
                   "{seed}, {distance}.")
@click.option("--format-filename-dest-visu-3d", show_default=True, type=str,
              default="{dataset}_{clustering_algorithm}_{min_samples}_{eps}_{seed}_{distance}_3d",
              help="Format of the destination filename for the clustering results. Variables need to be enclosed in "
                   "brackets. Possible variables are : {dataset}, {clustering_algorithm}, {min_samples}, {eps}, "
                   "{seed}, {distance}.")
@click.option("--zero-fill-eps", default=3, show_default=True, type=int,
              help="The desired length of the 'eps' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-min-samples", default=3, show_default=True, type=int,
              help="The desired length of the 'min_samples' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-seed", default=3, show_default=True, type=int,
              help="The desired length of the 'seed' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--zero-fill-weights", default=3, show_default=True, type=int,
              help="The desired length of the 'weights' parameter displayed on any output filename. If "
                   "this parameter as a smaller length that the one wanted, zeroes are padded to the left of the "
                   "string.")
@click.option("--url-scp", default=None, show_default=True, type=str,
              help="If given, any saved result will be sent to this ssh address by using the `scp` command. The file "
                   "destination will then be 'url_scp:path_dir_dest'. For it to works, you also need to set your "
                   "public key to the destination computer. You can easily do it with the `ssh-keygen` software.")
def dclus(datasets, clustering_algorithm, file_type, delimiter, header, array_name, eps, min_samples, max_eps,
          pairwise_distance, weights, save_clus, visualise, visualise_3d, save_visu, save_visu_3d, seed, normalization,
          quiet, path_dir_dest, format_filename_dest_results, format_filename_dest_visu, format_filename_dest_visu_3d,
          zero_fill_eps, zero_fill_min_samples, zero_fill_seed, zero_fill_weights,
          url_scp):
    """ Apply a density-based clustering algorithm to a CSV dataset. """
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    for dataset in datasets:
        print("Starting clustering with the following parameters :", parameters)

        if seed is not None:
            set_manual_seed(seed)

        # Load the clustering algorithm
        clustering_function = get_clustering_function(clustering_algorithm)

        # Load data
        data = load_data(dataset, file_type=file_type,
                         delimiter=delimiter, header=header, array_name=array_name)

        if normalization is not None:
            data = data.astype(np.float64)
            normalize(data, strategy=normalization)

        if weights is not None:
            # Sometimes weights are parse as a tuple, or as a string with space in them. Take both cases in
            # consideration
            if " " in weights[0]:
                weights = tuple(
                    map(lambda s: str_to_number(s), weights[0].split(" ")))
            else:
                weights = tuple(map(lambda s: str_to_number(s), weights))

        # Perform the clustering method
        clustering_result = clustering_function(
            data=data,
            eps=eps,
            min_samples=min_samples,
            max_eps=max_eps,
            weights=weights
        )

        # Create destination directory if it does not already exists
        os.makedirs(path_dir_dest, exist_ok=True)

        if save_clus:
            file_path = compute_file_saving_path_dclus(format_filename=format_filename_dest_results,
                                                       dataset=dataset,
                                                       clustering_algorithm=clustering_algorithm,
                                                       min_samples=min_samples,
                                                       eps=max_eps if clustering_algorithm in ALIASES_OPTICS else eps,
                                                       seed=seed,
                                                       distance=pairwise_distance,
                                                       weights=weights,
                                                       extension="npz",
                                                       dir_dest=path_dir_dest,
                                                       zero_fill_min_samples=zero_fill_min_samples,
                                                       zero_fill_eps=zero_fill_eps,
                                                       zero_fill_seed=zero_fill_seed,
                                                       zero_fill_weights=zero_fill_weights)

            np.savez_compressed(file_path, **clustering_result)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)

        if visualise or save_visu:
            file_path = compute_file_saving_path_dclus(format_filename=format_filename_dest_visu,
                                                       dataset=dataset,
                                                       clustering_algorithm=clustering_algorithm,
                                                       min_samples=min_samples,
                                                       eps=max_eps if clustering_algorithm in ALIASES_OPTICS else eps,
                                                       seed=seed,
                                                       distance=pairwise_distance,
                                                       weights=weights,
                                                       extension="png",
                                                       dir_dest=path_dir_dest,
                                                       zero_fill_min_samples=zero_fill_min_samples,
                                                       zero_fill_eps=zero_fill_eps,
                                                       zero_fill_seed=zero_fill_seed,
                                                       zero_fill_weights=zero_fill_weights)
            visualise_clustering_2d(data=data,
                                    clusters_center=None,
                                    affectations=clustering_result["affectations"],
                                    clustering_method=clustering_algorithm,
                                    dataset_name=ntpath.basename(dataset),
                                    header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                               header=0).columns.tolist(),
                                    saving_path=file_path,
                                    show=visualise,
                                    save=save_visu)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)

        if visualise_3d or save_visu_3d:
            file_path = compute_file_saving_path_dclus(format_filename=format_filename_dest_visu_3d,
                                                       dataset=dataset,
                                                       clustering_algorithm=clustering_algorithm,
                                                       min_samples=min_samples,
                                                       eps=max_eps if clustering_algorithm in ALIASES_OPTICS else eps,
                                                       seed=seed,
                                                       distance=pairwise_distance,
                                                       weights=weights,
                                                       extension="png",
                                                       dir_dest=path_dir_dest,
                                                       zero_fill_min_samples=zero_fill_min_samples,
                                                       zero_fill_eps=zero_fill_eps,
                                                       zero_fill_seed=zero_fill_seed,
                                                       zero_fill_weights=zero_fill_weights)
            visualise_clustering_3d(data=data,
                                    clusters_center=None,
                                    affectations=clustering_result["affectations"],
                                    clustering_method=clustering_algorithm,
                                    dataset_name=ntpath.basename(dataset),
                                    header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                               header=0).columns.tolist(),
                                    saving_path=file_path,
                                    show=visualise_3d,
                                    save=save_visu_3d)
            if url_scp is not None:
                retcode = execute("scp", file_path, url_scp if ":" in url_scp else (
                                  url_scp + ":" + path_dir_dest))
                if retcode == 0:
                    os.remove(file_path)


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("metric", type=click.Choice([
    *ALIASES_ADJUSTED_RAND_INDEX,
    *ALIASES_ADJUSTED_MUTUAL_INFO,
    *ALIASES_COMPLETENESS,
    *ALIASES_CONTINGENCY_MATRIX,
    *ALIASES_FOWLKES_MALLOWS_INDEX,
    *ALIASES_HOMOGENEITY,
    *ALIASES_MUTUAL_INFO,
    *ALIASES_NORMALIZED_MUTUAL_INFO,
    *ALIASES_PURITY,
    *ALIASES_INVERSE_PURITY,
    *ALIASES_V_MEASURE,
    *ALIASES_N11,
    *ALIASES_N10,
    *ALIASES_N01,
    *ALIASES_N00
]))
# Data loading options
@click.option("--file-affectations-true", type=click.Path(exists=True), default=None,
              help="npz file used to load the true affectations.")
@click.option("--file-affectations-pred", type=click.Path(exists=True), default=None,
              help="npz file used to load the predicted affectations.")
@click.option("--name-affectations-true", type=str, default=None,
              help="Array name of the true affectations in the npz file.")
@click.option("--name-affectations-pred", type=str, default=None,
              help="Array name of the predicted affectations in the npz file.")
# Evaluation options
@click.option("--average-method", type=str, default="arithmetic", show_default=True,
              help="How to compute the normalizer in the denominator.  The following methods are available :\n"
                   "- 'min'\n"
                   "- 'geometric'\n"
                   "- 'arithmetic'\n"
                   "- 'max'\n")
@click.option("--eps", type=float, default=None,
              help="If a float, that value is added to all values in the contingency matrix. This helps to stop NaN "
                   "propagation. If None, nothing is adjusted.")
@click.option("--sparse", type=bool, default=False, show_default=True,
              help="If True, return a sparse CSR matrix. If eps is not None, and sparse is True, will throw"
                   "ValueError.")
@click.option("--beta", type=float, default=1.0, show_default=True,
              help="Ratio of weight attributed to homogeneity vs completeness. If beta is greater than 1, completeness "
                   "is weighted more strongly in the calculation. If beta is less than 1, homogeneity is weighted more "
                   "strongly.")
# Miscellaneous options
@click.option("--seed", type=int, default=None, show_default=True,
              help="Random seed to set.")
@click.option("--quiet", is_flag=True,
              help="Set this flag if you want to completely silence all outputs to stdout.")
def eclus(metric, file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred,
          average_method, eps, sparse, beta, seed, quiet):
    """ Evaluate a supervised clustering performance between a ground truth clustering and a prediction (or compare two
    clustering).
    """
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    if seed is not None:
        set_manual_seed(seed)

    print("Starting clustering evaluation with the following parameters :", parameters)

    # Load affectations
    affectations_true = np.load(file_affectations_true)[name_affectations_true]
    affectations_pred = np.load(file_affectations_pred)[name_affectations_pred]

    evaluate(metric=metric, affectations_true=affectations_true, affectations_pred=affectations_pred,
             average_method=average_method, eps=eps, sparse=sparse, beta=beta)


@wrap_max_memory_consumption
def hierarchical_clustering(data, distance_metric, weights, linkage_method, flat_clusters_criterion,
                            flat_clusters_value, dataset, url_scp, path_dir_dest):
    import time

    t0 = time.time()
    # Load distance
    distance_mtx = None
    distance_metric = distance_metric.lower()
    if distance_metric == "euclidean":
        pass
    elif distance_metric == "weighted_euclidean":
        assert weights is not None, \
            "You need to precise the --weights parameter for the 'weighted_euclidean' distance."

        # Sometimes weights are parse as a tuple, or as a string with space in them. Take both cases in
        # consideration
        if isinstance(weights, tuple):
            weights = tuple(map(lambda s: str_to_number(s), weights))
        else:
            weights = tuple(
                map(lambda s: str_to_number(s), weights.split(" ")))

        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into
        # data weighted by the square root of the weights, see [5]
        assert len(weights) == data.shape[1], \
            "You need as much weights as you have features in your data. Expected %d, got %d" % \
            (data.shape[1], len(weights))
        data = data * np.sqrt(weights)
    else:
        # Apply a scipy pairwise distance (list of available methods here :
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.distance.pdist.html
        # Returns a condensed distance matrix.
        distance_mtx = pdist(data, distance_metric, w=weights)

    # Compute linkage
    if distance_mtx is not None:
        linkage_mtx = linkage(distance_mtx, method=linkage_method)
    else:
        linkage_mtx = linkage(data, method=linkage_method)

    affectations = fcluster(linkage_mtx, criterion=flat_clusters_criterion, t=flat_clusters_value)
    t1 = time.time()
    clustering_result = {
        # Clustering results
        "affectations": affectations,
        "extended_time": t1 - t0,

        # Evaluation : Affectations
        "silhouette": silhouette_score(data, affectations),
        "variance_ratio": calinski_harabasz_score(data, affectations),
        "davies_bouldin": davies_bouldin_score(data, affectations)
    }
    return clustering_result

if __name__ == '__main__':
    pass
