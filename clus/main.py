import ntpath
import os
import sys
import glob

import click

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, single as linkage_pairwise_single, fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn.neighbors.dist_metrics import DistanceMetric

from clus.src.core.data_loading import load_data
from clus.src.core.evaluation_metric import evaluate
from clus.src.core.methods.methods import get_clustering_function, use_distance_matrix, is_hard_clustering
from clus.src.core.normalization import normalization as normalize
from clus.src.utils.click import OptionInfiniteArgs
from clus.src.utils.common import str_to_number
from clus.src.utils.process import execute
from clus.src.utils.random import set_manual_seed
from clus.src.core.visualisation import visualise_clustering_2d, visualise_clustering_3d, plot_dendrogram
from clus.src.utils.path import compute_file_saving_path

_MAX_TEXT_OUTPUT_WIDTH = 120


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("datasets", type=str, nargs=-1)
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
@click.option("-m", "--fuzzifier", type=float, default=2, show_default=True,
              help="Fuzzification exponent applied to the membership degrees.")
@click.option("--pairwise-distance", type=str, default="euclidean", show_default=True,
              help="Metric used to compute the distance matrix when the clustering algorithm need it. Set to "
                   "\"precomputed\" if your data is already a distance matrix. All possible metrics are described at "
                   "the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("--weights", cls=OptionInfiniteArgs,
              help="Weights used for the \"weighted_euclidean\" pairwise distance. You need as much weights as "
                   "you have features in your data.")
@click.option("-p", "--membership-subset-size", type=int, default=None, show_default=True,
              help="Size of the highest membership subset examined during the medoids computation for LFCMdd.")
@click.option("--save-clus", is_flag=True,
              help="Set this flag if you want to save the clustering result. A .npz file will be created, containing "
                   "the memberships matrix 'memberships', the clusters' center matrix 'clusters_center' and the losses "
                   "across all iterations 'losses'.")
@click.option("--keep-memberships", is_flag=True,
              help="Set this flag if you want to keep the memberships matrix in your results. It has been removed by "
                   "default because it is usually not used, take a large amount of disk space and can be resumed by "
                   "the 'ambiguity' and 'entropy' scalars.")
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
@click.option("--url-scp", default=None, show_default=True, type=str,
              help="If given, any saved result will be sent to this ssh address by using the `scp` command. The file "
                   "destination will then be 'url_scp:path_dir_dest'. For it to works, you also need to set your "
                   "public key to the destination computer. You can easily do it with the `ssh-keygen` software.")
def clus(datasets, clustering_algorithm, file_type, delimiter, header, array_name, initialization_method,
         empty_clusters_method, components, eps, max_iter, fuzzifier, pairwise_distance, weights,
         membership_subset_size, save_clus, keep_memberships, visualise, visualise_3d, save_visu, save_visu_3d, seed,
         normalization, quiet, path_dir_dest, url_scp):
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

    for dataset in glob.glob(datasets):
        print("Starting clustering with the following parameters :", parameters)

        if seed is not None:
            set_manual_seed(seed)

        # Load the clustering algorithm
        clustering_function = get_clustering_function(clustering_algorithm)

        # Load data
        data = load_data(dataset, file_type=file_type, delimiter=delimiter, header=header, array_name=array_name)

        if normalization is not None:
            data = data.astype(np.float64)
            normalize(data, strategy=normalization)

        if weights is not None:
            # Sometimes weights are parse as a tuple, or as a string with space in them. Take both cases in
            # consideration
            if " " in weights[0]:
                weights = tuple(map(lambda s: str_to_number(s), weights[0].split(" ")))
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
                distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data * np.sqrt(weights))
            else:
                distance_matrix = DistanceMetric.get_metric(pairwise_distance).pairwise(data)

        # Perform the clustering method
        clustering_result = clustering_function(
            data=data,
            distance_matrix=distance_matrix,
            components=components,
            eps=eps,
            max_iter=max_iter,
            fuzzifier=fuzzifier,
            weights=weights,
            membership_subset_size=membership_subset_size,
            initialization_method=initialization_method,
            empty_clusters_method=empty_clusters_method,
        )
        if not keep_memberships:
            del clustering_result["memberships"]

        # Create destination directory if it does not already exists
        os.makedirs(path_dir_dest, exist_ok=True)

        if save_clus:
            file_path = compute_file_saving_path(dataset=dataset,
                                                 clustering_algorithm=clustering_algorithm,
                                                 components=components,
                                                 seed=seed,
                                                 distance=pairwise_distance,
                                                 weights=weights,
                                                 fuzzifier=None if is_hard_clustering(clustering_algorithm) else fuzzifier,
                                                 dir_dest=path_dir_dest,
                                                 extension="npz",
                                                 is_3d_visualisation=False)
            np.savez_compressed(file_path, **clustering_result)
            if url_scp is not None:
                execute("scp", file_path, url_scp + ":" + path_dir_dest)
                os.remove(file_path)

        if visualise or save_visu:
            file_path = compute_file_saving_path(dataset=dataset,
                                                 clustering_algorithm=clustering_algorithm,
                                                 components=components,
                                                 seed=seed,
                                                 distance=pairwise_distance,
                                                 weights=weights,
                                                 fuzzifier=None if is_hard_clustering(
                                                     clustering_algorithm) else fuzzifier,
                                                 dir_dest=path_dir_dest,
                                                 extension="png")
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
                execute("scp", file_path, url_scp + ":" + path_dir_dest)
                os.remove(file_path)

        if visualise_3d or save_visu_3d:
            file_path = compute_file_saving_path(dataset=dataset,
                                                 clustering_algorithm=clustering_algorithm,
                                                 components=components,
                                                 seed=seed,
                                                 distance=pairwise_distance,
                                                 weights=weights,
                                                 fuzzifier=None if is_hard_clustering(clustering_algorithm) else fuzzifier,
                                                 dir_dest=path_dir_dest,
                                                 extension="png",
                                                 is_3d_visualisation=True)
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
                execute("scp", file_path, url_scp + ":" + path_dir_dest)
                os.remove(file_path)


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("datasets", type=str, nargs=-1)
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
# Clustering option
@click.option("--distance-metric", type=str, default="euclidean", show_default=True,
              help="Metric used to compute distance between examples. If set to \"weighted_euclidean\", the --weights "
                   "parameter also needs to be set. All possible metrics are described at the following link :\n"
                   "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html")
@click.option("--weights", cls=OptionInfiniteArgs,
              help="Weights used for the \"weighted_euclidean\" distance. You need as much weights as you have "
                   "features in your data.")
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
def hclus(datasets, file_type, delimiter, header, array_name, distance_metric, weights, save_z, save_flat_clusters,
          flat_clusters_criterion, flat_clusters_value, visualise, save_dendrogram, depth_cut, seed, normalization,
          quiet, path_dir_dest):
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    for dataset in glob.glob(datasets):
        print("Starting hierarchical clustering with the following parameters :", parameters)

        if seed is not None:
            set_manual_seed(seed)

        # Load data
        dataset_name = os.path.splitext(ntpath.basename(dataset))[0]
        data = load_data(dataset, file_type=file_type, delimiter=delimiter, header=header, array_name=array_name)

        if normalization is not None:
            data = data.astype(np.float64)
            normalize(data, strategy=normalization)

        # Load distance
        distance_mtx = None
        distance_metric = distance_metric.lower()
        if distance_metric == "euclidean":
            pass
        elif distance_metric == "weighted_euclidean":
            assert weights is not None,\
                "You need to precise the --weights parameter for th 'weighted_euclidean' distance."

            # Sometimes weights are parse as a tuple, or as a string with space in them. Take both cases in
            # consideration
            if isinstance(weights, tuple):
                weights = tuple(map(lambda s: str_to_number(s), weights))
            else:
                weights = tuple(map(lambda s: str_to_number(s), weights.split(" ")))

            # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
            # weighted by the square root of the weights, see [5]
            assert len(weights) == data.shape[0], \
                "You need as much weights as you have features in your data. Expected %d, got %d" % \
                (data.shape[0], len(weights))
            data = data * np.sqrt(weights)
        else:
            # Apply a scipy pairwise distance (list of available methods here :
            # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.spatial.distance.pdist.html
            # Returns a condensed distance matrix.
            distance_mtx = pdist(data, distance_metric, w=weights)

        # Compute linkage
        if distance_mtx is not None:
            linkage_mtx = linkage_pairwise_single(distance_mtx)
        else:
            linkage_mtx = linkage(data)

        # Create destination directory if it does not already exists
        os.makedirs(path_dir_dest, exist_ok=True)

        if save_z:
            dir_file_linkage_mtx = os.path.join(path_dir_dest, "z_" + dataset_name)
            np.save(dir_file_linkage_mtx, linkage_mtx)

        if save_flat_clusters:
            flat_clusters = fcluster(linkage_mtx, criterion=flat_clusters_criterion, t=flat_clusters_value)
            dir_file_linkage_mtx = os.path.join(path_dir_dest, "f_" + dataset_name)
            np.save(dir_file_linkage_mtx, flat_clusters)

        if visualise or save_dendrogram:
            plot_dendrogram(linkage_mtx=linkage_mtx, depth_cut=depth_cut, dataset_name=dataset_name,
                            show=visualise, save=save_dendrogram)


@click.command(context_settings=dict(max_content_width=_MAX_TEXT_OUTPUT_WIDTH))
@click.argument("metric", type=click.Choice([
    "ari", "adjusted_rand_index"
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
# Miscellaneous options
@click.option("--seed", type=int, default=None, show_default=True,
              help="Random seed to set.")
@click.option("--quiet", is_flag=True,
              help="Set this flag if you want to completely silence all outputs to stdout.")
def eclus(metric, file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred, seed,
          quiet):
    parameters = locals()

    if quiet:
        sys.stdout = open(os.devnull, 'w')

    if seed is not None:
        set_manual_seed(seed)

    print("Starting clustering evaluation with the following parameters :", parameters)

    evaluate(metric, file_affectations_true, file_affectations_pred, name_affectations_true, name_affectations_pred)


if __name__ == '__main__':
    pass
