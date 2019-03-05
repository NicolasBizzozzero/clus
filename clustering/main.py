# TODO: Comment gérer les 0 (petit epsilon partout ou juste sur les 0) ou même supprimer les doublons
# MJ a dit de tester voir si j'obtiens les memes resultats (a epsilon pres) en ajoutant epsilon ou alors en mettant
# à 0 les valeurs De u_ij pour des exemples qui sont égaux
# Ne surtout pas supprimer les doublons. Un cluster avec plein d'exemples au même endroit aura plus de force qu'un
# cluster avec un seul exemple.
# TODO: Est-ce que je normalise mes données ?
# Ne pas faire de normalisation centrée-réduite sur un même attribut à cause des outliers
# On peut cependant faire une normalisation entre les attributs de manière à ce qu'ils soient dans le même
# intervalle de valeurs, et qu'un attribut ne soit pas plus fort qu'un autre.
# TODO: La matrice de dissimilarité, on la fournie ou je dois la calculer ? La dissimilarité est symétrique ?
# La dissimilarité n'est pas forcement symétrique, mais on peut suppose qu'elle l'est pour pouvoir simplifier des
# calculs.

import ntpath

import click

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors.dist_metrics import DistanceMetric

from clustering.src.methods.methods import get_clustering_function, ClusteringMethod, clusteringmethod_to_str, \
    str_to_clusteringmethod
from clustering.src.utils import set_manual_seed, normalization_mean_std


# TODO: Lister tous les algos disponibles et leurs acronymes
# TODO: Dans chaque option, lister pour quels acronymes elles sont disponibles (ou s'ils le sont pour tous sauf ...)
from clustering.src.vizualisation import vizualise_clustering_2d, vizualise_clustering_3d


@click.command()
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
@click.option("--delimiter", "--sep", type=str, default=",",
              help="Character or REGEX used for separating data in the CSV data file")
@click.option("--header", is_flag=True,
              help=("Set this flag if your dataset contains a header, it will then be ignored by the clustering algorit"
                    "hm. If you set this flag while not having a header, the first example of the dataset will be ignor"
                    "ed"))
# Clustering options
@click.option("-c", "-k", "--components", type=int, default=100,
              help="Number of clustering components")
@click.option("--eps", type=float, default=0.001,
              help="Minimal threshold caracterizing an algorithm's convergence")
@click.option("--max-iter", type=int, default=1000,
              help="Maximal number of iteration to make before stopping an algorithm")
@click.option("-m", "--fuzzifier", type=float, default=2,
              help="Fuzzification exponent applied to the membership degrees")
@click.option("-p", "--membership-subset-size", type=int, default=None,
              help="Size of the highest membership subset examined during the medoids computation for LFCMdd.")
# Miscellaneous options
@click.option("--seed", type=int, default=None,
              help="Random seed to set")
@click.option("--normalize", is_flag=True,
              help="Set this flag if you want to normalize your data to zero mean and unit variance")
@click.option("--vizualise", is_flag=True,
              help=("Set this flag if you want to vizualise the clustering re"
                    "sult. If your data's dimension is more than 2, a 2-compo"
                    "nents PCA is applied to the data before vizualising."))
@click.option("--vizualise-3d", is_flag=True,
              help=("Set this flag if you want to vizualise the clustering re"
                    "sult in 3D. If your data's dimension is more than 3, a 3"
                    "-components PCA is applied to the data before vizualisin"
                    "g."))
def main(dataset, clustering_algorithm, delimiter, header, components, eps,
         max_iter, fuzzifier, membership_subset_size, seed, normalize,
         vizualise, vizualise_3d):
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
    print("Starting clustering with the following parameters :", locals())

    if seed is not None:
        set_manual_seed(seed)

    # Load the clustering algorithm
    clustering_function = get_clustering_function(clustering_algorithm)

    # Load data
    data = pd.read_csv(dataset, delimiter=delimiter,
                       header=0 if header else None).values

    if normalize:
        # TODO: Which normalization ?
        data = normalization_mean_std(data)

    # Some methods need the data to be a pairwise distance matrix
    # If it is not the case, default to the euclidean distance
    if str_to_clusteringmethod(clustering_algorithm) in (
            ClusteringMethod.FUZZY_C_MEDOIDS,
            ClusteringMethod.LINEARIZED_FUZZY_C_MEDOIDS):
        if data.shape[0] != data.shape[1]:
            print("The data need to be a pairwise distance matrix for the {} clustering "
                  "method.".format(clustering_algorithm), "Applying euclidean distance.")
            data = DistanceMetric.get_metric('euclidean').pairwise(data)

    # Perform the clustering method
    memberships, clusters_center, losses = clustering_function(
        data,
        components=components,
        eps=eps,
        max_iter=max_iter,
        fuzzifier=fuzzifier,
        membership_subset_size=membership_subset_size
    )
    print("")

    if vizualise:
        vizualise_clustering_2d(data=data, clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist())

    if vizualise_3d:
        vizualise_clustering_3d(data=data, clusters_center=clusters_center,
                                clustering_method=clustering_algorithm,
                                dataset_name=ntpath.basename(dataset),
                                header=None if not header else pd.read_csv(dataset, delimiter=delimiter,
                                                                           header=0).columns.tolist())


if __name__ == '__main__':
    pass
