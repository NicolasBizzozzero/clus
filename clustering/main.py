import click
import pandas as pd

from clustering.src.methods import kmeans, fuzzy_c_means
from clustering.src.methods.methods import get_clustering_function
from clustering.src.utils import set_manual_seed


# TODO: Lister tous les algos disponibles et leurs acronymes
# TODO: Dans chaque option, lister pour quels acronymes elles sont disponibles (ou s'ils le sont pour tous sauf ...)
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
@click.option("--delimiter", "--sep", type=str, default=",", help="")
# Clustering options
@click.option("-c", "-k", "--components", type=int, default=100,
              help="Number of clustering components")
@click.option("--eps", type=float, default=0.001,
              help="Minimal threshold caracterizing an algorithm's convergence")
@click.option("--max-iter", type=int, default=1000,
              help="Maximal number of iteration to make before stopping an algorithm")
@click.option("-m", "--fuzzifier", type=int, default=2,
              help="Fuzzification exponent applied to the membership degrees")
# Miscellaneous options
@click.option("--seed", type=int, default=None,
              help="Random seed to set")
def main(dataset, clustering_algorithm, delimiter, components, eps, max_iter, fuzzifier, seed):
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
    if seed is not None:
        set_manual_seed(seed)

    # Load the clustering algorithm
    clustering_algorithm = get_clustering_function(clustering_algorithm)

    # Load data
    data = pd.read_csv(dataset, delimiter=delimiter).values

    # Perform the clustering method
    affectations, centroids, losses = clustering_algorithm(data, components=components, eps=eps, max_iter=max_iter,
                                                           fuzzifier=fuzzifier)

    # print("Affectations :\n", affectations)
    # print("Centroids    :\n", centroids)
    print("Losses       :\n", losses)


if __name__ == '__main__':
    main(dataset="qualbank", eps=0, max_iter=1000, seed=0)
