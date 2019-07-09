import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist

import pandas as pd

from sklearn.datasets import load_iris

from clus.src.core.methods import fuzzy_c_means_select
from clus.src.core.visualisation import plot_dendrogram, visualise_clustering_2d
from clus.src.utils.random import set_manual_seed


def test(seed):
    set_manual_seed(seed)

    components = 300
    eps = 1e-4
    max_iter = 100
    fuzzifier = 2.0
    batch_size = 500
    max_epochs = 20
    min_centroid_size = 100
    max_centroid_diameter = 10000000.0

    data = pd.read_csv("/local/bizzozzero/data/S-sets/s3.csv").values
    # data = load_iris().data
    print(data.shape)
    print(data)

    clus_results = \
        fuzzy_c_means_select(data, components=components, eps=eps, max_iter=max_iter, fuzzifier=fuzzifier,
                             batch_size=batch_size, weights=None, max_epochs=max_epochs,
                             min_centroid_size=min_centroid_size, max_centroid_diameter=max_centroid_diameter,
                             initialization_method="random_choice",
                             empty_clusters_method="nothing", centroids=None)

    flat_clusters = fcluster(clus_results["linkage_mtx"], criterion="distance", t=0.649)

    visualise_clustering_2d(clus_results["good_data_idx"], clusters_center=None, affectations=flat_clusters,
                            clustering_method="fcm-select",
                            dataset_name="iris", header=None,
                            show=True, save=False, saving_path=None)
    print(data[clus_results["good_data_idx"], :2])


if __name__ == "__main__":
    test(seed=0)
