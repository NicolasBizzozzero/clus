import numpy as np

import scipy.spatial.distance as distance
from sklearn.datasets import load_iris


def agglomerative_hierarchical_clustering(data, linkage_method="single",
                                          metric="euclidean"):
    linkage_mtx = np.empty(dtype=np.float64, shape=(data.shape[0] - 1, 4))

    distance_mtx = distance.pdist(data, metric=metric)  # Condensed dist mtx

    # Compute linkage
    # https://github.com/scipy/scipy/blob/8dba340293fe20e62e173bdf2c10ae208286692f/scipy/cluster/_hierarchy.pyx


def divisive_hierarchical_clustering(data):
    pass


def main():
    data = load_iris(False).data
    print("Shape :", data.shape)

    clus_results = agglomerative_hierarchical_clustering(data)


if __name__ == '__main__':
    main()
