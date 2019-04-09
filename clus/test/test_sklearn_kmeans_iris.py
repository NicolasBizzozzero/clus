import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans


PATH_IRIS_LIGHT = r"../../data/iris/iris_light.csv"


def main(random_state):
    X = pd.read_csv(PATH_IRIS_LIGHT, delimiter=";", header=None).values
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(X)
    clusters_center = kmeans.cluster_centers_
    closest_cluster = cdist(X, clusters_center, metric='euclidean').argmin(axis=-1)

    # plot
    plt.scatter(X[:, 0], X[:, 1], s=3, c=closest_cluster, marker='o')
    plt.scatter(clusters_center[:, 0], clusters_center[:, 1], c="black", s=20,
                marker='x', alpha=0.9)
    plt.show()


if __name__ == '__main__':
    seed = 0
    main(random_state=seed)
