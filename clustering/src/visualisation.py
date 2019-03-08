import warnings
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from clustering.src.utils import normalize_range

_TITLE_WRAP_SIZE = 60
_SIZE_EXAMPLES = 3
_SIZE_CLUSTERS_CENTER = 20
_CMAP_EXAMPLES = "hsv"
_COLOR_CLUSTERS_CENTER = "black"
_MARKER_EXAMPLES = 'o'
_MARKER_CLUSTERS_CENTER = 'x'
_ALPHA_CLUSTERS_CENTER = 0.9
_ELEVATION = 48
_AZIMUTH = 134
_DISTANCE_3D = 12


def visualise_clustering_2d(data, clusters_center, clustering_method, dataset_name, header=None,
                            show=True, saving_path=None):
    assert data.shape[-1] >= 2, "Data must have at least 2 dimensions for a 2D-visualisation"

    # Apply a 2-components PCA if the data has more than 2 dimensions
    data, clusters_center, applied_pca = _apply_pca_if_too_many_dimensions(
        data, clusters_center, n_components=2)

    # Transform the header if it exists or if a PCA has been applied
    if applied_pca:
        header = ["component_1", "component_2"]
    elif header is None:
        header = ["dimension_1", "dimension_2"]

    closest_cluster = np.linalg.norm(
        data - clusters_center[:, np.newaxis], axis=-1, ord=2).argmin(axis=0)

    # Plot the visualisation
    fig, ax = plt.subplots()

    # Set the most diverse colormap possible
    # cm = plt.get_cmap(_CMAP_EXAMPLES)
    # ax.set_prop_cycle(color=[cm(i / clusters_center.shape[0]) for i in range(clusters_center.shape[0])])
    c = _get_rainbow_color_cycle(closest_cluster)

    plt.scatter(data[:, 0], data[:, 1], c=c, s=_SIZE_EXAMPLES,
                marker=_MARKER_EXAMPLES)
    plt.scatter(clusters_center[:, 0], clusters_center[:, 1], c=_COLOR_CLUSTERS_CENTER, s=_SIZE_CLUSTERS_CENTER,
                marker=_MARKER_CLUSTERS_CENTER, alpha=_ALPHA_CLUSTERS_CENTER)

    plt.xlabel(header[0])
    plt.ylabel(header[1])
    title = _compute_title(clusters_center, clustering_method,
                           dataset_name, applied_pca, n_components_pca=2)
    plt.title("\n".join(wrap(title, _TITLE_WRAP_SIZE)))

    if saving_path is not None:
        plt.savefig(saving_path)
    if show:
        plt.show()


def visualise_clustering_3d(data, clusters_center, clustering_method, dataset_name, header=None,
                            show=True, saving_path=None):
    assert data.shape[-1] >= 3, "Data must have at least 3 dimensions for a 3D-visualisation"

    # Apply a 3-components PCA if the data has more than 3 dimensions
    data, clusters_center, applied_pca = _apply_pca_if_too_many_dimensions(
        data, clusters_center, n_components=3)

    # Transform the header if it exists or if a PCA has been applied
    if applied_pca:
        header = ["component_1", "component_2", "component_3"]
    elif header is None:
        header = ["dimension_1", "dimension_2", "dimension_3"]

    closest_cluster = np.linalg.norm(
        data - clusters_center[:, np.newaxis], axis=-1, ord=2).argmin(axis=0)

    # Plot the visualisation
    fig = plt.figure()

    # Set the most diverse colormap possible
    c = _get_rainbow_color_cycle(closest_cluster)

    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=_ELEVATION, azim=_AZIMUTH)

    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=c, s=_SIZE_EXAMPLES, cmap=_CMAP_EXAMPLES)
    ax.scatter(clusters_center[:, 0], clusters_center[:, 1], clusters_center[:, 2], c=_COLOR_CLUSTERS_CENTER,
               s=_SIZE_CLUSTERS_CENTER, alpha=_ALPHA_CLUSTERS_CENTER, marker=_MARKER_CLUSTERS_CENTER)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(header[0])
    ax.set_ylabel(header[1])
    ax.set_zlabel(header[2])
    ax.dist = _DISTANCE_3D

    title = _compute_title(clusters_center, clustering_method,
                           dataset_name, applied_pca, n_components_pca=3)
    ax.set_title("\n".join(wrap(title, _TITLE_WRAP_SIZE)))

    if saving_path is not None:
        plt.savefig(saving_path)
    if show:
        plt.show()


def _apply_pca_if_too_many_dimensions(data, clusters_center, n_components):
    applied_pca = False
    if data.shape[-1] > n_components:
        applied_pca = True
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
        clusters_center = pca.transform(clusters_center)
    return data, clusters_center, applied_pca


def _apply_tsne_if_too_many_dimensions(data, clusters_center, n_components):
    # TODO: Impossible selon l'auteur de t-SNE : https://lvdmaaten.github.io/tsne/
    # Solution potentielle : Je me dis faire une regression linéaire qui apprends le mapping et l'appliquer sur les
    # position des clusters comme il le suggère peut être pas mal
    applied_pca = False
    if data.shape[-1] > n_components:
        applied_pca = True
        tsne = TSNE(n_components=n_components)
        data = tsne.fit_transform(data)
        clusters_center = tsne.transform(clusters_center)
    return data, clusters_center, applied_pca


def _compute_title(clusters_center, clustering_method, dataset_name, applied_pca, n_components_pca):
    n_components = clusters_center.shape[0]
    if applied_pca:
        return ("{}-components PCA applied to the results of the \"{}\" clustering algorithm with {} clusters into the "
                "\"{}\" dataset").format(n_components_pca, clustering_method, n_components, dataset_name)
    else:
        return ("Results of the \"{}\" clustering algorithm with {} clusters applied to the "
                "\"{}\" dataset").format(clustering_method, n_components, dataset_name)


def _get_rainbow_color_cycle(clusters_center, borned_range=2000000):
    """

    Source:
    * https://stackoverflow.com/a/36802487
    :param clusters_center:
    :param borned_range: This value is added to the hex color codes to skip the white and black colors, white colors are
    not easily visibles in a plot, and black are already used for clusters' center.
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Normalize clusters ID to the int values of color codes [#000000, #FFFFFF]
        clusters_center = normalize_range(clusters_center.reshape(-1, 1),
                                          floor=0 + borned_range, ceil=16777215 - borned_range)
        clusters_center = clusters_center.astype(np.uint32).squeeze()

    # Convert clusters' value to hex color code
    return np.array(['#{0:06X}'.format(c) for c in clusters_center])


if __name__ == "__main__":
    pass
