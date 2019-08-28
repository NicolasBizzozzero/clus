import warnings
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist

from clus.src.core.normalization import rescaling

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# Plot configuration
_TITLE_WRAP_SIZE = 60
_SIZE_EXAMPLES = 1
_SIZE_NOISE = 1
_SIZE_CLUSTERS_CENTER = 20
_CMAP_EXAMPLES = "hsv"
_COLOR_CLUSTERS_CENTER = "black"
_COLOR_NOISE = "black"
_MARKER_EXAMPLES = 'o'
_MARKER_NOISE = 'o'
_MARKER_CLUSTERS_CENTER = 'x'
_ALPHA_EXAMPLES = 0.8
_ALPHA_NOISE = 0.8
_ALPHA_CLUSTERS_CENTER = 1.0
_ELEVATION = 48
_AZIMUTH = 134
_DISTANCE_3D = 12


def visualise_clustering_2d(data, clusters_center, affectations, clustering_method, dataset_name=None, header=None,
                            noise_id=-1, show=True, save=False, saving_path=None):
    assert data.shape[-1] >= 2, "Data must have at least 2 dimensions for a 2D-visualisation"

    # Apply a 2-components t-SNE if the data has more than 2 dimensions
    data, applied_tsne = _apply_tsne_if_too_many_dimensions(data, n_components=2)

    # Transform the header if it exists or if a t-SNE has been applied
    if applied_tsne:
        header = ["component_1", "component_2"]
    elif header is None:
        header = ["dimension_1", "dimension_2"]

    # Plot the visualisation
    fig, ax = plt.subplots()

    # Set the most diverse colormap possible
    c = _get_rainbow_color_cycle(affectations)

    # Find noise indexes
    if noise_id is not None:
        noise_idx, not_noise_idx = np.where(affectations == noise_id)[0], np.where(affectations != noise_id)[0]
    else:
        noise_idx, not_noise_idx = slice(None), slice(None)

    # Plot the data
    ax.scatter(data[noise_idx, 0], data[noise_idx, 1], c=_COLOR_NOISE, s=_SIZE_NOISE,
               marker=_MARKER_NOISE, alpha=_ALPHA_NOISE)
    ax.scatter(data[not_noise_idx, 0], data[not_noise_idx, 1], c=c[not_noise_idx], s=_SIZE_EXAMPLES,
               marker=_MARKER_EXAMPLES, alpha=_ALPHA_EXAMPLES)
    if (not applied_tsne) and (clusters_center is not None):
        ax.scatter(clusters_center[:, 0], clusters_center[:, 1], c=_COLOR_CLUSTERS_CENTER, s=_SIZE_CLUSTERS_CENTER,
                   marker=_MARKER_CLUSTERS_CENTER, alpha=_ALPHA_CLUSTERS_CENTER)

    # Configure labels and title
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    title = _compute_title(np.unique(affectations).size, clustering_method, dataset_name, applied_tsne,
                           n_components_tsne=2)
    plt.title("\n".join(wrap(title, _TITLE_WRAP_SIZE)))

    if save:
        fig.savefig(saving_path)
    if show:
        plt.show()
    plt.close()


def visualise_clustering_3d(data, clusters_center, affectations, clustering_method, dataset_name, header=None,
                            noise_id=-1, show=True, save=False, saving_path=None):
    assert data.shape[-1] >= 3, "Data must have at least 3 dimensions for a 3D-visualisation"

    # Apply a 3-components t-SNE if the data has more than 3 dimensions
    data, applied_tsne = _apply_tsne_if_too_many_dimensions(data, n_components=3)

    # Transform the header if it exists or if a t-SNE has been applied
    if applied_tsne:
        header = ["component_1", "component_2", "component_3"]
    elif header is None:
        header = ["dimension_1", "dimension_2", "dimension_3"]

    # Plot the visualisation
    fig = plt.figure()
    ax = Axes3D(fig, rect=(0, 0, 1, 1), elev=_ELEVATION, azim=_AZIMUTH)

    # Set the most diverse colormap possible
    c = _get_rainbow_color_cycle(affectations)

    # Find noise indexes
    if noise_id is not None:
        noise_idx, not_noise_idx = np.where(affectations == noise_id)[0], np.where(affectations != noise_id)[0]
    else:
        noise_idx, not_noise_idx = slice(None), slice(None)

    # Plot the data
    ax.scatter(data[noise_idx, 0], data[noise_idx, 1], data[noise_idx, 2],
               c=_COLOR_NOISE, s=_SIZE_NOISE, marker=_MARKER_NOISE, alpha=_ALPHA_NOISE)
    ax.scatter(data[not_noise_idx, 0], data[not_noise_idx, 1], data[not_noise_idx, 2],
               c=c[not_noise_idx], s=_SIZE_EXAMPLES, cmap=_CMAP_EXAMPLES, alpha=_ALPHA_EXAMPLES)
    if (not applied_tsne) and (clusters_center is not None):
        ax.scatter(clusters_center[:, 0], clusters_center[:, 1], clusters_center[:, 2], c=_COLOR_CLUSTERS_CENTER,
                   s=_SIZE_CLUSTERS_CENTER, alpha=_ALPHA_CLUSTERS_CENTER, marker=_MARKER_CLUSTERS_CENTER)

    # Configure labels and title
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(header[0])
    ax.set_ylabel(header[1])
    ax.set_zlabel(header[2])
    ax.dist = _DISTANCE_3D
    title = _compute_title(np.unique(affectations).size, clustering_method,
                           dataset_name, applied_tsne, n_components_tsne=3)
    ax.set_title("\n".join(wrap(title, _TITLE_WRAP_SIZE)))
    if save:
        plt.savefig(saving_path)
    if show:
        plt.show()
    plt.close()


def visualise_clustering_loss(losses, show=True):
    plt.plot(losses)

    if show:
        plt.show()


def plot_dendrogram(linkage_mtx, depth_cut, dataset_name=None, title=None, linkage_method=None,
                    dpi=1000, show=True, save=True):
    # Plot the dendrogram
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 18)
    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.rc("font", size=6)

    dendrogram(linkage_mtx, p=depth_cut, truncate_mode="level", orientation="right", ax=ax)

    if title is None:
        if dataset_name is None:
            title = "Dendrogram of the hierarchical clustering with parameters : p={p}, linkage_method={linkage_method}".format(
                p=depth_cut,
                linkage_method=linkage_method
            )
        else:
            title = "Dendrogram of the hierarchical clustering computed from {dataset} with parameters : p={p}, linkage_method={linkage_method}".format(
                dataset=dataset_name,
                p=depth_cut,
                linkage_method=linkage_method
            )
    plt.title("\n".join(wrap(title, _TITLE_WRAP_SIZE)))
    plt.xlabel("fusion_cost")

    if save:
        fig.savefig("%s_%d.png" % (dataset_name, depth_cut))
    if show:
        plt.show()
    plt.close()


def _apply_tsne_if_too_many_dimensions(data, n_components):
    # TODO: Impossible to apply t-SNE twice (for the data and the cluster's center) : https://lvdmaaten.github.io/tsne
    #  Multiple ideas:
    #  - Keep the affectations vector, apply the TSNE on the data then recompute the clusters center.
    #  - Keep the affectations vector and only apply TSNE to the data, do not care about the clusters center.
    #  - Do a linear regression between the data before applying TSNE and after. Then apply this regression to the
    #    clusters center.

    applied_tsne = False
    if data.shape[-1] > n_components:
        applied_tsne = True
        tsne = TSNE(n_components=n_components)
        data = tsne.fit_transform(data)
        # clusters_center = tsne.transform(clusters_center)
    return data, applied_tsne


def _compute_title(n_components, clustering_method, dataset_name, applied_tsne, n_components_tsne):
    if dataset_name is None:
        dataset_name = "."
    else:
        dataset_name = " applied to the \"{}\" dataset.".format(dataset_name)

    if applied_tsne:
        return ("{}-components t-SNE applied to the results of the \"{}\" clus algorithm "
                "with {} clusters{}").format(n_components_tsne, clustering_method, n_components, dataset_name)
    else:
        return ("Results of the \"{}\" clus algorithm "
                "with {} clusters{}").format(clustering_method, n_components, dataset_name)


def _get_rainbow_color_cycle(affectations, borned_range=2000000):
    """

    Source:
    * https://stackoverflow.com/a/36802487
    :param affectations:
    :param borned_range: This value is added to the hex color codes to skip the white and black colors, white colors are
    not easily visibles in a plot, and black are already used for clusters' center.
    :return:
    """
    unique_clusters = np.unique(affectations)
    spaced_values = np.linspace(borned_range, 16777215 - borned_range, num=unique_clusters.size, dtype=np.uint64)
    spaced_clusters_center = affectations.copy()

    for id_color, id_cluster in enumerate(unique_clusters):
        spaced_clusters_center[spaced_clusters_center == id_cluster] = spaced_values[id_color]

    # Convert clusters' value to hex color code
    return np.array(['#{0:06X}'.format(c) for c in spaced_clusters_center])


if __name__ == "__main__":
    pass
