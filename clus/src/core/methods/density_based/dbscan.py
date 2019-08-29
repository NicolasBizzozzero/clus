import queue

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

from clus.src.utils.decorator import remove_unexpected_arguments, wrap_max_memory_consumption

_FORMAT_PROGRESS_BAR = r"{n_fmt}/{total_fmt} examples, elapsed:{elapsed}, ETA:{remaining}{postfix}"

_LABEL_UNASSIGNED = -2
_LABEL_NOISE = -1


# TODO: Implement shuffle

@wrap_max_memory_consumption
@remove_unexpected_arguments
def dbscan(data, eps=1e-6, min_samples=3, shuffle=False, weights=None):
    global _LABEL_UNASSIGNED, _LABEL_NOISE

    assert len(data.shape) == 2, "The data must be a 2D array"
    assert data.shape[0] > 0, "The data must have at least one example"
    assert data.shape[1] > 0, "The data must have at least one feature"
    assert (weights is None) or (len(weights) == data.shape[1]),\
        "The number of weights given must be the same as the number of features. Expected size : %s, given size : %s" %\
        (data.shape[1], len(weights))
    assert min_samples >= 1, "The min_samples parameter must be greater or equals to 1"

    if weights is not None:
        # Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data
        # weighted by the square root of the weights, see [5]
        data = data * np.sqrt(weights)

    affectations = np.ones(shape=(data.shape[0],), dtype=np.int64) * _LABEL_UNASSIGNED
    with tqdm(total=affectations.shape[0], bar_format=_FORMAT_PROGRESS_BAR) as progress_bar:
        current_label = 0
        for idx_example in range(0, affectations.shape[0]):
            if affectations[idx_example] == _LABEL_UNASSIGNED:
                idx_neighbours = find_neighbours(data, idx_example, eps)
                if len(idx_neighbours) < min_samples:
                    affectations[idx_example] = _LABEL_NOISE
                else:
                    affectations[idx_example] = current_label
                    grow_cluster(data, affectations, idx_neighbours, current_label, eps, min_samples)
                    current_label += 1

            progress_bar.update()

    return {
        "affectations": affectations,
        "number_of_clusters": np.unique(affectations).size,
        "extended_time": progress_bar.last_print_t - progress_bar.start_t,

        "silhouette": silhouette_score(data, affectations),
        "variance_ratio": calinski_harabasz_score(data, affectations),
        "davies_bouldin": davies_bouldin_score(data, affectations)
    }


def grow_cluster(data, affectations, idx_neighbours, current_label, eps, min_samples):
    global _LABEL_UNASSIGNED, _LABEL_NOISE

    # Convert idx_neighbours as a Queue
    neighbours = queue.Queue()
    for neighbour in idx_neighbours:
        neighbours.put(neighbour)

    while not neighbours.empty():
        idx_neighbour = neighbours.get()

        if affectations[idx_neighbour] == _LABEL_NOISE:
            affectations[idx_neighbour] = current_label
        elif affectations[idx_neighbour] == _LABEL_UNASSIGNED:
            affectations[idx_neighbour] = current_label
            new_neighbours = find_neighbours(data, idx_neighbour, eps)
            if new_neighbours.shape[0] >= min_samples:
                # Add all new neighbours to the queue
                for new_neighbours in new_neighbours:
                    neighbours.put(new_neighbours)


def find_neighbours(data, idx_point, eps):
    dist_data_point = cdist(data, data[idx_point, :].reshape(1, -1), metric="euclidean") ** 2
    return np.where(dist_data_point <= eps)[0]


if __name__ == '__main__':
    pass
