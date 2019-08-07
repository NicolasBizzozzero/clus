import scipy
from sklearn.datasets import load_iris
from sklearn.neighbors.dist_metrics import DistanceMetric

import numpy as np


def square_idx_to_condensed_idx(x, y, n):
    """ Convert (x, y) indexes from a square matrix into the index of its corresponding condensed matrix. """
    assert x != y, "No diagonal elements in a condensed matrix"

    if x < y:
        x, y = y, x
    return (n * y) - y * (y + 1) // 2 + x - 1 - y


def square_idx_to_condensed(cndsd_matrix, x, y, n):
    """ Retrieve the element of a condensed matrix from the (x, y) indexes of its corresponding square matrix. """
    return cndsd_matrix[square_idx_to_condensed_idx(x, y, n)]


def square_row_idx_to_condensed_row(cndsd_matrix, x, n):
    """ Retrieve the row of a condensed matrix from the x index of its corresponding square matrix. """
    row = np.empty(shape=(n,), dtype=cndsd_matrix.dtype)
    for y in range(n):
        if y == x:
            row[y] = 0
            continue
        row[y] = square_idx_to_condensed(cndsd_matrix, x, y, n)
    return row


def square_rows_idx_to_condensed_rows(cndsd_matrix, indexes, n):
    """ Retrieve the rows of a condensed matrix from the indexes x indexes of its corresponding square matrix. """
    rows = np.empty(shape=(n, len(indexes)), dtype=cndsd_matrix.dtype)
    for i, x in enumerate(indexes):
        row = square_row_idx_to_condensed_row(cndsd_matrix, x, n)
        rows[:, i] = row
    return rows


if __name__ == "__main__":
    data = load_iris().data[:5, :]
    distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(data)
    distance_matrix_condensed = scipy.spatial.distance.pdist(data, "euclidean")

    x, y = 2, None
    print(distance_matrix[x, :])
    print(square_row_idx_to_condensed_row(distance_matrix_condensed, x, n=distance_matrix.shape[0]))
