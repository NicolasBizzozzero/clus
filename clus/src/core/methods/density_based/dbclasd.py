""" Python implementation of the DBCLASD algorithm: a non-parametric clustering algorithm.

@copyright: Deutsches Forschungszentrum fuer Kuenstliche Intelligenz GmbH or its licensors, as applicable (2015)
@author: Sebastian Palacio
@source: https://github.com/spalaciob/py-dbclasd
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.stats import chisquare as sci_chisquare
from scipy.stats import poisson

from clus.src.utils.decorator import wrap_max_memory_consumption


@wrap_max_memory_consumption
def dbclasd(data):
    """ Implementation of the DBCLASD clustering algorithm.
    :param data: input points as 2D-array where the first axis is the number of points and the second, the dimensions
    of each point
    :return: 1d-array with labels for each point in data.
    """
    assigned_cands = np.zeros(len(data)) - 1
    unsuccessful_cands = []
    proccessed_data = []
    nnfinder = NearestNeighbors(30, algorithm='ball_tree', p=2).fit(data)
    two_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(data)
    for pt_idx, pt in enumerate(data):
        if assigned_cands[pt_idx] == -1:
            candidates = []
            new_clust_idxs = nnfinder.kneighbors([pt])[1].flatten()  # It includes pt_idx already. shape = (1, size)
            new_clust_dists = two_nnfinder.kneighbors(data[new_clust_idxs])[0][:, 1]  # All 1-NNs distances in cluster

            # Retrieve Neighborhood
            two_clust_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(data[new_clust_idxs])
            r = two_clust_nnfinder.kneighbors(data[new_clust_idxs])[0][:, 1].max()  # Max NN distance of cluster points
            for clust_pt_idx in new_clust_idxs:
                query_nn_dists, query_nn_idxs = nnfinder.kneighbors([data[clust_pt_idx]], n_neighbors=len(data))
                answer_idxs = query_nn_idxs[query_nn_dists <= r][1:]  # Discard the input point itself

                # Update candidates
                for c_idx in answer_idxs:
                    if c_idx not in proccessed_data:
                        candidates.append(c_idx)
                        proccessed_data.append(c_idx)

            # Expand Cluster (if there are candidates and more than half of the 1NN haven't been assigned yet
            if len(candidates) == 0 or (assigned_cands[new_clust_idxs] == -1).sum() < new_clust_idxs.size / 2:
                continue

            change = True
            thresh0 = None
            while change:
                change = False
                while len(candidates) > 0:
                    new_candidate = candidates.pop(0)
                    is_stable, new_thresh = is_distribution_stable(data=data[new_clust_idxs],
                                                                   pt=data[new_candidate],
                                                                   thresh0=thresh0)
                    if thresh0 is None or (new_thresh > thresh0 and is_stable):
                        thresh0 = new_thresh
                    if is_stable:
                        # Insert into the cluster
                        new_clust_idxs = np.r_[new_clust_idxs, new_candidate]
                        new_clust_dists = np.r_[new_clust_dists, two_nnfinder.kneighbors(data[new_candidate])[0][:, 1]]
                        # Retrieve
                        answer_idxs = []
                        for clust_pt_idx in new_clust_idxs:
                            query_nn_dists, query_nn_idxs = nnfinder.kneighbors([data[clust_pt_idx]],
                                                                                n_neighbors=len(data))
                            answer_idxs = query_nn_idxs[query_nn_dists <= r][1:]  # Discard the input point itself
                        # Update
                        for c_idx in answer_idxs:
                            if c_idx not in proccessed_data:
                                candidates.append(c_idx)
                                proccessed_data.append(c_idx)
                        change = True
                    else:
                        unsuccessful_cands.append(new_candidate)
                candidates = unsuccessful_cands[:]
                unsuccessful_cands = []
            assigned_cands[new_clust_idxs] = pt_idx
    return assigned_cands


def is_distribution_stable(data, pt, thresh0=None):
    """
    Perform a chi-square test on data and compare the results to data with pt added to it.
    If the second test stays below the first one, the distribution hasn't changed.
    :param data: NxM 2d-array with N points of M dimensions.
    :param pt: point to be inserted into data to test whether adding it, changes the distribution significantly.
    :param thresh0: an external fixed threshold to compare against instead of the statistic from data. This is useful
    when setting an upper/lower bound for points that are being tested against the same set of reference points.
    :return: whether the distribution remains stable with respect to the one of data and the threshold that was used.
    """
    if thresh0 is None:
        area, gl, whole_grid = cluster_area(data)
        grid = whole_grid[whole_grid >= 1]  # Assume they are connected (they are, at least diagonally)
        grid_bins = np.unique(grid)
        lambda_hat = grid.mean()
        p_est = poisson.pmf(grid_bins, lambda_hat)
        p_est[-1] = 1 - p_est[:-1].sum()  # This last probability has to be expressed as P(x >= p_n) instead of P(x == p_n)
        chisq, p = sci_chisquare([grid[grid == i].sum() for i in grid_bins] / grid.sum(), f_exp=p_est,
                                 ddof=grid_bins.size - 2)
    else:
        chisq, p = thresh0, -1.

    area2, gl2, whole_grid2 = cluster_area(np.vstack((data, pt)))
    grid2 = whole_grid2[whole_grid2 >= 1]
    grid2_bins = np.unique(grid2)
    lambda_hat2 = grid2.mean()
    p_est2 = poisson.pmf(grid2_bins, lambda_hat2)
    p_est2[-1] = 1 - p_est2[:-1].sum()  # Same as before
    chisq_2, p_2 = sci_chisquare([grid2[grid2 == i].sum() for i in grid2_bins] / grid2.sum(), f_exp=p_est2,
                                 ddof=grid2_bins.size - 2)

    if thresh0 is None:
        new_thresh = chisq
    else:
        new_thresh = thresh0
    return chisq >= chisq_2, new_thresh


def cluster_area(data):
    """
    Approximate the area of a set of candidate points belonging to a cluster. This is only necessary to compute the
    lower bound for the radius of "retrieve_heighborhood".
    :param data: NxM 2d-array with N points of M dimensions.
    :return: area, grid_length (both floating point values), grid histogram (how many points per cell)
    """
    nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(data)
    grid_length = nnfinder.kneighbors(data)[0].max()

    # TODO: adjust the offset to fit in one cell the points with the largest distance (=grid_length)
    grid_x_lims = np.arange(np.ceil((data[:, 0].max() - data[:, 0].min()) / grid_length) + 1) * grid_length + data[:,
                                                                                                            0].min()
    grid_y_lims = np.arange(np.ceil((data[:, 1].max() - data[:, 1].min()) / grid_length) + 1) * grid_length + data[:,
                                                                                                            1].min()
    grid = np.histogram2d(data[:, 0], data[:, 1], bins=[grid_x_lims, grid_y_lims])[0]

    return (grid >= 1).sum() * grid_length, grid_length, grid


if __name__ == "__main__":
    pass
