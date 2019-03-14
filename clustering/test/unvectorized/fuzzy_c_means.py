import numpy as np


def _compute_memberships(data, centroids, fuzzifier):
    np.set_printoptions(suppress=True)
    u_ir = np.zeros(shape=(data.shape[0], centroids.shape[0]))
    for i in range(data.shape[0]):
        for r in range(centroids.shape[0]):
            d_ir = np.linalg.norm(data[i] - centroids[r], ord=2) ** 2
            if d_ir == 0:
                for s in range(centroids.shape[0]):
                    u_ir[i][s] = 0
                u_ir[i][r] = 1
                break
            big_sum = sum((d_ir / (np.linalg.norm(data[i] - centroids[s], ord=2) ** 2)) ** (2 / (fuzzifier - 1)) for s in range(centroids.shape[0]))
            u_ir[i][r] = 1 / big_sum
    return u_ir


def _compute_loss(data, memberships, centroids, fuzzifier):
    res = 0
    for i in range(centroids.shape[0]):
        for j in range(data.shape[0]):
            membership_fuzzified = memberships[j][i] ** fuzzifier
            dist_data_centroid = np.linalg.norm(data[j] - centroids[i], ord=2) ** 2
            res += membership_fuzzified * dist_data_centroid
    return res


if __name__ == "__main__":
    pass
