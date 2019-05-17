import os
import ntpath

import numpy as np

from clus.src.core.methods.methods import ALIASES_DBSCAN, ALIASES_OPTICS


def compute_file_saving_path(dataset, clustering_algorithm, components, seed, distance, weights, fuzzifier, dir_dest,
                             extension, zero_fill_components, zero_fill_seed, zero_fill_weights, zero_fill_fuzzifier,
                             is_3d_visualisation=False):
    os.makedirs(dir_dest, exist_ok=True)

    if weights is not None:
        weights = tuple(map(lambda n: str(n).zfill(zero_fill_weights), weights))

    return os.path.join(dir_dest,
                        "{dataset}_{clustering_algorithm}_{components}_{seed}_{distance}{weights}{fuzzifier}"
                        "{is_3d_visualisation}.{extension}".format(
                            dataset=os.path.splitext(ntpath.basename(dataset))[0],
                            clustering_algorithm=clustering_algorithm,
                            components=components.zfill(zero_fill_components),
                            distance=distance,
                            weights=("_(" + '-'.join(weights) + ")") if distance == "weighted_euclidean" else "",
                            fuzzifier=("_" + str(fuzzifier).ljust(zero_fill_fuzzifier, "0")) if fuzzifier is not None
                            else "",
                            seed=seed.zfill(zero_fill_seed),
                            is_3d_visualisation="_3d" if is_3d_visualisation else "",
                            extension=extension
                        ))


def compute_file_saving_path_dclus(dataset, clustering_algorithm, eps, min_samples, seed, distance, weights, dir_dest,
                                   extension, zero_fill_eps, zero_fill_min_samples, zero_fill_seed, zero_fill_weights,
                                   is_3d_visualisation=False):
    os.makedirs(dir_dest, exist_ok=True)

    if weights is not None:
        weights = tuple(map(lambda n: str(n).zfill(zero_fill_weights), weights))

    if (clustering_algorithm in ALIASES_OPTICS) and (np.isinf(eps)):
        eps = "inf"
    else:
        eps = eps.zfill(zero_fill_eps)

    return os.path.join(dir_dest,
                        "{dataset}_{clustering_algorithm}_{min_samples}_{eps}_{seed}_{distance}{weights}"
                        "{is_3d_visualisation}.{extension}".format(
                            dataset=os.path.splitext(ntpath.basename(dataset))[0],
                            clustering_algorithm=clustering_algorithm,
                            min_samples=min_samples.zfill(zero_fill_min_samples),
                            eps=eps,
                            distance=distance,
                            weights=("_(" + '-'.join(weights) + ")") if distance == "weighted_euclidean" else "",
                            seed=seed.zfill(zero_fill_seed),
                            is_3d_visualisation="_3d" if is_3d_visualisation else "",
                            extension=extension
                        ))


if __name__ == "__main__":
    pass
