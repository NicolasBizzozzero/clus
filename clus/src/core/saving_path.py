import os
import ntpath

import numpy as np


def compute_file_saving_path_clus(format_filename, dataset, clustering_algorithm, components, fuzzifier, seed, distance,
                                  weights, dir_dest, extension, zero_fill_components, zero_fill_fuzzifier,
                                  zero_fill_seed, zero_fill_weights):
    os.makedirs(dir_dest, exist_ok=True)

    dataset = os.path.splitext(ntpath.basename(dataset))[0]
    clustering_algorithm = clustering_algorithm.replace("_", "-")
    components = str(components).zfill(zero_fill_components)
    fuzzifier = str(fuzzifier).ljust(zero_fill_fuzzifier, "0") if fuzzifier is not None else fuzzifier
    seed = str(seed).zfill(zero_fill_seed)
    distance = distance.replace("_", "-")
    if weights is not None:
        weights = tuple(map(lambda n: str(n).zfill(zero_fill_weights), weights))
        distance += "-(" + '-'.join(weights) + ")"

    return os.path.join(dir_dest, (format_filename + ".{extension}").format(
                            dataset=dataset,
                            clustering_algorithm=clustering_algorithm,
                            components=components,
                            fuzzifier=fuzzifier,
                            seed=seed,
                            distance=distance,
                            extension=extension
                        ))


def compute_file_saving_path_dclus(format_filename, dataset, clustering_algorithm, min_samples, eps, seed, distance,
                                   weights, extension, dir_dest, zero_fill_min_samples, zero_fill_eps, zero_fill_seed,
                                   zero_fill_weights):
    os.makedirs(dir_dest, exist_ok=True)

    dataset = os.path.splitext(ntpath.basename(dataset))[0]
    clustering_algorithm = clustering_algorithm.replace("_", "-")
    min_samples = str(min_samples).zfill(zero_fill_min_samples)
    if np.isinf(eps):
        eps = "inf"
    else:
        eps = str(eps).zfill(zero_fill_eps)
    seed = str(seed).zfill(zero_fill_seed)
    distance = distance.replace("_", "-")
    if weights is not None:
        weights = tuple(map(lambda n: str(n).zfill(zero_fill_weights), weights))
        distance += "-(" + '-'.join(weights) + ")"

    return os.path.join(dir_dest, (format_filename + ".{extension}").format(
                            dataset=dataset,
                            clustering_algorithm=clustering_algorithm,
                            min_samples=min_samples,
                            eps=eps,
                            seed=seed,
                            distance=distance,
                            extension=extension
                        ))


if __name__ == "__main__":
    pass
