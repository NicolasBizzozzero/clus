import os
import ntpath


def compute_file_saving_path(dataset, clustering_algorithm, components, seed, distance, weights, fuzzifier, dir_dest,
                             extension, is_3d_visualisation=False):
    os.makedirs(dir_dest, exist_ok=True)

    dest = os.path.join(dir_dest,
                        "{dataset}_{clustering_algorithm}_{components}_{seed}_{distance}{weights}{fuzzifier}"
                        "{is_3d_visualisation}.{extension}".format(
                            dataset=os.path.splitext(ntpath.basename(dataset))[0],
                            clustering_algorithm=clustering_algorithm,
                            components=components,
                            distance=distance,
                            weights=("_(" + '-'.join(str(weights)) + ")") if distance == "weighted_euclidean" else "",
                            fuzzifier=("_" + str(fuzzifier)) if fuzzifier is not None else "",
                            seed=seed,
                            is_3d_visualisation="_3d" if is_3d_visualisation else "",
                            extension=extension
                        ))
    print(dest)
    return dest


if __name__ == "__main__":
    pass
