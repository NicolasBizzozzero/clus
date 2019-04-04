import os
import ntpath


def compute_file_saving_path(dataset, clustering_algorithm, components, seed, distance, fuzzifier, dir_dest, extension,
                             is_3d_visualisation=False) -> str:
    os.makedirs(dir_dest, exist_ok=True)

    return os.path.join(dir_dest,
                        "{dataset}_{clustering_algorithm}_{components}_{seed}_{distance}{fuzzifier}"
                        "{is_3d_visualisation}.{extension}".format(
                            dataset=os.path.splitext(ntpath.basename(dataset))[0],
                            clustering_algorithm=clustering_algorithm,
                            components=components,
                            distance=distance,
                            fuzzifier=("_" + fuzzifier) if fuzzifier is not None else "",
                            seed=seed,
                            is_3d_visualisation="_3d" if is_3d_visualisation else "",
                            extension=extension
                        ))


if __name__ == "__main__":
    pass
