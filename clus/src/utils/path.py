import os
import ntpath


def compute_file_saving_path(dataset, clustering_algorithm, components, seed, dir_dest, extension,
                             is_3d_visualisation=False) -> str:
    os.makedirs(dir_dest, exist_ok=True)

    return os.path.join(dir_dest,
                        "{dataset}_{clustering_algorithm}_{components}_{seed}{is_3d_visualisation}.{extension}".format(
                            dataset=os.path.splitext(ntpath.basename(dataset))[0],
                            clustering_algorithm=clustering_algorithm,
                            components=components,
                            seed=seed,
                            is_3d_visualisation="_3d" if is_3d_visualisation else "",
                            extension=extension
                        ))


if __name__ == "__main__":
    pass
