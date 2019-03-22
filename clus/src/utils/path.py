import os
import ntpath


def compute_visualisation_saving_path(dataset, clustering_algorithm, components, seed, dir_dest) -> str:
    os.makedirs(dir_dest, exist_ok=True)

    return os.path.join(dir_dest, "{}_{}_{}_{}.png".format(
        os.path.splitext(ntpath.basename(dataset))[0],
        clustering_algorithm,
        components,
        seed
    ))


if __name__ == "__main__":
    pass
