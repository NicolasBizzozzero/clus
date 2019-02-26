import click

from models import kmeans, fuzzy_c_means
from utils import set_manual_seed
from dataset import load_dataset


# TODO: Lister tous les algos disponibles et leurs acronymes
# TODO: Dans chaque option, lister pour quels acronymes elles sont disponibles (ou s'ils le sont pour tous sauf ...)
@click.command(help=("Apply a clustering algorithm to a dataset. "))
@click.argument("dataset", type=str)
@click.option("--eps", type=float, default=None,
              help=("Minimal threshold caracterizing an algorithm's convergen"
                    "ce"))
@click.option("--max_iter", type=int, default=None,
              help=("Maximal number of iteration to make before stopping an a"
                    "lgorithm"))
@click.option("--seed", type=int, default=None,
              help="Random seed to set")
def main(dataset, eps, max_iter, seed):
    if seed is not None:
        set_manual_seed(seed)

    datax, datay = load_dataset(dataset)
    affectations, centroids, losses = kmeans(datax, k=2, eps=eps,
                                             max_iter=max_iter)
    # affectations, centroids, losses = fuzzy_c_means(datax, c=2, eps=eps,
    #                                                 max_iter=max_iter)
    print(losses)


if __name__ == '__main__':
    main(dataset="qualbank", eps=0, max_iter=1000, seed=0)
