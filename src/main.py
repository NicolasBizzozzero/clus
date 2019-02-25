from models import kmeans, fuzzy_c_means
from utils import set_manual_seed
from dataset import load_dataset


def main(dataset, eps=None, max_iter=None, seed=None):
    if seed is not None:
        set_manual_seed(seed)

    datax, datay = load_dataset(dataset)
    # affectations, centroids, losses = kmeans(datax, k=2, eps=eps,
    #                                          max_iter=max_iter)
    affectations, centroids, losses = fuzzy_c_means(datax, c=2, eps=eps,
                                                    max_iter=max_iter)
    print(losses)


if __name__ == '__main__':
    main(dataset="qualbank", eps=0, max_iter=1000, seed=0)
