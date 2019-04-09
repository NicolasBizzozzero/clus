import numpy as np


def ambiguity(memberships):
    """ Compute the ambiguity of a memberships matrix.
    The ambiguity of a memberships matrix is defined as the vector containing for each sample the differences of the two
    highest memberships he has.
    """
    partition = -np.partition(-memberships, [0, 1], axis=1)
    top1 = partition[:, 0]
    top2 = partition[:, 1]
    return top1 - top2


def entropy():
    pass


if __name__ == "__main__":
    pass
