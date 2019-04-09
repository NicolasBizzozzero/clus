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


def partition_coefficient(memberships):
    """ Compute the partition coefficient of a memberships matrix.

    The partition coefficient is defined in [6]. The value $F_c$ it returns is
    contained between $$\frac{1}{c} \leq F_c \leq 0$$
    """
    return (np.power(memberships, 2) / memberships.shape[0]).sum()


def partition_entropy(memberships):
    """ Compute the partition entropy of a memberships matrix.

    The partition entropy is defined in [6]. The value $H_c$ it returns is
    contained between $$0 \leq H_c \leq log_a(c)$$
    """
    return -(memberships * np.log2(memberships, where=memberships != 0) / memberships.shape[0]).sum()


def entropy():
    pass


if __name__ == "__main__":
    pass
