import numpy as np

from scipy.stats import binom


def mini_batches(inputs, batch_size=1, allow_dynamic_batch_size=False,
                 shuffle=True):
    """ Generator that inputs a group of examples in numpy.ndarray by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is
        not a multiple of batch_size, this may result in unexpected behaviour
        if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before
        return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> for batch in mini_batches(inputs=X, batch_size=2, shuffle=False):
    >>>     print(batch)
    array([['a', 'a'], ['b', 'b']], dtype='<U1')
    array([['c', 'c'], ['d', 'd']], dtype='<U1')
    array([['e', 'e'], ['f', 'f']], dtype='<U1')

    Source
    ------
    https://github.com/tensorlayer/tensorlayer/blob/6fea9d9d165da88e3354f723c89a0a6ccf7d8e53/tensorlayer/iterate.py#L15
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple
    # of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        if isinstance(inputs, list) and shuffle:
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt]
        else:
            yield inputs[excerpt]


def mini_batches_idx(inputs, batch_size=1, allow_dynamic_batch_size=False,
                     shuffle=True):
    """ Generator that inputs a group of examples in numpy.ndarray by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is
        not a multiple of batch_size, this may result in unexpected behaviour
        if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before
        return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> for batch in mini_batches(inputs=X, batch_size=2, shuffle=False):
    >>>     print(batch)
    array([['a', 'a'], ['b', 'b']], dtype='<U1')
    array([['c', 'c'], ['d', 'd']], dtype='<U1')
    array([['e', 'e'], ['f', 'f']], dtype='<U1')

    Source
    ------
    https://github.com/tensorlayer/tensorlayer/blob/6fea9d9d165da88e3354f723c89a0a6ccf7d8e53/tensorlayer/iterate.py#L15
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple
    # of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield excerpt


def mini_batches_dist(inputs, distance_matrix, batch_size=1,
                      allow_dynamic_batch_size=False, shuffle=True):
    """ Generator that inputs a group of examples in numpy.ndarray and a respective distance matrix by the given batch
    size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    distance_matrix : numpy.array
        The squared distance matrix matching the inputs.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is
        not a multiple of batch_size, this may result in unexpected behaviour
        if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before
        return.

    Examples
    --------
    >>> from sklearn.neighbors.dist_metrics import DistanceMetric
    >>> X = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    >>> distance_matrix = DistanceMetric.get_metric("euclidean").pairwise(X)
    >>> for batch, batch_d in mini_batches_dist(inputs=X, distance_matrix=distance_matrix, batch_size=2, shuffle=False):
    >>>     print(batch, batch_d)
    (array([[0, 1], [2,   3]]), [[0., 2.82842712], [2.82842712, 0.]])
    (array([[4, 5], [6,   7]]), [[0., 2.82842712], [2.82842712, 0.]])
    (array([[8, 9], [10, 11]]), [[0., 2.82842712], [2.82842712, 0.]])

    Source
    ------
    https://github.com/tensorlayer/tensorlayer/blob/6fea9d9d165da88e3354f723c89a0a6ccf7d8e53/tensorlayer/iterate.py#L15
    """
    assert inputs.shape[0] == distance_matrix.shape[0] == distance_matrix.shape[1]

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple
    # of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        if (isinstance(inputs, list) or isinstance(distance_matrix, list)) and shuffle:
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt], [distance_matrix[i] for i in excerpt]
        else:
            yield inputs[excerpt], distance_matrix[excerpt][:, excerpt]


def square_idx_to_condensed_idx(idx, n):
    """ With `idx` being an iterable of indexes from a square matrix and `n` one shape of this square matrix,
    return the same iterable of indexes matching indexes from the corresponding condensed matrix.
    """
    return [(binom(2, n) - binom(2, n - 1) + (-i - 1), binom(2, n) - binom(2, n - 1) + ((n - 1) - i - 1)) for i in idx]


if __name__ == "__main__":
    pass
