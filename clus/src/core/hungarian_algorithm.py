""" Provides an implementation of the Hungarian (also called Munkres or Kuhn-Munkres) algorithm, used for solving the
Assignment Problem in cubic time.


Assignment Problem
==================

Let *C* be an *n* x *n* matrix representing the costs of each of *n* workers to perform any of *n* jobs. The
assignment problem is to assign jobs to workers in a way that minimizes the total cost. Since each worker can perform
only one job and each job can be assigned to only one worker the assignments represent an independent set of the
matrix *C*.

One way to generate the optimal set is to create all permutations of the indexes necessary to traverse the matrix so
that no row and column are used more than once. For instance, given this matrix (expressed in Python):

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]

You could use this code to generate the traversal indexes:

    def permute(a, results):
        if len(a) == 1:
            results.insert(len(results), a)

        else:
            for i in range(0, len(a)):
                element = a[i]
                a_copy = [a[j] for j in range(0, len(a)) if j != i]
                subresults = []
                permute(a_copy, subresults)
                for subresult in subresults:
                    result = [element] + subresult
                    results.insert(len(results), result)

    results = []
    permute(range(len(matrix)), results) # [0, 1, 2] for a 3x3 matrix

After the call to permute(), the results matrix would look like this:

    [[0, 1, 2],
     [0, 2, 1],
     [1, 0, 2],
     [1, 2, 0],
     [2, 0, 1],
     [2, 1, 0]]

You could then use that index matrix to loop over the original cost matrix and calculate the smallest cost of the
combinations::

    n = len(matrix)
    minval = sys.maxsize
    for row in range(n):
        cost = 0
        for col in range(n):
            cost += matrix[row][col]
        minval = min(cost, minval)

    print minval

While this approach works fine for small matrices, it does not scale. It executes in O(*n*!) time: Calculating the
permutations for an *n* x *n* matrix requires *n*! operations. For a 12x12 matrix, that's 479,001,600 traversals.
Even if you could manage to perform each traversal in just one millisecond, it would still take more than 133 hours to
perform the entire traversal. A 20x20 matrix would take 2,432,902,008,176,640,000 operations. At an optimistic
millisecond per operation, that's more than 77 million years.

The Munkres algorithm runs in O(*n* ^3) time, rather than O(*n*!). This module provides an implementation of that
algorithm.

This version is based on http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html.

This version was written for Python by Brian Clapper from the (Ada) algorithm at the above web site. (The
`Algorithm::Munkres`` Perl version, in CPAN, was clearly adapted from the same web site.)
It was then rewritten by Nicolas Bizzozzéro in numpy with full vectorization for performance enhancement.


Usage
=====

Construct a Munkres object::

    from munkres import Munkres

    m = Munkres()

Then use it to compute the lowest cost assignment from a cost matrix. Here's a sample program:

    from munkres import Munkres

    matrix = [[5,  9, 1],
              [10, 3, 2],
              [8,  7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print("Lowest cost through this matrix:")

    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total cost: %d' % total

Running that program produces:

    Lowest cost through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 0) -> 5
    (1, 1) -> 3
    (2, 2) -> 4
    total cost=12

The instantiated Munkres object can be used multiple times on different matrices.


Non-square Cost Matrices
========================

The Munkres algorithm assumes that the cost matrix is square. However, it's possible to use a rectangular matrix if you
first pad it with 0 values to make it square. This module automatically pads rectangular cost matrices to make them
square.


Calculating Profit, Rather than Cost
====================================

The cost matrix is just that: A cost matrix. The Munkres algorithm finds the combination of elements (one from each
row and column) that results in the smallest cost. It's also possible to use the algorithm to maximize profit. To do
that, however, you have to convert your profit matrix to a cost matrix. The simplest way to do that is to subtract all
elements from a large value. For example::

    from munkres import Munkres

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    print('Highest profit through this matrix:')

    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)

    print 'total profit=%d' % total

Running that program produces::

    Highest profit through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 1) -> 9
    (1, 0) -> 10
    (2, 2) -> 4
    total profit=23


References
==========

1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.

3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.

5. http://en.wikipedia.org/wiki/Hungarian_algorithm


Source
=====================
https://github.com/FJR-Nancy/joint-cluster-cnn/blob/master/munkres.py


Copyright and License
=====================
Copyright 2008-2016 Brian M. Clapper
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

__all__ = ['Munkres']

_LABEL_STAR = 1
_LABEL_PRIME = 2


class Munkres:
    """ Compute the Munkres solution to the classical assignment problem.
    See the module's documentation for usage.
    """

    def __init__(self):
        pass

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.

        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)

                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.

        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix

        """
        self.C = pad_matrix(cost_matrix)
        self.n = self.C.shape[0]
        self.row_covered = np.zeros(shape=(self.n,), dtype=np.bool)
        self.col_covered = np.zeros(shape=(self.n,), dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = np.zeros(shape=(self.n, self.n), dtype=np.int64)

        done = False
        step = 1

        steps = {
            1: self.__step1,
            2: self.__step2,
            3: self.__step3,
            4: self.__step4,
            5: self.__step5,
            6: self.__step6
        }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        return np.where(self.marked)[1][:cost_matrix.shape[0]]

    def __step1(self):
        """ For each row of the matrix, find the smallest element and subtract it from every element in its row.
        Then, go to Step 2.
        """
        min_elements = self.C.min(axis=1)
        self.C -= min_elements.reshape(-1, 1)
        return 2

    def __step2(self):
        """ Find a zero (Z) in the resulting matrix. If there is no starred zero in its row or column, star Z.
        Repeat for each element in the matrix. The, go to Step 3.
        """
        idx_zeros_rows, idx_zeros_cols = np.where(self.C == 0)
        while (idx_zeros_rows.size != 0) and (idx_zeros_cols.size != 0):
            # Pop firsts indexes
            idx_zero_row, idx_zero_col = idx_zeros_rows[0], idx_zeros_cols[0]
            mask_remove_idx = (idx_zeros_rows != idx_zero_row) & (idx_zeros_cols != idx_zero_col)
            idx_zeros_cols, idx_zeros_rows = idx_zeros_cols[mask_remove_idx], idx_zeros_rows[mask_remove_idx]

            # Mark theses indexes as starred
            self.marked[idx_zero_row, idx_zero_col] = _LABEL_STAR
            self.row_covered[idx_zero_row] = True
            self.col_covered[idx_zero_col] = True

        self.__clear_covered_vectors()
        return 3

    def __step3(self):
        """ Cover each column containing a starred zero. If K columns are covered, the starred zeros describe a
        complete set of unique assignments. In this case, Go to DONE, otherwise, go to Step 4.
        """
        self.col_covered = self.marked.sum(axis=0).astype(np.bool)
        count = self.col_covered.sum()
        return 7 if count >= self.n else 4

    def __step4(self):
        """ Find a noncovered zero and prime it. If there is no starred zero in the row containing this primed zero, go
        to Step 5. Otherwise, cover this row and uncover the column containing the starred zero.
        Continue in this manner until there are no uncovered zeros left. Save the smallest uncovered value and go to
        Step 6.
        """
        while True:
            row, col = self.__find_a_zero()
            if row < 0:
                return 6
            else:
                self.marked[row][col] = _LABEL_PRIME
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    self.Z0_r = row
                    self.Z0_c = col
                    return 5

    def __step5(self):
        """ Construct a series of alternating primed and starred zeros as follows :
        Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        Continue until the series terminates at a primed zero that has no starred zero in its column. Unstar each
        starred zero of the series, star each primed zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3
        """
        path = np.zeros(shape=(self.n * 2, 2), dtype=np.int64)
        done = False

        count = 0
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count - 1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count - 1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covered_vectors()
        self.__erase_primes()
        return 3

    def __step6(self):
        """ Add the value found in Step 4 to every element of each covered row, and subtract it from every element of
        each uncovered column. Return to Step 4 without altering any stars, primes, or covered lines.
        """
        minval = self.__find_smallest()
        self.C[np.repeat(self.row_covered.reshape(-1, 1), self.n, axis=1)] += minval
        self.C[np.tile(~self.col_covered, (self.n, 1))] -= minval
        return 4

    def __find_smallest(self):
        """ Find the smallest uncovered value in the matrix. """
        return self.C[self.C != 0].min()

    def __find_a_zero(self):
        """ Find the first uncovered elements with value 0. """
        possible_rows = np.repeat(~self.row_covered.reshape(-1, 1), self.n, axis=1)
        possible_cols = np.tile(~self.col_covered, (self.n, 1))
        possible_zeros = self.C == 0

        idx_zeros = np.where(possible_rows & possible_cols & possible_zeros)
        if idx_zeros[0].size == 0:
            return -1, -1
        else:
            return idx_zeros[0][0], idx_zeros[1][0]

    def __find_star_in_row(self, row):
        """ Find the first starred element in the specified row. Returns the row index, or -1 if no starred element
        was found.
        """
        starred_elements = np.where(self.marked[row] == _LABEL_STAR)[0]
        if starred_elements.size == 0:
            return -1
        return starred_elements[0]

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified column. Returns the column index, or -1 if no starred element
        was found.
        """
        starred_elements = np.where(self.marked[:, col] == _LABEL_STAR)[0]
        if starred_elements.size == 0:
            return -1
        return starred_elements[0]

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        starred_elements = np.where(self.marked[row] == _LABEL_PRIME)[0]
        if starred_elements.size == 0:
            return -1
        return starred_elements[0]

    def __convert_path(self, path, count):
        for idx in path[:count + 1, :2]:
            row, col = idx
            if self.marked[row][col] == _LABEL_STAR:
                self.marked[row][col] = 0
            else:
                self.marked[row][col] = _LABEL_STAR

    def __clear_covered_vectors(self):
        """ Clear all covered matrix cells. """
        self.row_covered = np.zeros(shape=(self.n,), dtype=np.bool)
        self.col_covered = np.zeros(shape=(self.n,), dtype=np.bool)

    def __erase_primes(self):
        """ Erase all prime markings. """
        self.marked[self.marked == _LABEL_PRIME] = 0


def pad_matrix(matrix, pad_value=0):
    """ Pad a possibly non-square matrix to make it square. """
    if matrix.shape[0] == matrix.shape[1]:
        # Matrix already square
        return matrix
    elif matrix.shape[0] < matrix.shape[1]:
        # Need to add rows
        diff = matrix.shape[1] - matrix.shape[0]
        return np.concatenate((matrix, np.full(shape=(diff, matrix.shape[1]), fill_value=pad_value)), axis=0)
    else:
        # Need too add columns
        diff = matrix.shape[0] - matrix.shape[1]
        return np.concatenate((matrix, np.full(shape=(matrix.shape[0], diff), fill_value=pad_value)), axis=1)


if __name__ == '__main__':
    pass
