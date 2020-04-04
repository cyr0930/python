"""
The main reason why NumPy is so efficient for numerical computations is that NumPy arrays use contiguous blocks of
memory that can be efficiently cached by the CPU.
NumPy arrays have a fixed size and are homogenous. Homogenous ndarray objects have the advantage that NumPy can carry
out operations using efficient C loops and avoid expensive type checks and other overheads of the Python API. Altering
the size of a NumPy array is very expensive since it requires creating a new array and carrying over the contents of the
old array that we want to expand or shrink.
"""
import numpy as np


def ndarray():
    ary = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    assert ary.dtype == np.int64
    float32_ary = ary.astype(np.float32)
    assert float32_ary.dtype == 'float32'
    assert ary.itemsize == 8    # the size of a single array element in bytes
    assert (ary.size, ary.ndim, ary.shape) == (6, 2, (2, 3))
    scalar = np.array(5)    # construct a zero-dimensional array
    assert (scalar, scalar.size, scalar.ndim, scalar.shape) == (5, 1, 0, ())


def array_construction():
    assert np.array_equal(np.eye(3), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert np.array_equal(np.diag((3, 3, 3)), np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]))
    assert np.array_equal(np.arange(1., 11., 2),
                          np.array([1., 3., 5., 7., 9.]))       # follows the same syntax as Python's range objects
    assert np.array_equal(np.linspace(0., 1., num=5),
                          np.array([0., 0.25, 0.5, 0.75, 1.]))  # create a particular number of evenly spaced values


def indexing():
    ary = np.array([[1, 2, 3], [4, 5, 6]])
    assert ary[-1, -1] == 6
    # Fancy indexing: can use tuple or list objects of non-contiguous integer indices
    assert np.array_equal(ary[:, [0, 2]], np.array([[1, 3], [4, 6]]))
    # Indexing using Boolean masks is also considered fancy indexing
    mask = (ary > 3) & (ary % 2 == 0)
    assert np.array_equal(mask, np.array([[False, False, False], [True, False, True]]))
    assert np.array_equal(ary[mask], np.array([4, 6]))


def universal_functions():
    ary = np.array([[1, 3, 2], [4, 5, 6]])
    assert np.array_equal(np.add.reduce(ary), np.array([5, 8, 8]))
    assert np.array_equal(np.add.reduce(ary, axis=1), np.array([6, 15]))
    assert np.array_equal(np.sum(ary, axis=0), np.array([5, 8, 8]))
    assert np.array_equal(ary.sum(axis=0), np.array([5, 8, 8]))
    assert np.array_equal(ary.argsort(), np.array([[0, 2, 1], [0, 1, 2]]))
    ary.sort()
    assert np.array_equal(ary, np.array([[1, 2, 3], [4, 5, 6]]))
    assert (ary.mean(), ary.std(), ary.var()) == (3.5, 1.707825127659933, 2.9166666666666665)
    assert (ary.min(), ary.max(), ary.argmin(), ary.argmax()) == (1, 6, 0, 5)


def broadcasting():
    assert np.array_equal(np.array([[4, 5, 6], [7, 8, 9]]) + np.array([1, 2, 3]), np.array([[5, 7, 9], [8, 10, 12]]))
    assert np.array_equal(np.array([[4, 5, 6], [7, 8, 9]]) + np.array([[1], [2]]), np.array([[5, 6, 7], [9, 10, 11]]))
    # If either argument is N-Dimensional, N > 2, it is treated as a stack of
    # matrices residing in the last two indexes and broadcast accordingly.
    mat1 = np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]])
    mat2 = np.array([[[1], [2]], [[3], [4]]])
    assert (mat1.shape, mat2.shape) == ((1, 2, 3, 2), (2, 2, 1))
    mat3 = np.matmul(mat1, mat2)        # the last two dimensions represent matrices
    assert mat3.shape == (1, 2, 3, 1)   # broadcasting result (1, 2), matmul result (3, 1)
    assert np.array_equal(mat3, np.array([[[[5], [11], [17]], [[53], [67], [81]]]]))


def memory_views_and_copies():
    ary = np.array([[1, 2, 3], [4, 5, 6]])
    ary[:, 1] += 99
    assert np.array_equal(ary, np.array([[1, 101, 3], [4, 104, 6]]))
    second_row = ary[1].copy()
    second_row += 99
    assert np.array_equal(ary, np.array([[1, 101, 3], [4, 104, 6]]))
    # Fancy indexing always returns a copy of an array
    this_is_a_copy = ary[:, [0, 2]]
    this_is_a_copy += 99
    assert np.array_equal(ary, np.array([[1, 101, 3], [4, 104, 6]]))


def comparison_operators_and_mask():
    ary = np.array([1, 2, 3, 4])
    assert np.array_equal((ary > 2).nonzero(), (np.array([2, 3]), ))
    assert np.array_equal(np.where(ary > 2), (np.array([2, 3]),))
    # np.where(condition, x, y), If condition is True, yield x, otherwise yield y.
    assert np.array_equal(np.where(ary > 2, 1, 0), np.array([0, 0, 1, 1]))
    ary[~(ary > 2)] = 0
    assert np.array_equal(ary, np.array([0, 0, 3, 4]))


def random_number_generators():
    np.random.seed(123)
    # uniform distribution
    print(np.random.rand(3))
    rng = np.random.RandomState(seed=456)
    print(rng.rand(3))
    # standard normal distribution
    print(rng.randn(2, 3))


def reshaping_arrays():
    # reshape doesn't create new arrays and copy values
    # the size is fixed, but the shape is not
    ary = np.array([1, 2, 3, 4, 5, 6])
    assert np.array_equal(ary.reshape(2, -1), np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.array_equal(ary.reshape(-1), np.array([1, 2, 3, 4, 5, 6]))
    # In contrast to reshape(-1), flatten returns a copy
    assert np.array_equal(ary.flatten(), np.array([1, 2, 3, 4, 5, 6]))
    # There is no efficient way to merge without creating a new array
    ary = np.array([[1, 2, 3]])
    assert np.array_equal(np.concatenate((ary, ary), axis=0), np.array([[1, 2, 3], [1, 2, 3]]))


def linear_algebra():
    row_vector = np.array([1, 2, 3])
    column_vector = row_vector[:, np.newaxis]
    assert np.array_equal(column_vector, np.array([[1], [2], [3]]))
    matrix = np.array([[1, 2], [3, 4]])
    assert np.array_equal(matrix.T, np.array([[1, 3], [2, 4]]))
    assert np.linalg.det(matrix) == -2.0000000000000004
    # If you want to perform a routine that isn't implemented in NumPy,
    # it is also worth consulting the scipy.linalg documentation


ndarray()
array_construction()
indexing()
universal_functions()
broadcasting()
memory_views_and_copies()
comparison_operators_and_mask()
random_number_generators()
reshaping_arrays()
linear_algebra()
