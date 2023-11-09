import numpy as np

from mexican_sign_language_toolkit.neighbors import Bruteforce, standard_normalization, AutomaticNearestNeighbors

__author__ = "sanchezcarlosjr"
__copyright__ = "sanchezcarlosjr"
__license__ = "MIT"


def test_should_find_k_nearest_neighbors_with_brute_force():
    matrix = np.array([[1, 2], [3, 4]])
    space = np.array([
        {'segment': '1', 'name': 'A', 'matrix': standard_normalization(matrix)},
        {'segment': '2', 'name': 'B', 'matrix': standard_normalization(np.array([[5, 5], [5, 4]]))}
    ])
    brute_force = Bruteforce(space)
    assert brute_force.classify(matrix) == 'A'
    assert brute_force.classify(np.rot90(matrix)) == 'A'
    assert brute_force.classify(matrix+10**10) == 'A'
    assert brute_force.classify(0.5*matrix) == 'A'
    assert brute_force.classify(0.5*np.rot90(matrix)+10**10) == 'A'


def test_should_find_k_nearest_neighbors_with_kdtree():
    matrix = np.array([[1, 2], [3, 4]])
    space = np.array([
        {'segment': '1', 'name': 'A', 'matrix': standard_normalization(matrix)},
        {'segment': '2', 'name': 'B', 'matrix': standard_normalization(np.array([[2, 4], [3, 2]]))}
    ])
    knn = AutomaticNearestNeighbors(space)
    assert knn.classify(matrix) == 'A'
    assert knn.classify(np.rot90(matrix)) == 'A'
    assert knn.classify(matrix + 10 ** 10) == 'A'
    assert knn.classify(0.5 * matrix) == 'A'
    assert knn.classify(0.5 * np.rot90(matrix) + 10**10) == 'A'
