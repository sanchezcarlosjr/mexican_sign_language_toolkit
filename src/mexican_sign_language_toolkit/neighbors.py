import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors
import math
from functools import lru_cache
import xxhash


def standard_normalization(shape):
    shape = shape.astype(float)
    shape -= np.mean(shape, axis=0)
    shape /= np.sqrt((shape ** 2).sum())
    return shape


def procrustes_distance(mtx1, mtx2):
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    return round(np.sum(np.square(mtx1 - mtx2)), 3)


def similarity(distance):
    return max(1 - round(distance, 3), 0)


class NearestNeighborClassifier:
    def __init__(self, space):
        self.space = space

    def classify(self, matrix, threshold=0.96):
        distances, indices = self.query(matrix)
        if similarity(float(distances[0])) >= threshold:
            return self.space[np.reshape(indices, -1)][0]['name']
        return " "

    def query(self, matrix):
        pass


class NumpyHasher:
    xxh64 = xxhash.xxh64()

    def __init__(self, value: np.array) -> None:
        self.value = value
        NumpyHasher.xxh64.update(value)
        self.hash = NumpyHasher.xxh64.intdigest()
        NumpyHasher.xxh64.reset()

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, __value: object) -> bool:
        return __value.hash == self.hash


def numpy_lru_cache(maxsize: int = 128):
    def wrapper_cache(func):
        # TODO: Implement a LFU version in C
        f = lru_cache(maxsize=maxsize)(lambda n1, n2: func(n1.value, n2.value))
        return lambda a1, a2: f(NumpyHasher(a1), NumpyHasher(a2))

    return wrapper_cache


class Bruteforce(NearestNeighborClassifier):
    def __init__(self, space, distance=procrustes_distance):
        super().__init__(space)
        self.distance = distance

    def query(self, X, k=1, return_distance=True):
        n = len(self.space)
        X = standard_normalization(X)
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = self.distance(X, self.space[i]['matrix'])
        nearest_indices = np.argsort(distances)[:k]
        if return_distance:
            return distances[nearest_indices], nearest_indices
        return nearest_indices


class AutomaticNearestNeighbors(NearestNeighborClassifier):
    def __init__(self, space, distance=procrustes_distance):
        super().__init__(space)
        n = 59

        @numpy_lru_cache()
        def metric(X, Y):
            return distance(np.reshape(X, (n, -1)), np.reshape(Y, (n, -1)))

        self.nearest_neighbors = NearestNeighbors(metric=metric, n_jobs=-1)
        self.nearest_neighbors.fit([np.reshape(element['matrix'], -1) for element in self.space])

    def query(self, X, k=1, return_distance=True):
        return self.nearest_neighbors.kneighbors([np.reshape(standard_normalization(X), -1)], k, return_distance)
