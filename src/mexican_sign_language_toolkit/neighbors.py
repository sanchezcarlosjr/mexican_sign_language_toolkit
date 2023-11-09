import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

from mexican_sign_language_toolkit.numpy_cache import numpy_lru_cache


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
    def __init__(self, space, n=59, distance=procrustes_distance):
        super().__init__(space)

        @numpy_lru_cache()
        def metric(X, Y):
            return distance(np.reshape(X, (n, -1)), np.reshape(Y, (n, -1)))

        self.nearest_neighbors = NearestNeighbors(metric=metric, n_jobs=-1)
        self.nearest_neighbors.fit([np.reshape(element['matrix'], -1) for element in self.space])

    def query(self, X, k=1, return_distance=True):
        return self.nearest_neighbors.kneighbors([np.reshape(standard_normalization(X), -1)], k, return_distance)
