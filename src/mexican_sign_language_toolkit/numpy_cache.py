from functools import lru_cache
import numpy as np
import xxhash


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
