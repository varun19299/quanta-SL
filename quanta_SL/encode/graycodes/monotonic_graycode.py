"""
Source: https://sciyoshi.com/2010/12/gray-codes/
"""
from numba import njit


@njit(cache=True)
def rotate_right(x, n):
    return x[-n:] + x[:-n]


@njit(cache=True)
def pi(n):
    if n <= 1:
        return [0]
    x = pi(n - 1) + [n - 1]
    return rotate_right([x[k] for k in x], 1)


def p(n, j, reverse=False):
    if n == 1 and j == 0:
        if not reverse:
            yield [0]
            yield [1]
        else:
            yield [1]
            yield [0]
    elif j >= 0 and j < n:
        perm = pi(n - 1)
        if not reverse:
            for x in p(n - 1, j - 1):
                yield [1] + [x[k] for k in perm]
            for x in p(n - 1, j):
                yield [0] + x
        else:
            for x in p(n - 1, j, reverse=True):
                yield [0] + x
            for x in p(n - 1, j - 1, reverse=True):
                yield [1] + [x[k] for k in perm]


def monotonic(n):
    for i in range(n):
        for x in p(n, i, reverse=i % 2):
            yield x


if __name__ == "__main__":
    from quanta_SL.encode import monotonic_gray_message

    print(monotonic_gray_message(4))
