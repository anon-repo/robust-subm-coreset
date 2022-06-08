import abc
from functools import partial
from typing import Callable

import numpy as np
from robustcore.util import ge, le, gt, lt


class Matroid(abc.ABC):
    '''
    The default is a uniform matroid, and could be a partition matroid.
    Behaves mostly like a set.
    '''
    def __init__(self, sol: set=None, r: tuple=None, g: Callable=lambda x: 0):
        self.r = tuple() if r is None else r # a list of ranks
        self.g = g # a grouping func V -> [k], where k = #groups
        self.c = [0] * len(self.r) # counter for each group

        self.sol = set() if sol is None else sol
        if len(self.r) > 0:
            for v in self.sol:
                j = self.g(v)
                self.c[j] = self.c[j] + 1

    @property
    def rank(self) -> int:
        # rank, or an upper bound of rank
        return sum(self.r)

    def feasible(self) -> bool:
        ans = [c <= r for c,r in zip(self.c, self.r)]
        return all(ans)

    def exchange(self, v, w: dict) -> set:
        x, wmin = None, 1e7
        jv = self.g(v)
        for u in self.sol:
            j = self.g(u)
            if j == jv and self.c[j] == self.r[j]:
                if lt(w[u], wmin):
                    x = u
                    wmin = w[u]
        return {x} if x is not None else set()

    def __len__(self):
        return len(self.sol)

    def __iter__(self):
        for v in self.sol:
            yield v

    def __contains__(self, key):
        return key in self.sol

    def copy(self, sol=None):
        sol = self.sol.copy() if sol is None else sol
        return Matroid(sol=sol, r=self.r, g=self.g)

    def add(self, key):
        self.sol.add(key)
        j = self.g(key)
        self.c[j] = self.c[j] + 1

    def union(self, another: set):
        s = self.sol.union(another)
        return self.copy(s)

    def difference(self, another: set):
        s = self.sol.difference(another)
        return self.copy(s)

    def remove(self, key):
        self.sol.remove(key)
        j = self.g(key)
        self.c[j] = self.c[j] - 1


class pMatroid(Matroid):
    def __init__(self, matroids: list, sol: set=None):
        super().__init__(sol=sol, r=(10**7,)) # as a unifM
        self.matroids = matroids
        self.sols = [m() for m in matroids]

        for v in self.sol:
            for m in self.sols:
                m.add(v)

    @property
    def rank(self) -> int:
        return min([m.rank for m in self.sols])

    def feasible(self) -> bool:
        ans = [m.feasible() for m in self.sols]
        return all(ans)

    def copy(self, sol=None):
        sol = self.sol.copy() if sol is None else sol
        return pMatroid(matroids=self.matroids, sol=sol)

    def exchange(self, v, w):
        '''
        remove the least valuable item in each infeasible matroid
        '''
        X = set()
        for m in self.sols:
            x = m.exchange(v, w)
            X = X.union(x)
        return X

    def add(self, v):
        self.sol.add(v)
        for m in self.sols:
            m.add(v)

    def union(self, another: set):
        s = self.sol.union(another)
        return self.copy(sol=s)

    def remove(self, v):
        self.sol.remove(v)
        for m in self.sols:
            m.remove(v)


UnifMatroid = lambda k: partial(Matroid, r=(k,))
ParMatroid = lambda ks: partial(Matroid, r=ks, g=lambda x: x[1])
