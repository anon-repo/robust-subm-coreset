import abc
from typing import Callable
import numpy as np
from bisect import bisect
from collections import defaultdict
from functools import partial

from robustcore.matroid import Matroid
from robustcore.util import ge, le, gt, lt, greedy


class Coreset(abc.ABC):
    def __init__(self, f, d, M: Callable, **kwargs):
        self.f = f
        self.d = d
        self.M = M # class for Matroid solution
        self.sol = self.M()
        self.deleted = set()
        self.eps = kwargs['eps'] if 'eps' in kwargs else 0.05

    @abc.abstractmethod
    def summarize(self, v=None) -> set:
        '''
        v=None indicates the end of stream
        :return: return {swapped items} if v is added; o.w. {v}.
            Only works for non-robust Coreset
        '''
        swaps = {v}
        if v is None:
            return set()
        fv = self.f({v})
        if np.isclose(fv, 0):
            return swaps

    def select(self) -> Matroid:
        sol = self.sol.difference(self.deleted)
        return sol

    def select_by_greedy(self) -> Matroid:
        memory = self.memory()
        if len(memory) == 0:
            return self.M(sol=set())
        gsol = greedy(memory, self.f, self.M)
        return gsol.difference(self.deleted)

    def delete(self, v):
        self.deleted.add(v)

    def memory(self) -> set:
        '''return the set of kept items'''
        return self.sol.sol.difference(self.deleted)


class Adversary(abc.ABC):
    def __init__(self, S, f, d):
        self.S = S # either V or coreset summary; S may be list or set
        self.f = f
        self.d = d

    @abc.abstractmethod
    def deletions(self) -> set:
        pass


class Data(abc.ABC):
    '''
    API: f(v|S)=f(v,S)
    Music: (#songs, users=[[liked songs 1 2 3],])
    Network:
    Web pages:
    Recommended vectors: f(v|S) = u'v + lambda * sum_w max_{z in S+v} w'z
    '''
    def __init__(self, V):
        self.V = V
        self.V2id = dict()
        self.deleted = set()
        for i,v in enumerate(V):
            self.V2id[v] = i

    def delete(self, v):
        return self.deleted.add(v)

    def __len__(self):
        return len(self.V)

    def __iter__(self):
        arrivals = np.random.permutation(len(self.V))
        for i in arrivals:
            v = self.V[i]
            if v in self.deleted:
                continue
            yield self.V[i]


class Function(abc.ABC):
    def __init__(self):
        self.cache = None # (last S, associated data)
        self.f = None

    @abc.abstractmethod
    def makeF(self):
        pass

    def hitcache(self, S: set):
        '''
        renew when S1 subset S and |S|-|S1|>1.
        works only if S grows sequentially.
        '''
        renew = False
        S1, D1 = self.cache
        if len(S.intersection(S1)) == len(S1):  # cache hit
            E = S.difference(S1)
            if len(E) > 1:  # false when f(v|S)
                renew = True
            return E, D1, renew
        else:
            return None, None, False
