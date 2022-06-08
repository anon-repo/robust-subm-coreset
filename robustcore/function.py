import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
from functools import partialmethod, partial

from robustcore.base import Function
from robustcore.util import sim


class CoverageFunction(Function):
    '''
    f as a coverage function.
    '''
    def __init__(self, subset: list, is_V_eq_subset=False):
        super().__init__()
        self.cache = (set(),set()) # (last S, coverage(S))
        self.subset = subset
        self.is_V_eq_subset = is_V_eq_subset
        self.makeF()

    def coverage(self, S):
        return S

    def makeF(self):
        # Turn a subset into a func f
        def _f(S):
            if len(S) == 0:
                return 0
            S = set(S)
            Sf = self.subset

            E, D1, renew = self.hitcache(S)
            if E is not None: # cache hit
                if len(E) == 0:
                    return len(D1) / len(Sf)
                taken = self.coverage(E)
                if not self.is_V_eq_subset:
                    taken = taken.intersection(Sf)
                taken = taken.union(D1)
            else:
                taken = self.coverage(S)
                if not self.is_V_eq_subset:
                    taken = taken.intersection(Sf)

            if renew:
                self.cache = (S,taken) # renew cache
            return len(taken) / len(Sf)

        self.f = _f


class NetworkFunction(CoverageFunction):
    '''
    In a network, a node takes its neighborhood as coverage.
    Let subset=V
    '''
    def __init__(self, V: list, N: dict):
        super().__init__(V, is_V_eq_subset=True)
        self.N = N # neighbood node2set

    def coverage(self, S):
        return set.union(*[self.N[i] for i in S])


class RelDivFunction(Function):
    def __init__(self, tradeoff: int, k: int,
                 sims: dict, sims_tar: dict=None):
        '''
        sims: V x subset
        sims_tar: V x target
        '''
        super().__init__()
        self.tradeoff = tradeoff
        self.k = k
        self.ns = len(next(iter(sims.values()))) # len(subset)
        self.sims = sims
        self.sims_tar = sims_tar
        self.cache = (set(), (0, [-1]*self.ns)) # (last S, (rel, sims_max))
        self.makeF()

    def f_rel(self, S):
        if self.sims_tar is None:
            return 0
        return sum([self.sims_tar[i] for i in S])

    def f_div(self, S):
        if len(S) == 0:
            return 0
        maxs = [-1] * self.ns
        for i in S:
            sims_i = self.sims[i]
            for j in range(len(maxs)):
                maxs[j] = max(maxs[j], sims_i[j])
        return np.mean(maxs), maxs

    def makeF(self):
        '''
        f(S) = (1-t) relevant_S + t * diversity_S
        '''
        def _f(S, extra=False):
            '''
            extra=True: return extra info
            '''
            if len(S) == 0:
                if extra:
                    return 0, (None,None)
                return 0
            S = set(S)
            E, D, renew = self.hitcache(S)
            if E is not None: # cache hit
                f1, maxs = D
                if len(E) == 0:
                    f2 = np.mean(maxs)
                else:
                    _, maxsE = self.f_div(E)
                    maxs = [max(a,b) for a,b in zip(maxs, maxsE)]
                    f1 = self.f_rel(E) + f1
                    f2 = np.mean(maxs)
            else:
                f1 = self.f_rel(S)
                f2, maxs = self.f_div(S)

            if renew:
                self.cache = (S,(f1,maxs)) # renew cache

            val = (1-self.tradeoff) * f1 + self.tradeoff * f2*self.k
            if extra:
                return val, (f1, f2)
            return val

        self.f = _f


class SumMaxFunction(RelDivFunction):
    '''
    Always pick first point w (target).
    f(S) = sum_v max_{u in S+w} d(w,v) - d(u,v)
    = sum_v max_{u in S} max(0, d(w,v) - d(u,v))
    '''
    def __init__(self, k: int,
                 dists: dict, dists_tar: list):
        sims = dict()
        for v,ds in dists.items():
            sims_i = [max(0, dists_tar[j]-d) for j,d in enumerate(ds)]
            sims[v] = sims_i
        super().__init__(tradeoff=1, k=k, sims=sims, sims_tar=None)
