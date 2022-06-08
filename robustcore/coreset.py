import numpy as np
from collections import deque
from typing import Callable
from functools import reduce, partial
from heapq import heappush, heappop

from robustcore.base import Coreset
from robustcore.matroid import Matroid
from robustcore.sieve import SieveThresholds
from robustcore.util import ge, le, gt, lt, \
    exponent, greedy_topd, greedy, sample_useless


class GreedyCoreset(Coreset):
    ''' offline greedy'''
    def __init__(self, f, d, M: Callable, **kwargs):
        super().__init__(f, d, M, **kwargs)
        self.V = set()

    def memory(self):
        return self.V.difference(self.deleted)

    def summarize(self, v=None):
        if v is not None:
            self.V.add(v)
            return set()

        self.sol = greedy(self.V, self.f, self.M)


class GreedyBackupCoreset(Coreset):
    def __init__(self, f, d, M: Callable, **kwargs):
        super().__init__(f, d, M, **kwargs)
        self.V = set()
        self.topd = set()
        self.Cs = list()
        self.w = dict() # v2gain, only for items in sol
        self.is_unifM = kwargs['is_unifM'] if 'is_unifM' in kwargs else False

    def memory(self):
        m = self.topd.union(set.union(*[C for C in self.Cs]))
        return m.difference(self.deleted)

    def summarize(self, v=None):
        if v is not None:
            self.V.add(v)
            return set()
        V = self.V.copy()

        # isolate top-d
        fs = dict([(v, self.f({v})) for v in V])
        topd = greedy_topd(self.f, V, self.d+1, fs=fs)
        for _,v in topd:
            self.topd.add(v)
            V.remove(v)
            fs.pop(v)

        # build coreset
        for i in range(self.sol.rank):
            # find top items
            l = max(1, self.d/self.eps/(i+1))
            fsol = self.f(self.sol)
            f = lambda x: self.f(self.sol.union(x)) - fsol
            tops = greedy_topd(f, V, l, fs=fs)

            # sample-and-keep
            mg, u = sample_useless(tops)
            self.sol.add(u)
            self.w[u] = mg
            self.Cs.append(set([v for _, v in tops]))

            # clean V
            for _,v in tops:
                V.remove(v)
                fs.pop(v)
            to_rm = [v for v in V if not self.sol.union({v}).feasible()]
            for v in to_rm:
                V.remove(v)
                fs.pop(v)
            if len(V) == 0:
                break

    def select(self):
        memory = self.memory()
        if len(memory) == 0:
            return self.M(sol=set())
        # if self.is_unifM:
        #     return self.select_unifM(memory)
        return self.select_pM(memory)

    def select_pM(self, memory):
        gsol = greedy(memory, self.f, self.M)
        sol = self.sol.difference(self.deleted)

        allsols = [sol, gsol]
        istar = np.argmax([self.f(sol) for sol in allsols])
        return allsols[istar]

    def select_unifM(self, memory):
        # prepare thresholds
        fs = dict([(v, self.f({v})) for v in memory])
        k = self.sol.rank
        base = self.eps + 1
        top = max(fs.values())
        thr = SieveThresholds(k, base)
        excl, incl = thr.update_thrs(top)

        # one tentative sol for each threshold
        sols = []
        for t in thr.thresholds:
            psol = set([u for u in self.sol
                        if u in memory and ge(self.w[u],t)])
            sol = self.M(sol=psol)
            fsol = self.f(sol)
            for v in memory:
                if sol.union({v}).feasible() and ge(self.f(sol.union({v})) - fsol, t):
                    sol.add(v)
                    self.f(sol)
            sols.append(sol)

        # one more greedy sol
        gsol = greedy(memory, self.f, self.M)

        allsols = sols + [gsol]
        istar = np.argmax([self.f(sol) for sol in allsols])
        return allsols[istar]


class ExchangeCoreset(Coreset):
    def __init__(self, f, d, M, **kwargs):
        super().__init__(f, d, M, **kwargs)
        self.ratio = kwargs['exc_ratio'] if 'exc_ratio' in kwargs else 2
        self.use_sampling = kwargs['use_sampling'] if 'use_sampling' in kwargs else False
        self.w = dict()
        self.C = dict() # v2w
        self.sC = self.d / self.eps if self.use_sampling else 1

    def memory(self):
        m = self.sol.sol.union(set(self.C.keys()))
        return m.difference(self.deleted)

    def exchange(self, v, mg=None):
        if mg is None:
            mg = self.f(self.sol.union({v})) - self.f(self.sol)

        swaps = self.sol.exchange(v, self.w)
        mg_old = sum([self.w[u] for u in swaps])
        if ge(mg, self.ratio * mg_old):
            self.w[v] = mg
            for u in swaps:
                self.sol.remove(u)
                self.w.pop(u)
            self.sol.add(v)
            return True, swaps

        return False, {v}

    def summarize(self, v=None):
        if v is None:
            return set()

        # Insert into buffer C
        mg = self.f(self.sol.union({v})) - self.f(self.sol)
        self.C[v] = mg
        if len(self.C) < self.sC:
            return set()

        # Exchange
        mg, v = sample_useless(self.C)
        self.C.pop(v)
        res, swaps = self.exchange(v, mg)
        if res: # exchange happens
            # Update buffer
            fsol = self.f(self.sol)
            for u,g in self.C.items():
                newg = self.f(self.sol.union({u})) - fsol
                self.C[u] = newg

        return swaps

    def select(self):
        if self.sC == 1:
            return self.sol.difference(self.deleted)

        memory = self.memory()
        for v in memory:
            self.exchange(v)
        sol = self.sol.difference(self.deleted)

        gsol = greedy(memory, self.f, self.M)
        allsols = [sol, gsol]
        istar = np.argmax([self.f(sol) for sol in allsols])
        return allsols[istar]


class CascadingCoreset(Coreset):
    def __init__(self, f, d, M, C: Callable, **kwargs):
        super().__init__(f, d, M, **kwargs)
        # build on top of another non-robust streaming Coreset C
        # create d+1 cascading solutions
        self.Cs = []
        for _ in range(self.d+1):
            #self.Cs.append(C(f, d, M, **kwargs))
            self.Cs.append(C(f=f, M=M, **kwargs))

    def memory(self):
        m = set.union(*[C.memory() for C in self.Cs])
        return m.difference(self.deleted)

    def summarize(self, v=None, idx: int=None):
        if v is None:
            return set()

        swaps = {v}
        for i, C in enumerate(self.Cs):
            if idx is not None and i != idx:
                continue
            if len(swaps) == 0:
                break
            swaps_nxt = set()
            for u in swaps:
                _ = C.summarize(u)
                swaps_nxt = swaps_nxt.union(_)
            swaps = swaps_nxt

    def select(self):
        '''
        selection after multi rounds of deletions is undefined.
        May return emptyset when k nears n, i.e., dk > n.
        '''
        # Find an intact coreset C
        ms = [C.memory() for i, C in enumerate(self.Cs)]
        j = 0 # first intact sol, j=0 if no deletion
        for i, C in enumerate(self.Cs):
            for v in self.deleted:
                if v in ms[i]:
                    j = i+1
                    break
            if j == i:
                break
        if j == len(self.Cs): # no intact C
            j = len(self.Cs) - 1

        # Send items in previous coresets to C
        C = self.Cs[j]
        swaps = set()
        for i in range(j):
            for v in ms[i]:
                self.summarize(v, idx=j)
        sol = C.select()
        sol = sol.difference(self.deleted)

        gsol = greedy(set.union(*ms).difference(self.deleted), self.f, self.M)
        allsols = [sol, gsol]
        istar = np.argmax([self.f(sol) for sol in allsols])
        return allsols[istar]


class DuttingCoreset(Coreset):
    '''
    Sieve + Exchange
    Maintain a single sol for Exc.
    Keep a cand set for each thr.
    Add v to the cand wrt the nearest thr.
    '''
    def __init__(self, f, d, M, **kwargs):
        super().__init__(f, d, M, **kwargs)
        self.C = ExchangeCoreset(f, d, M, use_sampling=False)
        self.topd = [] # a heap of tuples (value, item)
        self.top = None
        self.base = 1 + self.eps
        self.thr = SieveThresholds(self.sol.rank, self.base)
        self.cands = deque()

    def memory(self):
        topd = set([v for _,v in self.topd])
        m = topd.union(set.union(*[cand for cand in self.cands]))
        m = m.union(self.sol.sol)
        return m.difference(self.deleted)

    def update_topd(self, v):
        '''
        Update top d singletons
        :return: the swapped item in top-d if available; o.w. None
        '''
        if self.d == 0:
            return v
        fv = self.f({v})
        if len(self.topd) < self.d:
            heappush(self.topd, (fv, v))
            return None
        # now: len(self.topd) == self.d
        if lt(self.topd[0][0], fv): # swap v with the top d-th
            heappush(self.topd, (fv, v))
            fv, v = heappop(self.topd)
        return v

    def update_thrs(self, fv):
        # Update top singleton, thrs, and cands
        excl, incl = self.thr.update_thrs(fv)
        for _ in excl:
            self.cands.popleft()
        for _ in incl:
            self.cands.append(set())  # item2gain
        assert len(self.cands) == len(self.thr.exps)

    def add_v_to_cands(self, v):
        # Find closest thr
        fsol = self.f(self.sol)
        mg = self.f(self.sol.union({v})) - fsol
        j = self.thr.cloest_thr_le_than(mg)
        if j is None: # mg too small
            return None

        cand = self.cands[j]
        cand.add(v)
        return j

    def sample_and_update(self, j: int):
        cand = self.cands[j]
        if len(cand) >= self.d / self.eps:
            vs = [u for u in cand]
            idx = np.random.randint(0, len(vs))
            v = vs[idx]
            cand.remove(v)

            # Update Exc
            swaps = self.C.summarize(v)
            if swaps != {v}: # update happens
                # Update all cands
                fsol = self.f(self.sol)
                V = set.union(*[cand for cand in self.cands])
                for j in range(len(self.cands)):
                    self.cands[j] = set()

                for v in V:
                    mg = self.f(self.sol.union({v})) - fsol
                    j = self.thr.cloest_thr_le_than(mg)
                    if j is None:  # mg too small
                        continue
                    self.cands[j].add(v)

    def summarize(self, v=None):
        '''TODO: didn't return correct swaps'''
        if v is None:
            return set()

        v = self.update_topd(v) # swap good v with top d-th
        if v is None:
            return set()

        fv = self.f({v})
        if np.isclose(fv, 0): # o.w. update_thrs may fail
            return {v}
        excl = self.update_thrs(fv) # may discard some cands

        j = self.add_v_to_cands(v)
        if j is None: # mg of v too small
            return {v}
        self.sample_and_update(j)
        while True:
            flag = False
            for j,cand in enumerate(self.cands):
                if len(cand) >= self.d / self.eps:
                    flag = True
                    self.sample_and_update(j)
                    break
            if not flag:
                break

        return set()

    def select(self):
        memory = self.memory()

        gsol = greedy(memory, self.f, self.M)
        sol = self.sol.difference(self.deleted)

        allsols = [sol, gsol]
        istar = np.argmax([self.f(sol) for sol in allsols])
        return allsols[istar]
