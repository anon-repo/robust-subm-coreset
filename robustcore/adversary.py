import numpy as np
from heapq import heappush, heappop

from robustcore.base import Adversary
from robustcore.util import ge, le, gt, lt, greedy_topd


class StoGreedyAdversary(Adversary):
    def deletions(self):
        if self.d == 0:
            return set()
        n = len(self.S)
        if self.d >= n:
            return set(self.S)

        D = set()
        fs = dict([(v, self.f({v})) for v in self.S])
        for _ in range(self.d):
            fsol = self.f(D)
            sz = max(self.d, len(self.S) // self.d) # avoid small S
            sel = set(np.random.choice(len(self.S), size=sz, replace=True))
            subset = [v for i,v in enumerate(self.S) if i in sel]
            f = lambda x: self.f(D.union(x)) - fsol
            top1 = greedy_topd(f, subset, 1, fs=fs)
            fmax, vmax = top1[0]
            assert vmax is not None
            D.add(vmax)
        return D
