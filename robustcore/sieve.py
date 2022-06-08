import numpy as np
from collections import deque
import abc

from robustcore.util import exponent


class SieveThresholds(abc.ABC):
    '''
    A class to manage Sieve thresholds
    '''
    def __init__(self, k, base):
        self.k = k
        self.base = base
        self.top = None
        self.exps = deque() # only keep a range of exponents

    @property
    def thresholds(self):
        return np.power(float(self.base), self.exps)

    def cloest_thr_le_than(self, g):
        i = np.floor(exponent(g, self.base)).astype(int)
        assert i <= self.exps[-1]
        #assert i >= self.exps[0]
        if i < self.exps[0]:
            return None
        j = i - self.exps[0]
        return j

    def which_expo(self, expo):
        for i,expo_ in enumerate(self.exps): # TODO: slow
            if expo == expo_:
                return i
        return None

    def update_thrs(self, fv):
        '''
        :param fv: Assume non-dec fv
        :return: exponents newly excluded and included
        '''
        # Update top singleton
        if self.top is not None and (np.isclose(self.top, fv) or self.top > fv):
            return [], []
        self.top = fv

        # Update thrs within [top/2k, top], top thr at right
        # A thr takes a form of base^i; should not depend on top
        l = np.floor(exponent(self.top / (2 * self.k), self.base)).astype(int) # TODO: a better lower bound
        h = np.ceil(exponent(self.top, self.base)).astype(int)
        if len(self.exps) == 0:
            for i in range(l, h+1):
                self.exps.append(i)
            return [], range(l, h+1)
        res1, res2 = range(self.exps[0], l), range(self.exps[-1]+1, h+1)
        for i in res1: # keep l
            self.exps.popleft()
        for i in res2:
            self.exps.append(i)
        return res1, res2