import numpy as np
from numpy.linalg import norm
from typing import Callable
from heapq import heappush, heappop


def ge(a, b):
    return np.isclose(a,b) or a > b


def le(a, b):
    return np.isclose(a,b) or a < b


def gt(a, b):
    return not np.isclose(a,b) and a > b


def lt(a, b):
    return not np.isclose(a,b) and a < b


def exponent(n, base):
    # https://stackoverflow.com/questions/25169297/numpy-logarithm-with-base-n
    return np.log(n) / np.log(base)


def sim(a,b):
    '''
    minus 0.5 and normalize for dense vectors
    '''
    cos = np.dot(a, b) / (norm(a)*norm(b))
    cos = max(0, cos - 0.5)
    return 2 * cos


def sample_useless(cands) -> tuple:
    '''
    Uselessness sampling
    :param cands: dict[item]=gain
    :param cands: list of (gain,item)
    :return: sampled gain and item
    '''
    if len(cands) == 0:
        raise
    if type(cands) is dict:
        cands = [(g,v) for v,g in cands.items()]
    if len(cands) == 1:
        return cands[0]
    p = np.array([1 / (g + 1e-8) for g,v in cands])  # avoid 1/0
    i = np.random.choice(range(len(cands)), 1, p=p/np.sum(p))[0]
    return cands[i]


def greedy_topd(f, S, d: int, fs: dict=None):
    '''
    the point is to re-use fs across iterations (lazy evaluation).
    if d=1, a greedy step.
    f: f({v})
    :return: a heap of tuples (fv, v)
    '''
    fmax = [(-1, None)] # as a heap, min at fmax[0], only keep top d
    for v in S:
        if fs is not None and le(fs[v], fmax[0][0]):
            continue # lazy step

        fv = f({v})
        if fs is not None:
            fs[v] = fv # update fs
        if gt(fv, fmax[0][0]): # update fmax
            heappush(fmax, (fv, v))
            if len(fmax) > d:
                heappop(fmax)

    if fmax[0][1] is None: # S is empty
        heappop(fmax)
    return fmax


def greedy(S: set, f: Callable, M: Callable):
    if len(S) == 0:
        return M(sol=set())

    S = S.copy()
    gsol = M()
    f(set())
    fs = dict([(v, f({v})) for v in S])
    for _ in range(gsol.rank):
        fsol = f(gsol)
        f_ = lambda x: f(gsol.union(x)) - fsol
        top1 = greedy_topd(f_, S, 1, fs=fs)
        fmax, vmax = top1[0]
        gsol.add(vmax)

        S.remove(vmax)
        to_rm = [v for v in S if not gsol.union({v}).feasible()]
        for v in to_rm:
            S.remove(v)
        if len(S) == 0:
            break

    return gsol
