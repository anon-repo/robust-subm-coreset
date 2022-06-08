import pytest
import numpy as np
from functools import partial

from robustcore.base import Data
from robustcore.matroid import Matroid, UnifMatroid, ParMatroid, pMatroid
from robustcore.function import CoverageFunction, RelDivFunction
from robustcore.util import sim
from robustcore.coreset import GreedyCoreset, ExchangeCoreset, \
    GreedyBackupCoreset, CascadingCoreset, DuttingCoreset
from robustcore.adversary import StoGreedyAdversary


@pytest.fixture
def cov():
    V = np.arange(5)
    subset = [0,1,2]
    k = 2
    return (V,subset,k)


@pytest.fixture
def vec():
    np.random.seed(42)
    V = np.random.rand(5, 2)
    tar = [0.5, 0.5]
    k = 2
    tradeoff = 0.5
    sims = dict()
    for i,v in enumerate(V):
        sims[i] = [sim(v,u) for u in V]
    tarsims = dict([(i, sim(v,tar)) for i,v in enumerate(V)])
    return (V,tar,tradeoff,k,sims,tarsims)


@pytest.fixture
def mod():
    n = 11
    V = np.arange(n)
    f = lambda S: sum(S)
    d = 2
    k = 2
    dat = Data(V)
    return (dat, f, d, k)


def test_matroid():
    m1 = Matroid(r=(2,), sol={4})
    m2 = m1.union({5})
    assert len(m2) == 2
    assert m2.rank == 2
    assert m2.feasible()
    m2.add(6)
    assert not m2.feasible()
    m2.remove(6)
    assert m2.feasible()

    m = Matroid(r=(3,), sol={1,2,3})
    assert m.feasible()
    assert not m.union({4}).feasible()
    m.add(4)
    assert not m.feasible()

    # partition matroid
    g = lambda x: 0 if x < 10 else 1
    m = Matroid(r=(2,1), sol={1,2,11}, g=g)
    assert len(m) == 3
    assert not m.union({3}).feasible()
    assert not m.union({12}).feasible()

    # p-matroid
    V = [(0,0),(0,1),(1,0),(2,0)] # (i, its possible group)
    m1 = partial(Matroid, r=(2,1), g=lambda x: x[1])
    m2 = partial(Matroid, r=(1,1,1), g=lambda x: x[0])
    m = pMatroid(matroids=[m1,m2])
    for v in V:
        m.add(v)
    assert not m.feasible()
    m.remove((0,0))
    assert m.feasible()

    # exchange
    m1 = partial(Matroid, r=(2,))
    m = pMatroid(matroids=[m1])
    w = dict([(i, i) for i in [1,2,3]])
    assert len(m) == 0
    m.add(1)
    m.add(2)
    assert 1 in m and 2 in m
    X = m.exchange(3, w)
    assert len(X) == 1
    assert 1 in X


def test_coverage_func(cov):
    V,subset,k = cov
    F = CoverageFunction(subset)
    f = F.f

    assert f(set()) == 0
    assert np.isclose(f(set([1,2])), 2/3)
    assert np.isclose(f(set([1,3])), 1/3)
    assert np.isclose(f(set(V)), 1)


def test_rel_div_func(vec):
    V,tar,tradeoff,k,sims,sims_tar = vec
    V_ = np.arange(len(V))
    F = RelDivFunction(tradeoff, k, sims, sims_tar)
    f = F.f

    assert f(set()) == 0
    assert np.isclose(f(set([0])), 1.2927307447780727)
    assert np.isclose(f(set([0,1])), 1.8846777774425851)


def test_adv(mod):
    dat, f, d, k = mod

    np.random.seed(42)
    '''
    {10, 3, 6}
    {4, 6, 7}
    {9, 2, 6}
    '''
    Adv = StoGreedyAdversary(dat.V, f, 3)
    D = Adv.deletions()
    assert len(D) == 3
    assert D == {10,7,9}

    Adv = StoGreedyAdversary([1, 2, 3], f, 5)
    assert Adv.deletions() == {1, 2, 3}


def test_Greedy(mod):
    dat, f, d, k = mod
    C = GreedyCoreset(f, d, UnifMatroid(k))
    for v in dat:
        C.summarize(v)
    C.summarize(None)
    assert C.select().sol == {9,10}
    C.delete(9)
    assert C.select().sol == {10}


def test_Backup(mod):
    dat, f, d, k = mod
    C = GreedyBackupCoreset(f, d, UnifMatroid(k), is_unifM=True)
    for v in dat:
        C.summarize(v)
    C.summarize(None)
    assert C.select().sol == {9,10}
    C.delete(9)
    assert C.select().sol == {10,8}


def test_Dutting(mod):
    np.random.seed(42)
    dat, f, d, k = mod
    C = DuttingCoreset(f, d, UnifMatroid(k), eps=1)
    for v in dat.V:
        C.summarize(v)
    C.summarize(None)
    #print(C.cands) # [{2}, {7}, {8}]
    assert C.C.sol.sol == {4,5}
    assert C.topd == [(9, 9), (10, 10)]
    assert C.select().sol == {9,10}
    C.delete(9)
    C.delete(8)
    assert C.select().sol == {10,7}


def test_Cascade(mod):
    dat, f, d, k = mod
    innerC = partial(ExchangeCoreset, use_sampling=False, d=d)
    C = CascadingCoreset(f, d, UnifMatroid(k), innerC)
    for v in dat.V:
        C.summarize(v)
    C.summarize(None)
    # 1+2, 2+3, 3+4, no 5, 4+6, no 7, 6+8
    # 1, 1+2, 2+5, no 3, 5+7, no 4, no 9, 7+10
    # initial sols: {6,8}, {7, 10}, {4, 9}
    assert C.Cs[0].sol.sol == {6,8}
    assert C.Cs[1].sol.sol == {7,10}
    C.delete(8)
    #assert C.select().sol == {7,10}
    assert C.select().sol == {9,10}


#@pytest.mark.skip(reason="TODO")
def test_Exchange():
    '''
    print('Chekuri:', self.sol.sol, v, f'{mg} vs. {mg_old}', C)
    Chekuri: set() (0, 0) 0.15384615384615385 vs. 0 {}
    Chekuri: {(0, 0)} (0, 1) 0.23076923076923078 vs. 0.15384615384615385 {(0, 0): 0.15384615384615385}
    Chekuri: {(0, 0)} (1, 0) 0.16923076923076924 vs. 0 {}
    Chekuri: {(1, 0), (0, 0)} (2, 1) 0.15384615384615385 vs. 0 {}
    Chekuri: {(1, 0), (2, 1), (0, 0)} (2, 0) 0.2923076923076923 vs. 0.3076923076923077 {(0, 0): 0.15384615384615385, (2, 1): 0.15384615384615385}
    Chekuri: set() (0, 0) 0.10416666666666667 vs. 0 {}
    Chekuri: {(0, 0)} (0, 1) 0.15625 vs. 0.10416666666666667 {(0, 0): 0.10416666666666667}
    Chekuri: {(0, 0)} (1, 0) 0.11458333333333336 vs. 0 {}
    Chekuri: {(1, 0), (0, 0)} (2, 1) 0.10416666666666666 vs. 0 {}
    Chekuri: {(1, 0), (2, 1), (0, 0)} (2, 0) 0.5208333333333333 vs. 0.20833333333333331 {(0, 0): 0.10416666666666667, (2, 1): 0.10416666666666666}
    '''
    V = [(0,0),(0,1),(1,0),(2,1),(2,0)] # (i, its possible group)
    f = lambda S, **kwargs: sum([w[i] for i in S]) / sum(w.values())
    m1 = partial(Matroid, r=(2,1), g=lambda x: x[1])
    m2 = partial(Matroid, r=(1,1,1), g=lambda x: x[0])
    M = partial(pMatroid, matroids=[m1,m2])
    d = 2
    k = 2

    w = dict(zip(V, [1, 1.5, 1.1, 1, 1.9]))
    C = ExchangeCoreset(f, d, M, use_sampling=False)
    for v in V:
        C.summarize(v)
    C.summarize(None)
    sol = C.select()
    assert (0,0) in sol and (1,0) in sol and (2,1) in sol and len(sol) == 3

    w = dict(zip(V, [1, 1.5, 1.1, 1, 5]))
    C = ExchangeCoreset(f, d, M, use_sampling=False)
    for v in V:
        C.summarize(v)
    C.summarize(None)
    sol = C.select()
    assert (1,0) in sol and (2,0) in sol and len(sol) == 2

    D = {(2,0)}
    for v in D:
        C.delete(v)
    sol = C.select()
    assert (1,0) in sol and len(sol) == 1



