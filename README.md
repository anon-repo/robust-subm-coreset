# robust-subm-coreset-code

## Project structure 

Package `robustcore`:
* `coreset.py`: different coresets
* `base.py`: abstract class
* `matroid.py`: classes for a feasible solution
* `sieve.py`: class for managing thresholds
* `function.py`: subm functions for different types of data
* `adversary.py`: classes for adversaries

In addition,
* `data_scripts/` includes scripts for data pre-processing. 
* `tests/` includes test cases for classes.


## Example

```python
import numpy as np
from functools import partial
from robustcore.base import Data
from robustcore.matroid import Matroid, UnifMatroid, ParMatroid, pMatroid
from robustcore.coreset import ExchangeCoreset, GreedyBackupCoreset


# Example 1: offline for a uniform matroid
n = 11
V = np.arange(n)
f = lambda S: sum(S)
d = 2
k = 2
dat = Data(V)

C = GreedyBackupCoreset(f, d, UnifMatroid(k), is_unifM=True, eps=0.5)
for v in dat:
    C.summarize(v)
C.summarize(None)
# assert C.select().sol == {9,10}
C.delete(9)
assert C.select().sol == {10,8}


# Example 2: streaming for 2-matroid
V = [(0,0),(0,1),(1,0),(2,1),(2,0)] # (i, its possible group)
m1 = partial(Matroid, r=(2,1), g=lambda x: x[1])
m2 = partial(Matroid, r=(1,1,1), g=lambda x: x[0])
M = partial(pMatroid, matroids=[m1,m2])
w = dict(zip(V, [1, 1.5, 1.1, 1, 5]))
f = lambda S, **kwargs: sum([w[i] for i in S]) / sum(w.values())
d = 2
k = 2

C = ExchangeCoreset(f, d, M, use_sampling=True, eps=0.5)
for v in V:
    C.summarize(v)
C.summarize(None)
# assert C.select().sol == {(0, 1), (1, 0), (2, 0)}
C.delete((2,0))
assert C.select().sol == {(0, 1), (1, 0)}
```

## Tests

```
py.test -vv -s
```

## License

MIT license


[//]: # (Comment)
[//]: # (https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city, https://grouplens.org/datasets/movielens/)


