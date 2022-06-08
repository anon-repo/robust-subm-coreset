import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


rng = np.random.default_rng(12345)
fdata = '/NOBACKUP/guaz/msr-datasets/train_triplets.txt'


def sample_users(nsz, like_thr=1):
    users = set()
    with open(fdata, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            user, song, count = line.split()
            users.add(user)

    users = list(users)
    print('#user: ', len(users))

    idx = rng.choice(range(len(users)), size=nsz, replace=False)
    u2id = dict([(users[j],i) for i,j in enumerate(idx)])
    #u2id = dict([(u,i) for i,u in enumerate(users)])

    songs = dict() # every el is a subset of users
    s2id = dict()
    with open(fdata, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            user, song, count = line.split()
            count = int(count)
            if user in u2id and count > like_thr:
                uid = u2id[user]
                if song not in s2id:
                    s2id[song] = len(s2id)
                sid = s2id[song]
                if sid in songs:
                    songs[sid].add(uid)
                else:
                    songs[sid] = {uid}

    V = [v for v in s2id.values()]
    U = [v for v in u2id.values()]
    print('#song: ', len(V), '#user', len(U))
    pickle.dump((V,U,songs), open(f'./datasets/music.pkl', 'wb'))


if __name__ == '__main__':
    sample_users(nsz=50000)
