import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys
#sys.path.append('/home/g/u/guaz/Documents/robust-subm-coreset-code/robustcore')
#sys.path.append('../')
sys.path.append('./')

from robustcore.util import gt


def parse_file_name(name):
  # [age]_[gender]_[race]_[date&time].jpg
  pieces = name.split('_')
  if len(pieces) != 4:
    return None
  age = int(pieces[0])
  gender = int(pieces[1])
  race = int(pieces[2])
  return age, gender, race


def L1(a,b):
    return sum(np.abs(a-b))


path = './offline_datasets/UTKFace/'

files = [f for f in listdir(path) if isfile(join(path, f))]
features = []
sli = np.arange(0,200,10) # subsampling
metas = []

for f in files:
    pf = join(path, f)
    meta = parse_file_name(f)
    if meta is None:
      print(f)
      continue
    metas.append(meta)
    image = Image.open(pf).convert('L') # into grey scale
    row = np.asarray(image, dtype=np.int64) # uint8 o.w.
    row = row[sli][:,sli].flatten()
    features.append(row)


n = len(features)
nsam = 500
sam = np.random.choice(n, size=nsam, replace=False)
ages, genders, races = list(zip(*metas))
V = [(i,races[i]) for i in range(n)]

dists = dict()
for i,(v,feat) in enumerate(zip(V,features)):
    if i % 1000 == 0:
        print(i)
    dists_i = list()
    for j in range(nsam):
        d = L1(feat, features[j])
        dists_i.append(d)
    dists[v] = dists_i

with open('./datasets/face.pkl', 'wb') as fin:
    pickle.dump((V, dists), fin)
