#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[14]:


fname = './offline_datasets/uber-raw-data-apr14.csv'
df = pd.read_csv(fname)


# In[5]:


df.head()


# In[15]:


df['Base'] = df['Base'].astype('category').cat.codes
df['Base']


# In[20]:


len(df)


# In[21]:


n = 50000
df = df.sample(n=50000, random_state=42)


# In[22]:


len(df)


# In[27]:


nsam = 500
sam = np.random.choice(n, size=nsam, replace=False)


# In[29]:


sam.shape


# In[30]:


X = df['Lat'].values
Y = df['Lon'].values
P = [np.array([X[i], Y[i]]) for i in range(n)]
C = df['Base'].values


# In[35]:


def L1(a,b):
    return sum(np.abs(a-b))


dists = dict()
n = len(X)
V = [(i,C[i]) for i in range(n)]
for i,v in enumerate(V):
    if i % 1000 == 0:
        print(i)
    d = list()
    for j in sam:
        d.append(L1(P[i],P[j]))
    dists[v] = d


# In[39]:


with open('./datasets/uber.pkl', 'wb') as fin:
    pickle.dump((V, dists), fin)


