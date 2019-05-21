"virtual node isolated forest functions"
__author__ = 'Matias Carrasco Kind && Qi Xing'
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.stats import multivariate_normal
import random as rn
#import iso
import eif_vnif as iso
#import vnif_idiq as iso
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 2000
score = []
ntrees=50
sample = 256
for n in range(0,2):
    x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
    #x[0]=3
    #y[0]=3
    X=np.array([x,y]).T
    CT=[]
    S = np.zeros(Nobjs)
    c = iso.c_factor(len(X))
    for i in range(ntrees):
        ix = rn.sample(range(Nobjs),sample)
        X_p = X[ix]
        limit = int(np.ceil(np.log2(sample)))
        C=iso.iTree(X_p,0,limit)
        CT.append(C)
    for m in range(Nobjs):
        h_temp = 0
        for j in range(ntrees):
            h_temp += iso.PathFactor(X[m]+m,CT[j]).path*1.0
        Eh = h_temp/ntrees
        S[m] = 2.0**(-Eh/c)
    score.append(S)
    print(n)
score_pd = DataFrame(score)
score_pd.to_csv('NM_score.csv')

#ss=np.argsort(S)
#plt.plot(x,y,'bo')
#plt.plot(x[ss[-10:]],y[ss[-10:]],'ro')

#plt.figure()

#sv1 = []
#sv2 = []
#sv3 = []

#for j in range(ntrees):
    #sv1.append(2**(-iso.PathFactor(X[ss[0]],CT[j]).path*1.0/c))
    #sv2.append(2**(-iso.PathFactor(X[ss[Nobjs/2]],CT[j]).path*1.0/c))
    #sv3.append(2**(-iso.PathFactor(X[ss[-1]],CT[j]).path*1.0/c))

#plt.plot(sv1,label='normal')
#plt.plot(sv2, label='semi')
#plt.plot(sv3, label='outlier')
#plt.legend(loc=0)

#plt.show()
