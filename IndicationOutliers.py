import numpy as np
from sklearn import covariance
from sklearn.preprocessing import scale
import matplotlib.pylab as plt

d = np.loadtxt('/Users/jlesage/Dropbox/Eclipse/IndicationReport.csv',delimiter=',')

# F = d[:,(1,2,9)]

F = d[:,9].reshape(-1,1)

G = d[:,(3,4,5,6)]

Gkeep = G[:,3]>-55.

g = [(5.,-23.),(21.,-50.),(4.,-76.)]

FG = np.zeros((G.shape[0],1))

for i in range(G.shape[0]):

    FG[i,0] = min([np.sqrt(((G[i][0]+G[i][1] - gg[0])*0.5)**2 + ((G[i][2]+G[i][3])*0.5 - gg[1])**2) for gg in g])

    # FG[i,1] = np.arctan2(G[i][3]-G[i][2],G[i][1]-G[i][0])

F = np.hstack((F,FG))

F = scale(F[Gkeep,:])

# F = scale(np.hstack((F,FG)))

O = covariance.EllipticEnvelope(assume_centered=True, random_state=21)

O.fit(F)

y = np.array(O.predict(F))

L = O.mahalanobis(F)

# plt.hist(L,bins=50)
#
# plt.show()

print(np.sort(L))
print(np.argsort(L))

# print(O.mahalanobis(FG))
#
# print(np.where(y==-1)[0]+1)
