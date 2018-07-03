import numpy as np
from sklearn import covariance
from sklearn.preprocessing import scale
import matplotlib.pylab as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

d = np.loadtxt('/Users/jlesage/Dropbox/Eclipse/IndicationReport.csv',delimiter=',')

# F = d[:,(1,2,9)]

# F = d[:,9].reshape(-1,1)

X = d[:,(3,4,5,6)]

Xkeep = X[:,3]>-55.

# print(Xkeep)
# print(Xkeep.shape)
#
# print(X.shape)

X = X[Xkeep,:]

# print(X.shape)

# gm = GaussianMixture(n_components=2,random_state=21)

# gm.fit(X)
#
# y = gm.predict_proba(X)

c = KMeans(n_clusters=2,random_state=42)

c.fit(X)

y = c.predict(X)


print(c.cluster_centers_)

print(y)
# print(c.transform(X))

# plt.plot(np.min(np.array(c.transform(X)),axis=1))
#
# plt.show()

# print(y)


# g = [(5.,-23.),(21.,-50.),(4.,-76.)]
#
# FG = np.zeros((G.shape[0],1))

# for i in range(G.shape[0]):
#
#     FG[i,0] = min([np.sqrt(((G[i][0]+G[i][1] - gg[0])*0.5)**2 + ((G[i][2]+G[i][3])*0.5 - gg[1])**2) for gg in g])
#
#     # FG[i,1] = np.arctan2(G[i][3]-G[i][2],G[i][1]-G[i][0])
#
# F = scale(np.hstack((F,FG)))
#
# O = covariance.EllipticEnvelope(assume_centered=True, random_state=21)
#
# O.fit(F)
#
# y = np.array(O.predict(F))
#
# L = O.mahalanobis(F)
#
# # plt.hist(L,bins=50)
# #
# # plt.show()
#
# print(np.sort(L))
# print(np.argsort(L))
#
# savetxt('/Users/jlesage/Dropbox'())
# print(O.mahalanobis(FG))
#
# print(np.where(y==-1)[0]+1)
