import pickle
import os
from numpy import *
from matplotlib.pylab import *
from functools import reduce
from sklearn.mixture import GaussianMixture


pth = '/Users/jlesage/Dropbox/Eclipse/MP-CustomTFM/ToCompare/'

d = os.listdir(pth)

d = [dd for dd in d if dd.endswith('.p')]

A0 = []
A1 = []

# d = [d[0]]

I0 = []
I1 = []

for dd in d:

    w = pickle.load(open(pth+'/'+dd,'rb'))

    I0 = array([sum(abs(w['Images'][0][i][:,:,0]),axis=0) for i in range(len(w['Images'][0]))]).transpose()

    I1 = array([sum(abs(w['Images'][1][i][:,:,0]),axis=0) for i in range(len(w['Images'][1]))]).transpose()

    #
    # I0.append(I0[0:7,:])
    # I1.append(I1[0:7,:])

    # imshow(I0,aspect=0.08)
    #
    # imshow(I1,aspect=0.08)

    A0.append(sum(I0,axis=0))

    A1.append(sum(I1,axis=0))


X0 = reduce(lambda x,y: concatenate((y,x)), [A for A in A0])
X1 = reduce(lambda x,y: concatenate((y,x)), [A for A in A1])


X = hstack((X0.reshape(-1,1),X1.reshape(-1,1)))

gm = GaussianMixture(n_components=2,max_iter=200,n_init=100,init_params='random',random_state=51)

gm.fit(X)

y = array(gm.predict(X))

# print(y)

scatter(array(X)[y==0,0],array(X)[y==0,1],color='blue',alpha=0.1)

scatter(array(X)[y==1,0],array(X)[y==1,1],color='red',alpha=0.1)


savefig(pth+'MP-CustomTFM-Scatter.png',dpi=250)

close('all')

c = {}

for i in range(len(A0)):

    XX = hstack((A0[i].reshape(-1,1),A1[i].reshape(-1,1)))

    y = gm.predict(XX)

    plot(y,'b-o')
    ylim((-0.1,1.1))

    savefig('/Users/jlesage/Dropbox/Eclipse/PresentationImages/'+d[i]+'-calls.png')

    close('all')

#     ec = 100*sum(y)/len(y)
#
#     c[d[i].split('-')[2]] = {'PercentCall':ec, 'HitMiss':[int(any(y[10:30])),int(any(y[50:70]))]}
#
#
# pickle.dump(c,open(pth+'calls.p' ,'wb'))

    # plot(gm.predict(XX),'-o')
    #
    # print(d[i])
    #
    # ylim((-0.1,1.1))
    #
    # show()
