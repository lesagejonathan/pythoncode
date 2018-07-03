import numpy as np
from scipy.signal import hilbert
import _pickle
import os
import matplotlib.pylab as plt

pth = '/Users/jlesage/Dropbox/Eclipse/ANSFeederTubeProject/TVGs/'

# ang = np.linspace(40.,70.,32)*np.pi/180.

ang = np.linspace(40.,70.,4)*np.pi/180.


f = os.listdir(pth)

f = [ff for ff in f if ff.endswith('.txt')]

c = 3.279

cw = 2.33

angw = np.arcsin((cw/3.279)*np.sin(ang))


thick = {'2':5.5472, '2p5': 7.0104, '3': 7.62 , '3p5': 8.0772}
Nt = {'2':1207, '2p5':1836, '3':2131, '3p5':2414}

Aref = {'2':0.22319, '2p5':0.217244, '3': 0.255863,'3p5':0.204192}

l = {'2':81, '2p5': 66, '3': 71, '3p5': 76}

H = 2.52

D = {}

for ff in f:

    k = ff.split('_')[1]

    print(k)

    nt = Nt[k]

    a = np.genfromtxt(pth+ff, delimiter=';', skip_header=11)[:,l[k]+2:-1]


    a = 100.*np.amax(np.abs(hilbert(a,axis=1)).reshape((4,nt,l[k])),axis=2)/Aref[k]


    A = np.zeros((32,3,2))

    d = thick[k]


    for i in range(len(ang)):

        Tw = 2*(H/(cw*np.cos(angw[i])))

        aa = a[i,int(round(Tw*25.))::]

        i1 = (int(np.round(2*(1.4*d)/(c*np.cos(ang[i]))*25)), int(np.round(2*(1.6*d)/(c*np.cos(ang[i]))*25)))

        i2 = (int(np.round(2*(2.4*d)/(c*np.cos(ang[i]))*25)), int(np.round(2*(2.6*d)/(c*np.cos(ang[i]))*25)))

        i3 = (int(np.round(2*(3.4*d)/(c*np.cos(ang[i]))*25)), int(np.round(2*(3.6*d)/(c*np.cos(ang[i]))*25)))

        # print(aa.shape)
        # print(i1[1])

        A[i,0,0] = (np.argmax(aa[i1[0]:i1[1]]) + i1[0])/25.

        A[i,0,1] = np.amax(aa[i1[0]:i1[1]])


        A[i,1,0] = (np.argmax(aa[i2[0]:i2[1]]) + i2[0])/25.

        A[i,1,1] = np.amax(aa[i2[0]:i2[1]])


        A[i,2,0] = (np.argmax(aa[i3[0]:i3[1]]) + i3[0])/25.

        A[i,2,1] = np.amax(aa[i3[0]:i3[1]])






    D[ff.split('_')[1]] = A

print(A[0,:,:])
print(A[1,:,:])
print(A[2,:,:])
print(A[3,:,:])

d = _pickle.dump(D,open(pth+'TVGs.p', 'wb'))


# d = _pickle.load(open(pth+'TVGs.p', 'rb'))
#
# k = list(d.keys())
#
# l = {'2': '2"', '2p5':'2.5"', '3': '3"', '3p5':'3.5"'}
#
# ang = np.linspace(40.,70.,32)
#
# for i in range(len(ang)):
#
#     ll = []
#
#     for kk in k:
#
#         plt.plot(d[kk][i,:,0].flatten(),d[kk][i,:,1].flatten(),'-o')
#
#         ll.append(l[kk])
#
#     plt.xlabel('Time (Microseconds)')
#     plt.ylabel('Amplitude')
#
#     plt.ylim((0,100))
#
#     plt.title(str(int(round(ang[i])))+' Degree VPA')
#
#     plt.legend(ll, loc='best')
#
#     plt.savefig(pth+str(int(round(ang[i])))+'.png',dpi=300)
#
#     plt.close()
