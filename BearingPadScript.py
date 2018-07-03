import numpy as np
import matplotlib.pylab as plt
from os import listdir
from scipy.signal import hilbert

pth = '/Users/jlesage/Dropbox/Eclipse/CompositeBearing/'

d = listdir(pth)

d = [dd for dd in d if dd.endswith('.txt')]

D = {}

for dd in d:

    D[dd.strip('.txt')] = np.fromfile(pth+dd,dtype='float32')[0:640*49].reshape((49,640))


fs1 = 1./(19.2/640)


xref = np.abs(hilbert(D['BearingRef'][20,:]))

plt.plot(xref)

gate = plt.ginput(4,timeout=0)

plt.close()


gate = ((int(gate[0][0]), int(gate[1][0])), (int(gate[2][0]), int(gate[3][0])))


t1 = (np.argmax(np.abs(hilbert(D['BearingRef'],axis=1))[:,gate[0][0]:gate[0][1]], axis=1) + gate[0][0])/fs1

t2 = (np.argmax(np.abs(hilbert(D['BearingRef'],axis=1))[:,gate[1][0]:gate[1][1]], axis=1) + gate[1][0])/fs1

c = 2*5.9/(t2-t1)

plt.scatter(np.linspace(1,len(c), len(c)), c)
plt.plot(np.linspace(1,len(c),len(c)), np.mean(c)*np.ones(len(c)))
plt.plot(np.linspace(1,len(c),len(c)), (np.mean(c)-2*np.std(c))*np.ones(len(c)), '--b')
plt.plot(np.linspace(1,len(c),len(c)), (np.mean(c)+2*np.std(c))*np.ones(len(c)), '--b')

plt.ylabel('Velocity (mm/$\mu$s)')
plt.xlabel('VPA')

print(2*np.std(c))

plt.ylim((2.,2.7))

plt.xlim((1,50))

plt.legend(['Mean Velocity = 2.27 mm/$\mu$s', '95% Confidence Band = $\pm$ 0.03 mm/$\mu$s'],loc='best')
plt.savefig('/Users/jlesage/Dropbox/Eclipse/CompositeBearing/VelocityPlot.png', dpi=200)




# c = np.mean(2*5.9/(t2-t1))

# xa = [np.vstack((np.abs(hilbert(D['Bearing1'])), np.abs(hilbert(D['Bearing2']))))]
#
# xa.append(np.vstack((np.abs(hilbert(D['Bearing3'])), np.abs(hilbert(D['Bearing4'])))))
#
# xa.append(np.vstack((np.abs(hilbert(D['Bearing5'])), np.abs(hilbert(D['Bearing6'])))))
#
# xa.append(np.vstack((np.abs(hilbert(D['Bearing7'])), np.abs(hilbert(D['Bearing8'])))))
#
# xa.append(np.vstack((np.abs(hilbert(D['Bearing9'])), np.abs(hilbert(D['Bearing10'])))))
#
#
#
# fs = 50.
#
# l0 = [5.9,5.37,5.1,4.91,2.85]
#
# L = None
#
#
# for i in range(len(xa)):
#
#     plt.plot(xa[i][30,:])
#
#     gate = plt.ginput(4,timeout=0)
#
#     plt.close()
#
#
#     gate = ((int(gate[0][0]), int(gate[1][0])), (int(gate[2][0]), int(gate[3][0])))
#
#
#     t1 = (np.argmax(xa[i][:,gate[0][0]:gate[0][1]], axis=1) + gate[0][0])/fs
#
#     t2 = (np.argmax(xa[i][:,gate[1][0]:gate[1][1]], axis=1) + gate[1][0])/fs
#
#     l = c*(t2-t1)*0.5
#
#     print(l.shape)
#
#     if L is None:
#
#         L = np.hstack((l.reshape(-1,1),l0[i]*np.ones((len(l),1))))
#
#     else:
#
#         L = np.vstack((L,np.hstack((l.reshape(-1,1),l0[i]*np.ones((len(l),1))))))
#
#
# L = L[1::,:]
#
# np.savetxt(pth+'ThicknessMeasurements.csv', L, delimiter=',')
