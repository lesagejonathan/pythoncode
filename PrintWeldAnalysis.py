import numpy as np
from numpy.fft import rfft
from scipy.signal import tukey, detrend
import matplotlib.pylab as plt

import _pickle as pickle

a = pickle.load(open('/Users/jlesage/Dropbox/PrintedWeldMaterial/PrintWeldScan-25MHz-HighGain.p','rb'))

A = np.array(a['AScans']).reshape((83,87,2500))

Nwin = int(100*12.5/6.3)

Nring = 100

fband = (3.,10.)

atten = np.zeros((83,87))

attenrate = np.zeros((83,87))

ampmean = np.zeros((83,87))

ampstd = np.zeros((83,87))

ampmax = np.zeros((83,87))


T = np.zeros((83,87))

d = 12.5


for i in range(A.shape[0]):

    for j in range(A.shape[1]):

        indmax = int(np.argmax(np.abs(A[i,j,:])))



        amp = np.abs(A[i,j,indmax+Nring:indmax+2*Nwin-Nring])

        # plt.plot(amp)
        #
        # plt.show()

        ampmax[i,j] = np.amax(amp)

        ampmean[i,j] = np.mean(amp)

        ampstd[i,j] = np.std(amp)


        ampmax[i,j] = np.max(np.abs(A[i,j,indmax+Nring:]))

        y = detrend(A[i,j,int(indmax+Nwin):int(indmax+3*Nwin)])

        x = detrend(A[i,j,int(indmax-Nwin):int(indmax+Nwin)])

        indbw = Nwin + np.argmax(np.abs(y))

        T[i,j] = int(round(indbw/100.))

        Y = rfft(y, int(20*Nwin))

        X = rfft(x, int(20*Nwin))

        f = np.linspace(0.,50.,len(X))


        X = X[(f>=fband[0])&(f<=fband[1])]

        Y = Y[(f>=fband[0])&(f<=fband[1])]

        f = f[(f>=fband[0])&(f<=fband[1])]

        G = -np.log(np.abs(Y)/np.abs(X))



        attenrate[i,j] = np.polyfit(f.flatten(), G.flatten(), 1)[0]

        atten[i,j] = np.mean(G)


plt.plot(x)
plt.plot(y)

plt.show()

cavg = np.mean(2*d/(T.ravel()))

print(cavg)

D = cavg*T

atten = atten/D

attenrate = attenrate/D

pickle.dump({'MaxAmplitude':ampmax, 'MeanAmplitude':ampmean, 'AmplitudeDeviation':ampstd, 'Attenuation':atten, 'AttenuationRate':attenrate}, open('/Users/jlesage/Dropbox/PrintedWeldMaterial/ComputedFeatures.p','wb'))
