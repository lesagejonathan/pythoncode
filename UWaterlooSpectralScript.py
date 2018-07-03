import numpy as np
from numpy.linalg import norm
from numpy.fft import rfft
import matplotlib.pylab as plt
from scipy.io import wavfile
from os import listdir
from _pickle import dump
from scipy.signal import detrend
from Signal import ToApproxZeroPhase
from misc import FFTLengthPower2

pth = '/Volumes/STORAGE/UWaterlooStudy/Spectra/'

d = listdir(pth)

d = [dd for dd in d if dd.endswith('.wav') and dd[0] is 'S']

print(d)

NFFT = FFTLengthPower2(150000)

D = {'Frequency':np.linspace(0.,44.1/2,int(np.floor(NFFT/2)+1)),'A':{}, 'B':{}, 'C':{}}

for dd in d:

    print(pth+dd)

    w = wavfile.read(pth+dd)[1]

    plt.plot(w)

    gates = plt.ginput(0,timeout=0)

    plt.close()

    W = None

    for n in range(int(len(gates)/2)):

        WW = rfft(detrend(w[int(gates[2*n][0]):int(gates[2*n+1][0])]), NFFT)
        WW = WW/norm(WW)

        WW = np.sqrt(np.real(WW*np.conj(WW))).reshape(-1,1)

        if W is None:

            W = WW

        else:

            W = np.hstack((W,WW))

    fln = dd.split('_')

    D[fln[1]][fln[2]] = np.mean(W,axis=1).flatten()

dump(D,open(pth+'Spectra.p','wb'))
