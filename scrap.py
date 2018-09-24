import numpy as np
from numpy.fft import rfft, irfft,ifftshift


def TargetConvolution(l,FWHM,dx):

    N = int(10*FWHM/dx)

    c = FWHM/(2.*np.sqrt(2.*np.log(2.)))

    s = np.linspace(0.,1/(2*dx),N)


    H = np.sinc(l*s)*l

    G = np.exp(-2*(c**np.pi*s)**2)*np.sqrt(np.pi*2)*c

    f = ifftshift(irfft(H*G))

    f = f/(np.amax(np.abs(f)))

    ind = np.where(f>=0.5)[0]

    FWHMout = len(ind)*dx

    return f,FWHMout


import _pickle as pickle
import FMC
import os
import matplotlib.pylab as plt
from scipy.interpolate import griddata

pth = '/Users/jlesage/Dropbox/Eclipse/FMCCalibration/NavShipAveraged/'

Dir = os.listdir(pth)

Dir = [dd for dd in Dir if dd.endswith('.p')]


b = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/FMCCalibration/NavShipAveragedProbeDelay.p','rb'))


d = FMC.EstimateProbeDelays(b['AScans'][0], 25., 0.6, 31.97)[0]

Amp = []
XX = []
YY = []


# for dd in Dir:

a = pickle.load(open(pth+Dir[0],'rb'))

aref = a['AScans'][1::2]

atest = a['AScans'][0::2]

a = [aref,atest]

for aa in a:

    F = FMC.LinearCapture(25.,aa,0.6,64,d)

    F.ProcessScans(50,70)

    X = np.arange(-21.,21.,1.)

    Y = np.arange(5.,40.,1.)

    F.GetContactDelays(X,Y,5.92)

    I = [F.ApplyTFM(i) for i in range(len(F.AScans))]

    x = np.zeros(len(I))
    y = np.zeros(len(I))
    amp = np.zeros(len(I))

    for i in range(len(I)):

        ind = np.unravel_index(np.argmax(np.abs(I[i])),I[i].shape)

        F.GetContactDelays(np.arange(X[ind[1]]-5.,X[ind[1]]+5.,0.1),np.arange(Y[ind[0]]-5.,Y[ind[0]]+5.,0.1),5.92)

        II = F.ApplyTFM(i)

        iind = np.unravel_index(np.argmax(np.abs(II)), II.shape)

        x[i] = F.xRange[iind[1]]
        y[i] = F.yRange[iind[0]]

        amp[i] = np.abs(II[iind[0],iind[1]])

    Amp.append(amp)
    XX.append(x)
    YY.append(y)


pickle.dump({'Amplitude':Amp, 'x':XX, 'y':YY}, open(pth+'NavshipAveragedAmplitudes.p','wb'))

# pickle.dump({'Amplitude':amp, 'x':x, 'y':y}, open(pth+'NavshipAveragedAmplitudes.p','wb'))


#
#
for i in range(len(Amp)):

    cond = (abs(XX[i]>=-20.))&(abs(YY[i]-0.5*25.4)<2.)&(abs(YY[i]-0.75*25.4)<2.)&(abs(YY[i]-1*25.4)<2.)&(abs(YY[i]-1.25*25.4)<2.)&(abs(XX[i]<=20.))

    XX[i] = XX[i][cond]
    XX[i] = YY[i][cond]
    Amp[i] = Amp[i][cond]


    G = 80./Amp[0]

    xi = XX[0]
    yi = YY[0]

y = np.array([0.5, 0.75, 1., 1.25])*25.4

for i in range(1, len(Amp)):

    for j in range(len(y)):

        cond = abs((YY[i]-y[j])<2.)

        a = Amp[i][cond]

        xx = XX[i][cond]

        yy = YY[i][cond]

        g = G[cond]

        xxi = xi[i][cond]
        yyi = yi[i][cond]

        aa = griddata((xxi,yyi), g, (xx,yy), method='nearest')*a

        plt.plot(xx, aa, '.-')
        plt.plot(xx, np.mean(aa)*np.ones(len(aa)), 'r')

        plt.plot(xx, 75*np.ones(len(aa)), 'g--')
        plt.plot(xx, 85*np.ones(len(aa)), 'g--')

        plt.xlabel('Hole Position with Respect to Centre Aperture (mm)')

        plt.ylabel('Percent Screen Height (%)')

        plt.savefig('/Users/jlesage/Dropbox/Eclipse/FMCCaibration/Scan'+str(i)+'Depth'+str(np.round(y[j]))+'.png', dpi=450)

        plt.close()
