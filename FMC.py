from functools import reduce
import numpy as np
from numpy.fft import rfft, ifft, fftn, ifftn, fftshift, fft
from pathos.multiprocessing import ProcessingPool

def NextPow2(x):
    return int(2**int(np.ceil(np.log2(x))))

# def EstimateProbeDelays(Scans,fsamp,p,h,c=5.92):
#
#     M = Scans.shape[0]
#     N = Scans.shape[1]
#
#     x = np.abs(hilbert(Scans,axis=2))
#
#     W = int(np.round(fsamp*0.25*h/c))
#
#     Delays = np.zeros((M,N))
#
#     Amps = np.zeros((M,N))
#
#     for m in range(M):
#
#         for n in range(N):
#
#             T = int(np.round(fsamp*(2*np.sqrt((0.5*(n-m)*p)**2 + h**2)/c)))
#
#             indmax = np.argmax(np.abs(x[m,n,T-W:T+W]))+T-W - T
#
#             Amps[m,n] = np.abs(x[m,n,indmax])
#
#             Delays[m,n] = indmax/fsamp
#
#     return Delays,Amps


class LinearCapture:

    def __init__(self, fs, scans, p, N, probedelays=None, elementamps = None):

        import copy

        self.SamplingFrequency = fs
        self.Pitch = p
        # self.Velocity = c
        self.NumberOfElements = N

        if probedelays is None:

            self.ProbeDelays = np.zeros((N, N))

        else:

            self.ProbeDelays = probedelays

        if elementamps is None:

            self.ElementAmplitudes = np.ones((N, N))

        else:

            self.ElementAmplitudes = elementamps

        self.AScans = copy.deepcopy(scans)

        # if wedgeparams is not None:
        #
        #     self.WedgeParameters = wedgeparams.copy()

    def ProcessScans(self, zeropoints=0, bp=10):

        from scipy.signal import detrend,hilbert

        L = self.AScans[0].shape[2]

        # Lpad = NextPow2(np.round((L + np.amax(self.ProbeDelays)*self.SamplingFrequency - 1)))

        Lpad = int(np.round((L + np.amax(self.ProbeDelays)*self.SamplingFrequency - 1)))

        f = np.linspace(0.,self.SamplingFrequency/2,np.floor(Lpad/2)+1)

        f = f.reshape((1,1,len(f)))

        D = np.exp(2j*np.pi*np.repeat(self.ProbeDelays[:,:,np.newaxis],f.shape[2],2)*f)

        A = np.repeat(self.ElementAmplitudes[:,:,np.newaxis], f.shape[2], 2).astype(np.complex64)

        for i in range(len(self.AScans)):

            if zeropoints != 0:
                self.AScans[i][:,:,0:zeropoints] = 0.0

            self.AScans[i] = detrend(self.AScans[i],bp=list(np.linspace(0,L-1,bp).astype(int)))

            X = rfft(self.AScans[i],n=Lpad)

            self.AScans[i] = self.AScans[i].astype(np.complex64)

            self.AScans[i] = 2*ifft((X/A)*D,n=Lpad)[:,:,0:L]


    def PlaneWaveSweep(self, ScanIndex, Angles, c):

        # from numpy import sum as asum

        d = self.Pitch * self.NumberOfElements

        d = np.linspace(-d / 2, d / 2, self.NumberOfElements)

        L = self.AScans[ScanIndex].shape[2]

        if type(Angles) is tuple:

            Angles = (Angles[0] * np.pi / 180., Angles[1] * np.pi / 180.)

            T = np.abs(np.sin(np.repeat((Angles[0].reshape(-1, 1)*Angles[1].reshape(1,-1))[:,:,np.newaxis],len(d),2))*d.reshape((1,1,len(d)))/c).flatten()


        else:

            Angles = Angles * np.pi / 180.

            T = np.abs(np.sin(np.repeat((Angles.reshape(-1, 1)*Angles.reshape(1,-1))[:,:,np.newaxis],len(d),2))*d.reshape((1,1,len(d)))/c).flatten()




        Npad = int(np.round(self.SamplingFrequency*np.max(T)))


        Lpad = L+Npad -1
        # X = rfft(np.concatenate((np.zeros((self.NumberOfElements, self.NumberOfElements, Npadstart)),
        #                       np.real(self.AScans[ScanIndex]), np.zeros((self.NumberOfElements,
        #                       self.NumberOfElements, Npadend))), axis=2), axis=2)

        X = rfft(np.real(self.AScans[ScanIndex]),Lpad)

        f = np.linspace(0, self.SamplingFrequency / 2, X.shape[2])

        def PlaneWaveFocus(angles):

            T = np.meshgrid(f, d * np.sin(angles[1]) / c)

            XX = np.sum(X * np.exp(-2j * np.pi * T[0] * T[1]), axis=1, keepdims=False)

            T = np.meshgrid(f, d * np.sin(angles[0]) / c)

            XX = np.sum(XX * np.exp(-2j * np.pi * T[0] * T[1]), axis=0, keepdims=False)

            x = ifft(XX, n=Lpad)

            return x[0:L]

        if type(Angles) is tuple:

            return np.array([[PlaneWaveFocus((ta, ra))
                         for ra in Angles[1]] for ta in Angles[0]])

        else:

            return np.array([PlaneWaveFocus((ta, ta)) for ta in Angles])

    def GetContactDelays(self, xrng, yrng, c):

        # if c is None:
        #
        #     c = self.Velocity

        self.Delays = [[[np.sqrt((x - n * self.Pitch)**2 + y**2) / c for y in yrng] for x in xrng] for n in range(self.NumberOfElements)]

        self.xRange = xrng.copy()

        self.yRange = yrng.copy()

    # def GetWedgeDelays(self, xrng, yrng):
    #
    #     from scipy.optimize import minimize
    #
    #     p = self.Pitch
    #     h = self.WedgeParameters['Height']
    #
    #     cw = self.WedgeParameters['Velocity']
    #
    #     cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
    #     sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)
    #
    #     c = self.Velocity
    #
    #     def f(x,X,Y,n):
    #         return np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2) / cw + np.sqrt(Y**2 + (X - x)**2) / c
    #
    #     def J(x,X,Y,n):
    #         return -(cphi * n * p - x) / (cw * np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2)) - \
    #                 (X - x) / (c * np.sqrt(Y**2 + (X - x)**2))
    #
    #     self.Delays = [[[minimize(f,x0=0.5 * np.abs(x - n * self.Pitch * cphi),
    #                     args=(x,y,n),method='BFGS',jac=J).fun for y in yrng] for x in xrng]
    #                    for n in range(self.NumberOfElements)]
    #
    #     self.xRange = xrng.copy()
    #     self.yRange = yrng.copy()

    def KeepElements(self, Elements):

        for i in range(len(self.AScans)):

            self.AScans[i] = np.take(np.take(self.AScans[i], Elements, axis=0), Elements, axis=1)

        self.ProbeDelays = np.take(np.take(self.ProbeDelays, Elements, axis=0), Elements, axis=1)

        self.NumberOfElements = len(Elements)

    def FitInterfaceCurve(self, ScanIndex, hrng, c, smoothparams=(0.1,0.1)):

        from scipy.interpolate import interp1d
        # from scipy.signal import guassian, convolve
        from skimage.filters import threshold_li, gaussian
        from misc import DiffCentral

        xrng = np.linspace(0, self.NumberOfElements*self.Pitch, self.NumberOfElements)

        self.GetContactDelays(xrng, hrng, c)

        I = gaussian(np.abs(self.ApplyTFM(ScanIndex)),smoothparams)

        dh = hrng[1] - hrng[0]

        hgrid = hrng[0] + dh*np.argmax(I, axis = 0)

        dhdx = np.abs(DiffCentral(hgrid))

        indkeep = np.where(dhdx<threshold_li(dhdx))

        hgrid = hgrid[indkeep]

        xrng = xrng[indkeep]


        # hgrid = np.zeros((options['NPeaks'], I.shape[1]))
        #
        # for i in range(I.shape[0]):
        #
        #     indmax,valmax = argrelmax(np.abs(I[:,i]), order=options['MinSpacing'])
        #
        #     indmax = indmax[np.argsort(valmax)[-options['NPeaks']::]]
        #
        #     hgrid[:,i] = hrng[0] + dh*indmax


        h = interp1d(xrng, hgrid, kind='quadratic', bounds_error=False, fill_value=np.nan)

        dhdx = interp1d(xrng[1::],np.diff(h(xrng))/np.diff(xrng), bounds_error=False, fill_value=np.nan)


        return h, dhdx

        # return h

    def FitInterfaceLine(self, ScanIndex, angrng, gate, c):

        """ gate specified in terms of mm in medium with Velocity c """

        angles = np.arange(angrng[0],angrng[1],angrng[2])

        X = np.abs(self.PlaneWaveSweep(ScanIndex, angles, c))[:,int(np.round(2*gate[0]*self.SamplingFrequency/c)):int(np.round(2*gate[1]*self.SamplingFrequency/c))]

        imax = np.unravel_index(X.argmax(),X.shape)

        angmax = -angles[imax[0]]*np.pi/180.

        hmax = gate[0] + imax[1]*c/(2*self.SamplingFrequency)

        h0 = hmax/(np.cos(angmax))

        h = lambda x: np.tan(angmax)*x + h0 - self.Pitch*self.NumberOfElements*np.tan(angmax)/2

        return h

    def GetAdaptiveDelays(self, xrng, yrng, h, cw, cs, AsParallel=False):

    # def GetAdaptiveDelays(self, xrng, yrng, h, dhdx, cw, cs, AsParallel=False):


        from scipy.optimize import minimize_scalar, brentq
        # from scipy.interpolate import interp1d
        # from skimage.filters import threshold_li

        # xrng = np.linspace(0, self.NumberOfElements - 1, self.NumberOfElements) * self.Pitch

        # self.GetContactDelays(xrng, hrng, cw)
        #
        # I = self.ApplyTFM(ScanIndex,filterparams)
        #
        # dh = hrng[1] - hrng[0]
        #
        # hgrid = np.argmax(np.abs(I), axis=0) * dh + hrng[0]
        #
        # hthresh = threshold_li(hgrid)
        #
        # # hgrid = hgrid[hgrid>hthresh]
        #
        # h = interp1d(xrng, hgrid, bounds_error=False)

        def f(x, X, Y, n):
            return np.sqrt((x - n * self.Pitch)**2 + h(x) ** 2) / cw + np.sqrt((X - x)**2 + (Y - h(x))**2) / cs


        def dfdx(x ,X, Y, n):
            return ((x-n*self.Pitch + h(x))**(-1/2.))*(x-n*self.Pitch + h(x)*dhdx(x))/cw - (((X-x)**2 + (Y-h(x))**2)**(-1/2.))*((X-x)**2 + (Y-h(x))*dhdx(x))/cs

        # hmin = np.min(hgrid[hgrid>hthresh])
        #
        # yrng = np.linspace(hmin,hmin+depth[0],int(round(depth[0]/depth[1])))
        #
        # xrng = np.linspace(0,xrng[-1],int(round(xrng[-1]/depth[1])))

        # h0 = h(xrng[0])
        # mh = (h(xrng[-1]) - h0)/(xrng[-1]-xrng[0])
        #
        # def x0(X,Y,n):
        #
        #
        #     m = Y/(X-n*self.Pitch)
        #
        #     if np.isfinite(m):
        #
        #         return (m*n*self.Pitch - mh*xrng[0] + h0 - Y)/(m - mh)
        #
        #     else:
        #
        #         return X
        #


        # def DelayMin(n):
        #
        #     return [[float(minimize(f,x0=x0(x,y,n),args=(x,y,n),method='BFGS',options={'disp': False,
        #         'gtol': 1e-3,'eps': 1e-4,'return_all': False,'maxiter': 50, 'norm': np.inf}).fun)
        #         if y >= h(x) else np.nan for y in yrng] for x in xrng]

        def DelayMin(n):

            return [[float(minimize_scalar(f,bracket=(n*self.Pitch,x),args=(x,y,n),method='Brent',options={'xtol':1e-3,'maxiter':50}).fun)
                    if y >= h(x) else np.nan for y in yrng] for x in xrng]

        # def DelayMin(n):
        #
        #     return [[ float(dfdx(brentq(dfdx, n*self.Pitch,x,args=(x,y,n), maxiter=20, xtol=1e-3), x,y,n))
        #             if y >= h(x) else np.nan for y in yrng] for x in xrng]



        # self.Delays =
        # [[[float(minimize(f,x0=0.5*abs(x-n*self.Pitch),args=(x,y,n),method='BFGS',options={'disp':
        # False, 'gtol': 1e-3, 'eps': 1e-4, 'return_all': False, 'maxiter': 50,
        # 'norm': inf}).fun) if y>=h(x) else nan for y in yrng] for x in xrng]
        # for n in range(self.NumberOfElements)]

        if AsParallel:

            self.Delays = ProcessingPool().map(DelayMin, [n for n in range(self.NumberOfElements)])

        else:

            self.Delays = [ DelayMin(n) for n in range(self.NumberOfElements) ]


        self.xRange = xrng

        self.yRange = yrng

    def FilterByAngle(self, ScanIndex, filtertype, angle, FWHM, c):

        L = self.AScans[ScanIndex].shape[2]

        # Lpad = NextPow2(L)


        X = fftshift(fftn(np.real(self.AScans[ScanIndex]), s=(self.NumberOfElements,L),axes=(0, 2)), axes=(0))

        # X = fftshift(rfft(fft(np.real(self.AScans[ScanIndex]), axis = 0)), axes = (0))

        X = X[:, :, 0:int(np.floor(X.shape[2] / 2) + 1)]

        kx = 2 * np.pi * np.linspace(-1 / (2 * self.Pitch),
                               1 / (2 * self.Pitch), X.shape[0]).reshape(-1, 1)

        w = 2 * np.pi * np.linspace(0., self.SamplingFrequency /
                              2, X.shape[2]).reshape(1, -1)

        th = np.arcsin(c * kx / w)

        th = np.repeat(th.reshape((kx.shape[0],1, w.shape[1])), X.shape[1], axis=1) * 180 / np.pi

        alpha = ((2.35482)**2) / (2 * FWHM**2)

        FilterFunction = {'Band': np.exp(-alpha * (th - angle)**2), 'Notch': 1. - np.exp(-alpha * (th - angle)**2)}

        H = np.nan_to_num(FilterFunction[filtertype]).astype(type(X[0,0,0]))

        X = H * X

        X = fftshift(X, axes=(0))

        return 2 * ifftn(X, s=(X.shape[0], L), axes=(0, 2))

    def ApplyTFM(self, ScanIndex, FilterParams=None, AsParallel=False):

        IX = len(self.Delays[0])
        IY = len(self.Delays[0][0])

        if FilterParams is None:

            a = self.AScans[ScanIndex]

        else:

            a = self.FilterByAngle(ScanIndex, FilterParams[0], FilterParams[1], FilterParams[2], FilterParams[3])

        # L = self.AScans[ScanIndex].shape[2]

        L = a.shape[2]

        Nd = len(self.Delays)

        # delaytype = (len(self.Delays) == len(self.AScans))

        def PointFocus(pt):

            # Nd = len(self.Delays)

            ix = pt[0]
            iy = pt[1]

            # return reduce(lambda x,y: x+y,
            # (A[m,n,int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy]
            # + self.ProbeDelays[m,n])*self.SamplingFrequency))] if
            # ((type(self.Delays[n][ix][iy]) is float or float64) and
            # type(self.Delays[m][ix][iy]) is float or float64) else 0.+0j for
            # n in range(Nd) for m in range(Nd)))

            return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+
                            self.Delays[n][ix][iy]) * self.SamplingFrequency))]
                            if (np.isfinite(self.Delays[m][ix][iy]) and np.isfinite(self.Delays[n][ix][iy])
                            and int(round((self.Delays[m][ix][iy]+
                            self.Delays[n][ix][iy]) * self.SamplingFrequency)) < L)
                            else 0. +0j for n in range(Nd) for m in range(Nd)))

            # return reduce(lambda x,y: x+y,
            # (A[m,n,int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy]
            # + self.ProbeDelays[m,n])*self.SamplingFrequency))] if (
            # isfinite(self.Delays[m][ix][iy]) and
            # isfinite(self.Delays[n][ix][iy]) and
            # int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy] +
            # self.ProbeDelays[m,n])*self.SamplingFrequency)) < L) else 0.+0j
            # for n in range(Nd) for m in range(Nd)))

        # return array([PointFocus(ix,iy,a) for ix in range(IX) for iy in
        # range(IY)]).reshape((IX,IY)).transpose()

        if AsParallel:

            return np.array(ProcessingPool().map(PointFocus, [(ix, iy) for ix in range(IX) for iy in range(IY)])).reshape((IX, IY)).transpose()

        else:

            return np.array([PointFocus((ix,iy)) for ix in range(IX) for iy in range(IY)]).reshape((IX, IY)).transpose()
