hi
from functools import reduce
import numpy as np
from numpy.fft import rfft, ifft, fftn, ifftn, fftshift
from pathos.multiprocessing import ProcessingPool
import os
import multiprocessing

def NumericalAperture(x,y,L):

    x,y = np.meshgrid(x,y)

    rp = np.sqrt((x - L/2)**2 + y**2)

    rm = np.sqrt((x + L/2)**2 + y**2)

    A = np.sin(0.5*np.arccos((rp**2 + rm**2 - L**2)/(2*rp*rm)))

    A[0,:] = A[1,:]

    return A





def NextPow2(x):
    return int(2**int(np.ceil(np.log2(x))))


def RBF(x, ai, xi, beta):

    return ai * np.exp(-beta * (x - xi)**2)


def FitRBF(x, f, beta):

    from numpy.linalg import solve

    return solve(np.exp(-beta * (x.reshape(-1, 1) -
                                 x.reshape(1, -1))**2), f.reshape(-1, 1))


def EstimateProbeDelays(Scans, fsamp, p, h, c=5.92):

    from scipy.signal import hilbert

    M = Scans.shape[0]
    N = Scans.shape[1]

    x = np.abs(hilbert(Scans, axis=2))

    W = int(np.round(fsamp * 0.25 * h / c))

    Delays = np.zeros((M, N))

    A = np.zeros(M)

    for m in range(M):

        A[m] = np.sqrt(np.max(np.abs(x[m, m, W::])))

        for n in range(N):

            T = int(
                np.round(fsamp * (2 * np.sqrt((0.5 * (n - m) * p)**2 + h**2) / c)))

            indmax = np.argmax(np.abs(x[m, n, T - W:T + W])) + T - W - T

            Delays[m, n] = indmax / fsamp

    return Delays, A.reshape(-1, 1) * A.reshape(1, -1)


def EstimateDirectivityAttenuation(
        Scans,
        positions,
        fsamp,
        pitch,
        c,
        amps,
        delays,
        thetagrid,
        rgrid,
        fcentre=5.):

    from scipy.optimize import minimize

    betat = 4.0 * np.log(2) / (min(thetagrid) / 2)**2
    betar = 4.0 * np.log(2) / (min(rgrid) / 2)**2

    N = Scans[0].shape[0]

    F = LinearCapture(fsamp, Scans, pitch, N, delays, amps)

    F.ProcessScans(100)

    Imax = []

    A = []

    theta = []

    r = []

    for i in range(len(F.AScans[0:2])):

        F.GetContactDelays(
            np.linspace(
                positions[i][0] - 2.5,
                positions[i][0] + 2.5,
                50),
            np.linspace(
                positions[i][1] - 2.5,
                positions[i][1] + 2.5,
                50),
            c)

        I = np.abs(F.ApplyTFM(i))

        indmax = np.unravel_index(np.argmax(I), I.shape)

        xmax = F.xRange[indmax[1]]

        ymax = F.yRange[indmax[0]]

        theta.append([np.arctan2(n * pitch - xmax, ymax) for n in range(N)])

        r.append([np.sqrt((xmax - n * pitch)**2 + ymax**2) for n in range(N)])

        A.append(np.array([[np.abs(F.AScans[i][m, n, int(round(
            fsamp * (r[i][m] + r[i][n]) / c))]) for n in range(N)] for m in range(N)]))

        Imax.append(np.amax(I))

        print(str(i))

    def Loss(x):

        atheta = x[0:len(thetagrid)]
        ar = x[len(thetagrid)::]

        print(type(atheta))
        print(type(ar))

        def loss(i):

            return reduce(lambda x,
                          y: x + y,
                          [(1. - (A[i][m,
                                       n] / Imax[i]) * RBF(theta[i][m],
                                                           atheta[t],
                                                           thetagrid[t],
                                                           betat) * RBF(theta[i][n],
                                                                        atheta[t],
                                                                        thetagrid[t],
                                                                        betat) * RBF(r[i][m],
                                                                                     ar[r],
                                                                                     rgrid[r],
                                                                                     betar) * RBF(r[i][n],
                                                                                                  ar[r],
                                                                                                  rgrid[r],
                                                                                                  betar))**2 for m in range(N) for n in range(N) for t in range(len(atheta)) for r in range(len(ar))])

        return reduce(lambda x, y: x + y, (loss(i) for i in range(len(Imax))))

    a0theta = FitRBF(thetagrid, np.sinc(
        pitch * np.sin(thetagrid) / (fcentre / c)), betat)
    a0r = FitRBF(rgrid, 1. / np.sqrt(rgrid), betar)

    print(type(A[0][0, 0]))
    print(type(Imax[0]))
    print(type(theta[0][0]))
    # print(type(atheta[0]))
    print(type(thetagrid[0]))
    print(type(r[0][0]))
    # print(type(ar[0]))
    print(type(rgrid[0]))

    X = minimize(Loss, x0=list(a0theta) + list(a0r))

    def Directivity(x):
        return reduce(lambda x, y: x + y, [RBF(x, a, betat) for a in X[0:len(thetagrid)]])

    def Attenuation(x):
        return reduce(lambda x, y: x + y, [RBF(x, a, betar) for a in X[len(thetagrid)::]])

    return Directivity, Attenuation


class LinearCapture:

    def __init__(self, fs, scans, p, N, probedelays=None, WedgeParameters=None):

        import copy

        self.SamplingFrequency = fs
        self.Pitch = p
        # self.Velocity = c
        self.NumberOfElements = N

        if probedelays is None:

            self.ProbeDelays = np.zeros((N, N))

        else:

            self.ProbeDelays = probedelays


        self.AScans = copy.deepcopy(scans)

        self.AScans = [a.astype(np.float)/32768. for a in self.AScans]

        self.AmplitudeCorrection = None

        self.WedgeParameters = WedgeParameters


    def ProcessScans(self, zeropoints=20, bp=10, normalize=True):

        from scipy.signal import detrend, hilbert
        from numpy.linalg import norm

        L = self.AScans[0].shape[2]

        d = np.round(self.ProbeDelays*self.SamplingFrequency).astype(int)

        dmax = np.amax(d)

        if dmax<zeropoints:

            for i in range(len(self.AScans)):

                for m in range(self.NumberOfElements):

                    for n in range(self.NumberOfElements):

                        self.AScans[i][m,n,0:zeropoints-d[m,n]] = 0.

                self.AScans[i] = hilbert(detrend(self.AScans[i], bp=list(np.arange(0, L, bp).astype(int))))

                if normalize:

                    self.AScans[i] = self.AScans[i]/norm(self.AScans[i])



    def ReverseElements(self):

        self.AScans = [a[::-1,::-1,:] for a in self.AScans]


        # Lpad = NextPow2(np.round((L + np.amax(self.ProbeDelays)*self.SamplingFrequency - 1)))
        #
        # Lpad = int(
        #     np.round(
        #         (L +
        #          np.amax(
        #              self.ProbeDelays) *
        #             self.SamplingFrequency -
        #             1)))
        #
        # f = np.linspace(0., self.SamplingFrequency / 2, np.floor(Lpad / 2) + 1)
        #
        # f = f.reshape((1, 1, len(f)))
        #
        # D = np.exp(
        #     2j * np.pi * np.repeat(self.ProbeDelays[:, :, np.newaxis], f.shape[2], 2) * f)
        #
        # A = np.repeat(self.ElementAmplitudes[:, :, np.newaxis], f.shape[2], 2).astype(
        #     np.complex64)
        #
        # for i in range(len(self.AScans)):
        #
        #     if zeropoints != 0:
        #         self.AScans[i][:, :, 0:zeropoints] = 0.0
        #
        #     self.AScans[i] = detrend(
        #         self.AScans[i], bp=list(
        #             np.linspace(
        #                 0, L - 1, bp).astype(int)))
        #
        #     X = rfft(self.AScans[i], n=Lpad)
        #
        #     self.AScans[i] = self.AScans[i].astype(np.complex64)
        #
        #     self.AScans[i] = 2 * ifft((X / A) * D, n=Lpad)[:, :, 0:L]
        #
        # if T0 is not None:
        #
        #     Npad = int(round(T0*self.SamplingFrequency))
        #
        #     zpad = np.zeros(Npad,dtype=np.complex64)
        #
        #     self.AScans = [np.concatenate((zpad, a)) for a in self.AScans]

    def PlaneWaveSweep(self, ScanIndex, Angles, c):

        # from numpy import sum as asum

        d = self.Pitch * (self.NumberOfElements - 1)

        # d = self.Pitch*self.NumberOfElements

        d = np.linspace(-d / 2, d / 2, self.NumberOfElements)

        L = self.AScans[ScanIndex].shape[2]

        if isinstance(Angles, tuple):

            Angles = (Angles[0] * np.pi / 180., Angles[1] * np.pi / 180.)

            T = np.abs(np.sin(np.repeat((Angles[0].reshape(-1, 1) * Angles[1].reshape(1, -1))[
                       :, :, np.newaxis], len(d), 2)) * d.reshape((1, 1, len(d))) / c).flatten()

        else:

            Angles = Angles * np.pi / 180.

            T = np.abs(np.sin(np.repeat((Angles.reshape(-1, 1) * Angles.reshape(1, -1))
                                        [:, :, np.newaxis], len(d), 2)) * d.reshape((1, 1, len(d))) / c).flatten()

        Npad = int(np.round(self.SamplingFrequency * np.max(T)))

        Lpad = L + Npad - 1
        # X = rfft(np.concatenate((np.zeros((self.NumberOfElements, self.NumberOfElements, Npadstart)),
        #                       np.real(self.AScans[ScanIndex]), np.zeros((self.NumberOfElements,
        # self.NumberOfElements, Npadend))), axis=2), axis=2)

        X = rfft(np.real(self.AScans[ScanIndex]), Lpad)

        f = np.linspace(0, self.SamplingFrequency / 2, X.shape[2])

        def PlaneWaveFocus(angles):

            T = np.meshgrid(f, d * np.sin(angles[1]) / c)

            XX = np.sum(X * np.exp(-2j * np.pi *
                                   T[0] * T[1]), axis=1, keepdims=False)

            T = np.meshgrid(f, d * np.sin(angles[0]) / c)

            XX = np.sum(XX * np.exp(-2j * np.pi *
                                    T[0] * T[1]), axis=0, keepdims=False)

            x = ifft(XX, n=Lpad)

            return x[0:L]

        if isinstance(Angles, tuple):

            return np.array([[PlaneWaveFocus((ta, ra))
                              for ra in Angles[1]] for ta in Angles[0]])

        else:

            return np.array([PlaneWaveFocus((ta, ta)) for ta in Angles])

    def GetContactDelays(self, xrng, yrng, c):

        if c is None:

            c = self.Velocity

        # self.Delays = [[[np.sqrt((x - n * self.Pitch)**2 + y**2) / c for y in yrng]
        #                 for x in xrng] for n in range(self.NumberOfElements)]

        xn = np.linspace(-(self.NumberOfElements-1)*self.Pitch*0.5, (self.NumberOfElements-1)*self.Pitch*0.5, self.NumberOfElements)

        x,y = np.meshgrid(xrng, yrng)


        # self.Delays = [np.sqrt((x - n*self.Pitch)**2 + y**2)/c for n in range(self.NumberOfElements)]

        self.Delays = [np.sqrt((x - xn[n])**2 + y**2)/c for n in range(self.NumberOfElements)]


        # self.GetDelayIndices()


        self.xRange = xrng.copy()

        self.yRange = yrng.copy()

    # def GetDelayIndices(self):
    #
    #         self.DelayIndices = [[[int(np.round(self.Delays[n][ix][iy]*self.SamplingFrequency)) for iy in range(len(self.Delays[0][0]))] for ix in range(len(self.Delays[0]))] for n in range(len(self.Delays))]
    #

    def GetContactCorrections(self, x,y,amplitude,sensitivity=None, isongrid=False):

        from scipy.interpolate import griddata

        if isongrid:

            xyi = np.meshgrid(x, y)

            xyi = (xyi[0].flatten(), xyi[1].flatten())

        else:

            xyi = (x.flatten(), y.flatten())


        # if sensitivity is not None:
        #
        #     sensitivity = np.sqrt(sensitivity/np.amax(sensitivity))

        self.AmplitudeCorrection = []

        for n in range(self.NumberOfElements):

            xyp = np.meshgrid(self.xRange - n * self.Pitch, self.yRange)

            if sensitivity is not None:

                A = griddata(xyi,sensitivity[n]*amplitude.flatten(),(xyp[0].flatten(),xyp[1].flatten()),fill_value=np.nan,method='linear').reshape(xyp[0].shape)



            else:

                A = griddata(xyi,amplitude.flatten(),(xyp[0].flatten(),xyp[1].flatten()),fill_value=np.nan,method='linear').reshape(xyp[0].shape)



            ind = np.where(np.isfinite(A[0,:]))[0]


            A[:,0:ind[0]]=A[:,ind[0]].reshape((-1,1))

            A[:,ind[-1]::]=A[:,ind[-1]].reshape((-1,1))

            self.AmplitudeCorrection.append(A)




    def GetWedgeDelays(self, xrng, yrng, c):

        from scipy.optimize import minimize_scalar,minimize
        # from scipy.optimize import brentq

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = np.cos(self.WedgeParameters['Angle'] * np.pi / 180.)
        sphi = np.sin(self.WedgeParameters['Angle'] * np.pi / 180.)

        # x,y = np.meshgrid(xrng,yrng)


        # def f(x,X,Y,n):
        #
        #     return np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2) / cw + np.sqrt(Y**2 + (X - x)**2) / c
        # #
        # def J(x,X,Y,n):
        #     return -(cphi * n * p - x) / (cw * np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2)) - \
        #             (X - x) / (c * np.sqrt(Y**2 + (X - x)**2))


        # def f(x,X,Y,n):
        #
        #     return (x - cphi*n*p)*(c * np.sqrt(Y**2 + (X - x)**2))/(cw * np.sqrt((h + n * p * sphi)**2 + (x - cphi*n*p)**2)) - (X - x)


        # self.Delays = [np.array([[minimize(f,x0=0.5 * np.abs(x - n * self.Pitch * cphi),
        #                 args=(x,y,n),method='BFGS',jac=J).fun for y in yrng] for x in xrng])
        #                for n in range(self.NumberOfElements)]


        def f(X,Y,n):

            P = np.zeros(5)

            P[0]=-c**2 + cw**2
            P[1]=2*X*c**2 - 2*X*cw**2 + 2*c**2*cphi*n*p - 2*cphi*cw**2*n*p
            P[2]=-X**2*c**2 + X**2*cw**2 - 4*X*c**2*cphi*n*p + 4*X*cphi*cw**2*n*p - Y**2*c**2 - c**2*cphi**2*n**2*p**2 + cphi**2*cw**2*n**2*p**2 + cw**2*h**2 + 2*cw**2*h*n*p*sphi + cw**2*n**2*p**2*sphi**2
            P[3]=2*X**2*c**2*cphi*n*p - 2*X**2*cphi*cw**2*n*p + 2*X*c**2*cphi**2*n**2*p**2 - 2*X*cphi**2*cw**2*n**2*p**2 - 2*X*cw**2*h**2 - 4*X*cw**2*h*n*p*sphi - 2*X*cw**2*n**2*p**2*sphi**2 + 2*Y**2*c**2*cphi*n*p
            P[4]=-X**2*c**2*cphi**2*n**2*p**2 + X**2*cphi**2*cw**2*n**2*p**2 + X**2*cw**2*h**2 + 2*X**2*cw**2*h*n*p*sphi + X**2*cw**2*n**2*p**2*sphi**2 - Y**2*c**2*cphi**2*n**2*p**2

            r = np.roots(P)

            r = r[(np.real(r)>=0.)&(~(np.abs(np.imag(r))>0.))]


            if len(r)>0:

                x = np.real(r[0])

                return np.sqrt((h + n * p * sphi)**2 + (cphi * n * p - x)**2) / cw + np.sqrt(Y**2 + (X - x)**2) / c

            else:

                return np.nan


        x,y = np.meshgrid(xrng,yrng)

        ComputeDelays = np.vectorize(f,excluded=['n'])

        self.Delays = [ComputeDelays(x,y,n) for n in range(self.NumberOfElements)]


        # ComputeDelays = np.vectorize(f)

        # self.Delays = []
        #
        # for n in range(self.NumberOfElements):
        #
        #     ComputeDelays = np.vectorize(f)
        #
        #     self.Delays.append(ComputeDelays(x,y))






        # def ComputeDelay(n,x,y):
        #
        #     try:
        #         # return brentq(f,n*p*cphi, (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi), args=(x,y,n),xtol=1e-6)
        #
        #         # return minimize_scalar(f,(n*p*cphi,(x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi)),args = (x,y,n),tol=1e-4,options={'maxiter':30}).x
        #         # return brentq(f,n*p*cphi, x, args=(x,y,n),xtol=-6)
        #
        #         return minimize(f,0.5*(n*p*cphi + (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi)),args=(x,y,n),jac=J,tol=1e-4,options={'maxiter':20}).fun
        #
        #
        #     except ValueError:
        #
        #         return np.nan

        # self.Delays = [np.array([[brentq(f, n*p*cphi, (x-n*p*cphi)*(h+n*p*sphi)/(y+h+n*p*sphi), args=(x,y,n)) for x in xrng] for y in yrng]) for n in range(self.NumberOfElements)]

        # self.Delays = [np.array([[ComputeDelay(n,x,y) for y in yrng] for x in xrng]) for n in range(self.NumberOfElements)]



        self.xRange = xrng.copy()
        self.yRange = yrng.copy()

    def KeepElements(self, Elements):

        for i in range(len(self.AScans)):

            self.AScans[i] = np.take(
                np.take(
                    self.AScans[i],
                    Elements,
                    axis=0),
                Elements,
                axis=1)

        if self.ProbeDelays is not None:

            self.ProbeDelays = np.take(
                np.take(
                    self.ProbeDelays,
                    Elements,
                    axis=0),
                Elements,
                axis=1)

        self.NumberOfElements = len(Elements)

    def FitInterfaceCurve(self, ScanIndex, hrng, c, smoothparams=(0.1, 0.1)):

        from scipy.interpolate import interp1d
        # from scipy.signal import guassian, convolve
        from skimage.filters import threshold_li, gaussian
        from misc import DiffCentral

        xrng = np.linspace(
            0,
            self.NumberOfElements *
            self.Pitch,
            self.NumberOfElements)

        self.GetContactDelays(xrng, hrng, c)

        I = gaussian(np.abs(self.ApplyTFM(ScanIndex)), smoothparams)

        dh = hrng[1] - hrng[0]

        hgrid = hrng[0] + dh * np.argmax(I, axis=0)

        dhdx = np.abs(DiffCentral(hgrid))

        indkeep = np.where(dhdx < threshold_li(dhdx))

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

        h = interp1d(
            xrng,
            hgrid,
            kind='quadratic',
            bounds_error=False,
            fill_value=np.nan)

        dhdx = interp1d(xrng[1::],
                        np.diff(h(xrng)) / np.diff(xrng),
                        bounds_error=False,
                        fill_value=np.nan)

        return h, dhdx

        # return h

    def FitInterfaceLine(self, ScanIndex, angrng, gate, c):
        """
        gate specified in terms of mm in medium with Velocity c
        """
        angles = np.arange(angrng[0], angrng[1], angrng[2])

        X = np.abs(
            self.PlaneWaveSweep(
                ScanIndex, angles, c))[
            :, int(
                np.round(
                    2 * gate[0] * self.SamplingFrequency / c)):int(
                        np.round(
                            2 * gate[1] * self.SamplingFrequency / c))]

        imax = np.unravel_index(X.argmax(), X.shape)

        angmax = -angles[imax[0]] * np.pi / 180.

        hmax = gate[0] + imax[1] * c / (2 * self.SamplingFrequency)

        h0 = hmax / (np.cos(angmax))

        def h(x): return np.tan(angmax) * x + h0 - self.Pitch * \
            self.NumberOfElements * np.tan(angmax) / 2

        return h

    def GetAdaptiveDelays(self, xrng, yrng, h, cw, cs, AsParallel=False):

        # def GetAdaptiveDelays(self, xrng, yrng, h, dhdx, cw, cs,
        # AsParallel=False):

        from scipy.optimize import minimize_scalar
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
            return np.sqrt((x - n * self.Pitch)**2 + h(x) ** 2) / \
                cw + np.sqrt((X - x)**2 + (Y - h(x))**2) / cs

        # def dfdx(x, X, Y, n):
        #     return ((x - n * self.Pitch + h(x))**(-1 / 2.)) * (x - n * self.Pitch + h(x) * dhdx(x)) / \
        #         cw - (((X - x)**2 + (Y - h(x))**2)**(-1 / 2.)) * ((X - x)**2 + (Y - h(x)) * dhdx(x)) / cs

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

            return [
                [
                    float(
                        minimize_scalar(
                            f,
                            bracket=(
                                n *
                                self.Pitch,
                                x),
                            args=(
                                x,
                                y,
                                n),
                            method='Brent',
                            options={
                                'xtol': 1e-3,
                                'maxiter': 50}).fun) if y >= h(x) else np.nan for y in yrng] for x in xrng]

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

            self.Delays = ProcessingPool().map(
                DelayMin, [n for n in range(self.NumberOfElements)])

        else:

            self.Delays = [DelayMin(n) for n in range(self.NumberOfElements)]

        self.xRange = xrng

        self.yRange = yrng

    def FilterByAngle(self, ScanIndex, filtertype, angle, FWHM, c):

        L = self.AScans[ScanIndex].shape[2]

        # Lpad = NextPow2(L)

        X = fftshift(
            fftn(
                np.real(
                    self.AScans[ScanIndex]), s=(
                    self.NumberOfElements, L), axes=(
                    0, 2)), axes=(0))

        # X = fftshift(rfft(fft(np.real(self.AScans[ScanIndex]), axis = 0)), axes = (0))

        X = X[:, :, 0:int(np.floor(X.shape[2] / 2) + 1)]

        kx = 2 * np.pi * np.linspace(-1 / (2 * self.Pitch),
                                     1 / (2 * self.Pitch), X.shape[0]).reshape(-1, 1)

        w = 2 * np.pi * np.linspace(0., self.SamplingFrequency /
                                    2, X.shape[2]).reshape(1, -1)

        th = np.arcsin(c * kx / w)

        th = np.repeat(
            th.reshape(
                (kx.shape[0],
                 1,
                 w.shape[1])),
            X.shape[1],
            axis=1) * 180 / np.pi

        alpha = ((2.35482)**2) / (2 * FWHM**2)

        FilterFunction = {
            'Band': np.exp(-alpha * (th - angle)**2), 'Notch': 1. - np.exp(-alpha * (th - angle)**2)}

        H = np.nan_to_num(FilterFunction[filtertype]).astype(type(X[0, 0, 0]))

        X = H * X

        X = fftshift(X, axes=(0))

        return 2 * ifftn(X, s=(X.shape[0], L), axes=(0, 2))

    def ApplyTFM(self, ScanIndex, FilterParams=None, stablecoeff=1e-4, Normalize=False):
        #
        # IX = len(self.Delays[0])
        # IY = len(self.Delays[0][0])

        if FilterParams is None:

            a = self.AScans[ScanIndex]

        else:

            a = self.FilterByAngle(
                ScanIndex,
                FilterParams[0],
                FilterParams[1],
                FilterParams[2],
                FilterParams[3])


        # L = self.AScans[ScanIndex].shape[2]
        #
        L = a.shape[2]

        t = np.linspace(0.,L-1,L)/self.SamplingFrequency
        #


        # delaytype = (len(self.Delays) == len(self.AScans))
        #
        # if self.AmplitudeCorrection is None:
        #
        #     def PointFocus(pt):
        #
        #     # Nd = len(self.Delays)
        #
        #         ix = pt[0]
        #         iy = pt[1]
        #
        #
        #
        #         I = 0.+0j
        #
        #         for m in range(Nd):
        #             for n in range(Nd):
        #
        #                 try:
        #
        #                     d = int(np.round((self.Delay[m][ix][iy]+self.Delay[n][ix][iy])*self.SamplingFrequency))
        #
        #                     I += a[m,n,int(d)]
        #
        #                 except:
        #
        #                     pass
        #
        #         return I
        #
        #
        #         # return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy]) * self.SamplingFrequency))] if (np.isfinite(self.Delays[m][ix][iy])\
        #         # and np.isfinite(self.Delays[n][ix][iy]) and int(round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy]) * self.SamplingFrequency)) < L)\
        #         # else 0. +0j for n in range(Nd) for m in range(Nd)))
        #
        #
        # else:
        #
        #     def PointFocus(pt):
        #
        #         ix = pt[0]
        #         iy = pt[1]
        #
        #
        #         I = 0.+0j
        #
        #         for m in range(Nd):
        #             for n in range(Nd):
        #
        #                 try:
        #
        #                     d = int(np.round((self.Delay[m][ix][iy]+self.Delay[n][ix][iy])*self.SamplingFrequency))
        #
        #                     A = self.self.AmplitudeCorrection[m][iy,ix]*self.AmplitudeCorrection[n][iy,ix]
        #
        #                     if (not(np.isnan(A))):
        #
        #                         I += a[m,n,int(d)]/(A+stablecoeff)
        #
        #                 except:
        #
        #                     pass
        #
        #         return I
        #
        #         # return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy])*self.SamplingFrequency))]/(self.AmplitudeCorrection[m][iy,ix]*self.AmplitudeCorrection[n][iy,ix] + stablecoeff)
        #         # if (np.isfinite(self.Delays[m][ix][iy]) and np.isfinite(self.Delays[n][ix][iy]) and np.isfinite(self.AmplitudeCorrection[m][iy,ix]) and np.isfinite(self.AmplitudeCorrection[n][iy,ix])
        #         # and int(round((self.Delays[m][ix][iy]+self.Delays[n][ix][iy])*self.SamplingFrequency)) < L) else 0.+0j for n in range(Nd) for m in range(Nd)))
        #
        # if AsParallel:
        #
        #     pool_size = multiprocessing.cpu_count()
        #     os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))
        #
        #     return np.array(
        #         ProcessingPool(pool_size).map(
        #             PointFocus, [
        #                 (ix, iy) for ix in range(IX) for iy in range(IY)])).reshape(
        #         (IX, IY)).transpose()
        #
        # else:
        #
        #     return np.array([PointFocus((ix, iy)) for ix in range(IX)
        #                      for iy in range(IY)]).reshape((IX, IY)).transpose()



        if self.AmplitudeCorrection is None:

            def ElementFocus(m,n):

                I = np.interp((self.Delays[m]+self.Delays[n]).flatten(),t,a[m,n,:])

                np.nan_to_num(I,copy=False)

                return I

        else:

            def ElementFocus(m,n):

                I = np.interp((self.Delays[m]+self.Delays[n]).flatten(),t,a[m,n,:])

                I = I/((self.AmplitudeCorrection[m]*self.AmplitudeCorrection[n]+stablecoeff).flatten())

                np.nan_to_num(I,copy=False)

                return I


        I = reduce(lambda x,y: x+y, (ElementFocus(m,n) for m in range(self.NumberOfElements) for n in range(self.NumberOfElements))).reshape(self.Delays[0].shape)

        if Normalize:

            I/np.amax(np.abs(I))

        return I


    def AttenuationTomography(
        self, ScanIndex, RefIndex, c, d, fband, fpower, resolution=(
            0.6, 0.6), windowparams=(
            50, 0.1, 4), rcondnum = 1e-5):

        from scipy.signal import tukey
        from numpy.fft import irfft2, irfft
        from matplotlib.pylab import plot, show, imshow

        dx = resolution[0]
        dy = resolution[1]

        Nx = int(np.floor(self.NumberOfElements * self.Pitch / dx)) + 1
        Ny = int(np.floor(d / dy)) + 1

        # G0 = np.zeros((int(self.NumberOfElements**2), 1), dtype=complex)
        # G1 = np.zeros((int(self.NumberOfElements**2), 1), dtype=complex)


        G = np.zeros((int(self.NumberOfElements**2), 1), dtype=float)


        # kx, ky = np.meshgrid(2. * np.pi * np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nx),
        #                      2. * np.pi * np.linspace(0., 1 / (2 * dy), int(round(Ny / 2))))
        #
        # kxyshape = kx.shape
        #
        # kx = kx.flatten()
        # ky = ky.flatten()

        # B = np.zeros((int(self.NumberOfElements * self.NumberOfElements),
        #               int(kxyshape[0] * kxyshape[1])), dtype=complex)
        #

        B = np.zeros((int(self.NumberOfElements * self.NumberOfElements),
                      int(Nx*Ny)), dtype=float)


        # B = np.zeros((int(self.NumberOfElements * self.NumberOfElements/2),
        #                   int(Nx*Ny)), dtype=float)

        w = tukey(int(2 * windowparams[0]), windowparams[1])
        #
        f = np.linspace(0., self.SamplingFrequency / 2,
                        int(np.floor(len(w) * windowparams[2] / 2)) + 1)

        indf = (f >= fband[0]) & (f <= fband[1])

        f = f[indf]



        for m in range(self.NumberOfElements):

            xm = m * self.Pitch

            # for n in range(m):

            for n in range(self.NumberOfElements):

            # for n in range(m,self.NumberOfElements):


                xn = n * self.Pitch

                ind = int(round(2 * self.SamplingFrequency *
                                np.sqrt((0.5 * (xn - xm))**2 + d**2) / c))


                # RR = np.max(np.abs(self.AScans[RefIndex][m, n, ind -
                #                                          windowparams[0]:ind -
                #                                          windowparams[0] +
                #                                          int(2 *
                #                                              windowparams[0])]))
                #
                # R = np.max(np.abs(self.AScans[ScanIndex][m, n, ind -
                #                                              windowparams[0]:ind -
                #                                              windowparams[0] +
                #                                              int(2 *
                #                                                  windowparams[0])]))
                #
                #
                indrr = np.argmax(np.abs(self.AScans[RefIndex][m, n, ind -
                                                         windowparams[0]:ind -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])])) + ind - windowparams[0]


                indr = np.argmax(np.abs(self.AScans[ScanIndex][m, n, ind -
                                                         windowparams[0]:ind -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])])) + ind - windowparams[0]


                #

                RR = rfft(w*np.real(self.AScans[RefIndex])[m, n, indrr -
                                                         windowparams[0]:indrr -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])], int(len(w) *
                                                                                    windowparams[2]))

                R = rfft(w*np.real(self.AScans[ScanIndex])[m, n, indr -
                                                         windowparams[0]:indr -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])], int(len(w) *
                                                                                    windowparams[2]))

                #
                # RR = rfft(w*np.real(self.AScans[RefIndex])[m, n, ind -
                #                                                  windowparams[0]:ind -
                #                                                  windowparams[0] +
                #                                                  int(2 *
                #                                                      windowparams[0])], int(len(w) *
                #                                                                             windowparams[2]))
                #
                # R = rfft(w*np.real(self.AScans[ScanIndex])[m, n, ind -
                #                                                  windowparams[0]:ind -
                #                                                  windowparams[0] +
                #                                                  int(2 *
                #                                                      windowparams[0])], int(len(w) *
                #                                                                             windowparams[2]))

                RR = RR[indf]
                R = R[indf]
                #
                GG = np.log(np.abs(RR)/np.abs(R))
                # #
                # # GGG.append(GG)
                #
                G[n+m*self.NumberOfElements, 0] = np.trapz(GG/(f**fpower),dx=(f[1]-f[0]))/(fband[1]-fband[0])

                # G[n+m*self.NumberOfElements, 0] = np.log(RR/R)


                # G = np.log(np.abs(R * np.conj(RR) /
                #                   (RR * np.conj(RR) + 1e-2 * np.amax(np.abs(RR)))))

                # pfit = np.polyfit(f, G, 1)

                # G0[n + m * self.NumberOfElements, 0] = pfit[1]
                # G1[n + m * self.NumberOfElements, 0] = -pfit[0]


                # G0[n+m*self.NumberOfElements, 0] = np.trapz(-G,dx=(f[1]-f[0]))/(fband[1]-fband[0])
                #
                # G1[n+m*self.NumberOfElements, 0] = np.trapz(-G/f,dx=(f[1]-f[0]))/(fband[1]-fband[0])
                #
                # GG.append(G0)


                def xf(y):
                    return (((xn-xm)*y)/(2*d)) + xm

                def yf(x):
                    return ((x - xm)*2*d)/(xn - xm)

                def xb(y):
                    return (((xm-xn)*y)/(2*d)) + xn

                def yb(x):
                    return ((x - xn)*2*d)/(xm - xn)


                # def xf(y):
                #     return (xn-3*xm)*y/(2*d) + xm
                #
                # def yf(x):
                #     return 2*d*(x-xm)/(xn-3*xm)
                #
                # def xb(y):
                #     return -(xm+xn)*y/(2*d) + xn
                #
                # def yb(x):
                #     return -(x - xn)*2*d/(xm + xn)


                xy = []


                for p in range(int(round(np.ceil(xm/dx))), int(round(np.floor(xf(d)/dx)))):

                    xp = dx*p

                    yp = yf(xp)

                    # pq.append([p, int(np.floor(yp/dy))])
                    #
                    # r.append(np.sqrt(xp**2 + yp**2))

                    xy.append([xp, yp])

                for q in range(Ny):

                    yq = dy*q

                    xq = xf(yq)

                    # pq.append([int(np.floor(xq/dx)), q])
                    #
                    # r.append(np.sqrt(xq**2 + yq**2))

                    xy.append([xq,yq])


                xy = np.array(xy).reshape((len(xy),2))

                ixysort = np.argsort(xy[:,1])

                xy = xy[ixysort,:]

                # print(xy[:,0])
                #
                # print(xy[:,1])

                for i in range(xy.shape[0]-1):

                    x0 = xy[i, 0]
                    x1 = xy[i+1 , 0]

                    y0 = xy[i, 1]
                    y1 = xy[i+1, 1]

                    p = int(np.floor(x0/dx))
                    q = int(np.floor(y0/dy))

                    r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

                    B[n + m * self.NumberOfElements, p+q*Nx] = B[n + m * self.NumberOfElements, p+q*Nx] + r




                xy = []


                for p in range(int(np.ceil(xf(d)/dx)), int(np.floor(xn/dx))):

                    xp = dx*p

                    yp = yb(xp)

                    # pq.append([p, int(np.floor(yp/dy))])
                    #
                    # r.append(np.sqrt(xp**2 + yp**2))

                    xy.append([xp, yp])

                for q in range(Ny):

                    yq = dy*q

                    xq = xb(yq)

                    # pq.append([int(np.floor(xq/dx)), q])
                    #
                    # r.append(np.sqrt(xq**2 + yq**2))

                    xy.append([xq,yq])


                xy = np.array(xy).reshape((len(xy),2))

                ixysort = np.argsort(xy[:,1])

                ixysort = ixysort[::-1]

                xy = xy[ixysort, :]

                # print(xy[:,0])
                # print(xy[:,1])


                # print(xy)


                for i in range(xy.shape[0]-1):

                    x0 = xy[i, 0]
                    x1 = xy[i+1 , 0]

                    y0 = xy[i, 1]
                    y1 = xy[i+1, 1]

                    p = int(np.floor(x0/dx))
                    q = int(np.floor(y0/dy))

                    r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

                    B[n + m * self.NumberOfElements, p+q*Nx] = B[n + m * self.NumberOfElements, p+q*Nx] + r




                # r = np.array(r)
                # pq = np.array(pq)
                #
                # indr = np.argsort(r)
                #
                # pq = pq[indr,:]
                # r = r[indr]
                #
                #
                # r = np.diff(r)


                #
                # for i in range(len(p):
                #
                #     B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Nx] =
                #
                #
                # r = []
                #
                # pq = []
                #
                # for p in range(int(round(np.ceil(xf(d)/dx))), int(round(np.floor(xn/dx)))):
                #
                #     xp = dx*p
                #
                #     yp = yb(xp)
                #
                #     pq.append([p, int(np.floor(yp/dy))])
                #
                #     r.append(np.sqrt(xp**2 + yp**2))
                #
                #     # r.append(np.sqrt(xp**2 + yp**2))
                #
                #
                # for q in range(Ny):
                #
                #     yq = dy*q
                #
                #     xq = xb(yq)
                #
                #     # print(yq)
                #
                #
                #     pq.append([int(np.floor(xq/dx)), q])
                #
                #     # r.append(np.sqrt(xq**2 + yq**2))
                #
                #     r.append(np.sqrt(xq**2 + yq**2))
                #
                #
                #
                #
                # r = np.array(r)
                # pq = np.array(pq)
                #
                # indr = np.argsort(r)
                #
                # pq = pq[indr,:]
                # r = r[indr]
                #
                # r = np.diff(r)
                #
                # # print(pq)
                # # print(r)
                #
                # for i in range(pq.shape[0]-1):
                #
                #     # B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Ny] = r[i]
                #
                #     B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Nx] = r[i]
                #
                #
                #






                # B[ n + m*Nx, :] =

                # B[n + m*Nx,:] = (np.exp(1j*kx*xm)*(np.exp(1j*(kx*(xn-xm) + ky*2*d)) - 1)/(kx*(xn-xm)/(2*d) + ky)).flatten()

                # B[n + m * self.NumberOfElements, :] = (np.exp(1j * kx * xm) * (np.exp(1j * (kx * 0.5 * (xn - xm) + ky * d)) - 1)) / (1j * (kx * 0.5 * (
                #     xn - xm) / d + ky)) - (np.exp(1j * kx * xn) * (np.exp(1j * (kx * 0.5 * (xm - xn) + ky * d)) - 1)) / (1j * (kx * 0.5 * (xm - xn) / d + ky))

        # indnz = (np.real(G0) > 0.).flatten()
        #
        # # G0 = np.nan_to_num(G0)
        # #
        # # G1 = np.nan_to_num(G1)
        #
        # #
        # G0 = G0[indnz]
        #
        # G1 = G1[indnz]
        #
        # # B = np.nan_to_num(B)
        #
        # B = B[indnz,:]

        # print(np.amax(np.abs(B)))

        # A0 = np.linalg.lstsq(B, G0)[0].reshape(kxyshape)

        # print(np.linalg.cond(B))
        #
        # G0[G0<0.] = 0.
        #
        # G1[G1<0.] = 0.
        #
        #
        # I0 = np.linalg.lstsq(B, G0, rcond=rcondnum)[0].reshape((Ny,Nx))
        #
        # I1 = np.linalg.lstsq(B, G1, rcond=rcondnum)[0].reshape((Ny,Nx))


        I = np.linalg.lstsq(B, G, rcond=rcondnum)[0].reshape((Ny,Nx))



        #
        # A0 = np.nan_to_num(A0)
        #
        # A1 = np.linalg.lstsq(B, G1)[0].reshape(kxyshape)
        #
        # A1 = np.nan_to_num(A1)
        #
        # I0 = fftshift(ifftn(A0, axes=(1, 0), s=(Nx, Ny)), axes=(1,))
        #
        # I1 = fftshift(ifftn(A1, axes=(1, 0), s=(Nx, Ny)))



        return I, G
