from functools import reduce
import numpy as np
from numpy.fft import rfft, ifft, fftn, ifftn, fftshift
from pathos.multiprocessing import ProcessingPool


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

    def __init__(self, fs, scans, p, N, probedelays=None, elementamps=None):

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

    def ProcessScans(self, zeropoints=0, bp=10, T0 = None):

        from scipy.signal import detrend

        L = self.AScans[0].shape[2]

        # Lpad = NextPow2(np.round((L + np.amax(self.ProbeDelays)*self.SamplingFrequency - 1)))

        Lpad = int(
            np.round(
                (L +
                 np.amax(
                     self.ProbeDelays) *
                    self.SamplingFrequency -
                    1)))

        f = np.linspace(0., self.SamplingFrequency / 2, np.floor(Lpad / 2) + 1)

        f = f.reshape((1, 1, len(f)))

        D = np.exp(
            2j * np.pi * np.repeat(self.ProbeDelays[:, :, np.newaxis], f.shape[2], 2) * f)

        A = np.repeat(self.ElementAmplitudes[:, :, np.newaxis], f.shape[2], 2).astype(
            np.complex64)

        for i in range(len(self.AScans)):

            if zeropoints != 0:
                self.AScans[i][:, :, 0:zeropoints] = 0.0

            self.AScans[i] = detrend(
                self.AScans[i], bp=list(
                    np.linspace(
                        0, L - 1, bp).astype(int)))

            X = rfft(self.AScans[i], n=Lpad)

            self.AScans[i] = self.AScans[i].astype(np.complex64)

            self.AScans[i] = 2 * ifft((X / A) * D, n=Lpad)[:, :, 0:L]

        if T0 is not None:

            Npad = int(round(T0*self.SamplingFrequency))

            zpad = np.zeros(Npad,dtype=np.complex64)

            self.AScans = [np.concatenate((zpad, a)) for a in self.AScans]



    # def RemoveProbeDelays(self):
    #
    #     d = int(np.round(self.Delays*self.SamplingFrequency))
    #
    #     dmax = np.max(d)
    #
    #     for i in range(len(self.AScans)):
    #
    #         x = np.zeros(())
    #
    #
    #
    #         self.AScans[i][:,:,] = x[]
    #
    #
    # def ApplyWindow(self):
    #
    #
    #
    # def ApplyBandpass(self):
    #
    #

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

    def GetContactDelays(self, xrng, yrng, c, ampcorr=None):

        from scipy.interpolate import griddata
        # if c is None:
        #
        #     c = self.Velocity

        self.Delays = [[[np.sqrt((x - n * self.Pitch)**2 + y**2) / c for y in yrng]
                        for x in xrng] for n in range(self.NumberOfElements)]

        self.xRange = xrng.copy()

        self.yRange = yrng.copy()

        if ampcorr is not None:

            xyi = np.meshgrid(ampcorr['x'], ampcorr['y'])

            xyi = (xyi[0].ravel(), xyi[1].ravel())

            self.AmplitudeCorrection = []

            for n in range(len(self.NumberOfElements)):

                xyp = np.meshgrid(xrng - n * self.Pitch, yrng)

                self.AmplitudeCorrection.append(
                    list(
                        griddata(
                            xyi,
                            ampcorr['AmplitudeCorrection'].ravel(),
                            (xyp[0].flatten(),
                             xyp[1].flatten()),
                            fill_value=np.nan,
                            method='linear').reshape(
                            xyi[0].shape)))

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

            self.AScans[i] = np.take(
                np.take(
                    self.AScans[i],
                    Elements,
                    axis=0),
                Elements,
                axis=1)

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

    def ApplyTFM(self, ScanIndex, FilterParams=None, AsParallel=False):

        IX = len(self.Delays[0])
        IY = len(self.Delays[0][0])

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
        # L = a.shape[2]
        #
        # Nd = len(self.Delays)

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

            # if  :
            #
            #
            # else:
            #
            #     return reduce(lambda x, y: x +y, (a[m, n, int(np.round((self.Delays[m][ix][iy]+
            #                 self.Delays[n][ix][iy]) * self.SamplingFrequency))]
            #                 if (np.isfinite(self.Delays[m][ix][iy]) and np.isfinite(self.Delays[n][ix][iy])
            #                 and int(round((self.Delays[m][ix][iy]+
            #                 self.Delays[n][ix][iy]) * self.SamplingFrequency)) < L)
            # else 0. +0j for n in range(Nd) for m in range(Nd)))

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

            return np.array(
                ProcessingPool().map(
                    PointFocus, [
                        (ix, iy) for ix in range(IX) for iy in range(IY)])).reshape(
                (IX, IY)).transpose()

        else:

            return np.array([PointFocus((ix, iy)) for ix in range(IX)
                             for iy in range(IY)]).reshape((IX, IY)).transpose()

    def PhasedArrayImage(
            self,
            ScanIndex,
            Angles,
            c,
            Resolution,
            SweepArray=None):

        from scipy.interpolate import griddata

        if SweepArray is None:

            a = np.abs(self.PlaneWaveSweep(ScanIndex, Angles, c))

        else:

            a = SweepArray

        r = np.linspace(
            0., c * a.shape[1] / (2 * self.SamplingFrequency), a.shape[1]).reshape(1, -1)

        angrad = Angles.reshape(-1, 1) * np.pi / 180.

        x = r * np.sin(angrad)
        # y = r/np.cos(angrad)

        y = r * np.cos(angrad)

        xmin = np.min(x)
        xmax = np.max(x)

        xgrid = np.linspace(
            xmin, xmax, int(
                np.round(
                    xmax - xmin) / Resolution))

        ymin = np.min(y)
        ymax = np.max(y)

        ygrid = np.linspace(
            ymin, ymax, int(
                np.round(
                    ymax - ymin) / Resolution))

        xygrid = np.meshgrid(xgrid, ygrid)

        I = griddata(
            (x.flatten(),
             y.flatten()),
            a.flatten(),
            (xygrid[0].flatten(),
             xygrid[1].flatten()),
            method='cubic',
            fill_value=0.,
            rescale=True)

        # I = griddata((x.flatten(), y.flatten()), a.flatten(), (xygrid[0].flatten(), xygrid[1].flatten()), method='nearest', fill_value=0., rescale = False)

        I = I.reshape(xygrid[0].shape)

        # I[I>100.]=100.

        # I =

        return I

    def AdvancedPhasedArrayImage(
            self,
            ScanIndex,
            Angles,
            c,
            L,
            fband,
            Resolution,
            SweepArray=None):

        from scipy.interpolate import griddata

        # from scipy import integrate
        #
        # a = np.abs(self.PlaneWaveSweep(ScanIndex, Angles, c))
        #
        # angrad = np.pi*np.array(Angles).reshape(a.shape)/180.
        #
        # A = a*np.exp(1j*(np.pi + np.angrad))
        #
        # xgrid = np.linspace()
        #
        # ygrid = np.linspace()
        #
        # def CoefficientMatrix(th, pt):

        if SweepArray is None:

            a = np.real(self.PlaneWaveSweep(ScanIndex, Angles, c))

        else:

            a = SweepArray

        A = rfft(a)

        # print(a.shape)

        # print(A.shape)

        f = np.linspace(0., self.SamplingFrequency /
                        2, A.shape[-1]).reshape(1, -1)

        indf = (f >= fband[0]) & (f <= fband[1])
        indf = np.array(indf).flatten()

        # print(indf.shape)

        A = A[:, indf]

        w = 2 * np.pi * f[:, indf]

        # print(A.shape)
        # print(w.shape)

        angrad = np.pi * np.array(Angles).reshape(-1, 1) / 180.

        # AA = A*np.cos(angrad)

        # print(AA.shape)

        # AA = A

        # print(AA.shape)

        kx = 2 * w * np.sin(angrad) / c

        ky = 2 * w * np.cos(angrad) / c

        # print(kx)

        # ky = 2*w/(c*np.cos(angrad))

        # ky = 2*np.sqrt((w/c)**2 - kx**2 + 0j)
        # ky = np.real(ky)+0j

        # ky = ((w/c)**2)/ky
        #
        # AA = A*(ky/(w/c))
        # AA = A*np.cos(angrad)*(np.real(ky)>=0.)

        AA = A / np.cos(angrad)

        # AA = A

        # ky = 2*w*np.cos(angrad)/c

        # print(ky)

        # print(ky.shape)

        # ky = np.sqrt((w/c)**2 - kx**2 + 0j)

        # ky = np.real(ky)
        # L = c*(a.shape[-1]/self.SamplingFrequency)*0.5

        # Nx = np.round(L[0]/self.Pitch)
        #
        # kymax = 2*np.pi*fmax/c
        #
        # dy = np.pi/(2*kymax)
        #
        # Ny = np.round(L[1],dy)
        #
        # kxgrid = 2*np.pi*np.linspace(-1/(2*self.Pitch), 1/(2*self.Pitch), Nx)
        #
        # kygrid = np.linspace(0.,kymax, Ny)

        # Nx = int(np.round(L[0]/Resolution))
        #
        # Ny = int(np.round(L[1]/(2*Resolution)))
        #
        # kxgrid = 2*np.pi*np.linspace(-1/(2*Resolution),1/(2*Resolution), Nx)
        #
        # # kygrid = 2*np.linspace(0.,1/(2*Resolution), Ny)
        #
        # kygrid = 2*np.pi*np.linspace(0,1/(2*Resolution), Ny)

        Nx = int(np.round(L[0] / self.Pitch))

        # Ny = int(np.round(L[1]/(2*self.Pitch)))

        dy = c / (2 * fband[1])

        Ny = int(np.round(L[1] / (2 * dy)))

        kxgrid = 2 * np.pi * \
            np.linspace(-1 / (2 * self.Pitch), 1 / (2 * self.Pitch), Nx)

        kygrid = 2 * np.pi * np.linspace(0, 1 / (2 * dy), Ny)

        kgrid = np.meshgrid(kxgrid, kygrid)

        # print(kx)
        # print(kxgrid)

        # print(kx.shape)
        # print(ky.shape)
        #
        # print(AA.shape)
        #
        # print(kxgrid.shape)
        # print(kygrid.shape)
        #
        # print(kgrid[0].shape)
        # print(kgrid[1].shape)

        I = griddata(
            (kx.flatten(),
             ky.flatten()),
            AA.flatten(),
            (kgrid[0].flatten(),
             kgrid[1].flatten()),
            method='cubic',
            fill_value=0 + 0j,
            rescale=True).reshape(
            kgrid[0].shape)

        # print(Nx)
        # print(Ny)
        # print(I.shape)

        # return np.abs(fftshift(ifftn(I,axes=(1,0),s=(Nx,2*Ny - 2)),
        # axes=(1,)))

        Nx = int(np.round(L[0] / Resolution))
        Ny = int(np.round(L[1] / (2 * Resolution)))

        return np.abs(
            fftshift(ifftn(I, axes=(1, 0), s=(Nx, 2 * Ny - 2)), axes=(1,)))

    def AttenuationTomography(
        self, ScanIndex, RefIndex, c, d, fband, resolution=(
            0.6, 0.6), windowparams=(
            50, 0.1, 4)):

        from scipy.signal import tukey
        from numpy.fft import irfft2, irfft
        from matplotlib.pylab import plot, show, imshow

        dx = resolution[0]
        dy = resolution[1]

        Nx = int(round(self.NumberOfElements * self.Pitch / dx))
        Ny = int(round(d / dy))

        # G0 = np.zeros((int(self.NumberOfElements**2), 1), dtype=complex)
        # G1 = np.zeros((int(self.NumberOfElements**2), 1), dtype=complex)

        G0 = np.zeros((int(self.NumberOfElements**2), 1), dtype=float)
        G1 = np.zeros((int(self.NumberOfElements**2), 1), dtype=float)


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

        w = tukey(int(2 * windowparams[0]), windowparams[1])
        #
        f = np.linspace(0., self.SamplingFrequency / 2,
                        int(np.floor(len(w) * windowparams[2] / 2)) + 1)

        indf = (f >= fband[0]) & (f <= fband[1])

        f = f[indf]

        GG = []

        for m in range(self.NumberOfElements):

            xm = m * self.Pitch

            # for n in range(m):

            for n in range(self.NumberOfElements):

                xn = n * self.Pitch

                ind = int(round(2 * self.SamplingFrequency *
                                np.sqrt((0.5 * (xn - xm))**2 + d**2) / c))

                RR = rfft(np.real(self.AScans[RefIndex])[m, n, ind -
                                                         windowparams[0]:ind -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])], int(len(w) *
                                                                                    windowparams[2]))

                R = rfft(np.real(self.AScans[ScanIndex])[m, n, ind -
                                                         windowparams[0]:ind -
                                                         windowparams[0] +
                                                         int(2 *
                                                             windowparams[0])], int(len(w) *
                                                                                    windowparams[2]))

                # rr =np.real(self.AScans[RefIndex])[m,n,ind-windowparams[0]:ind-windowparams[0] + int(2*windowparams[0])]
                #
                # r = np.real(self.AScans[ScanIndex])[m,n,ind-windowparams[0]:ind-windowparams[0] + int(2*windowparams[0])]

                GG.append([R, RR])

                RR = RR[indf]
                R = R[indf]

                G = np.log(np.abs(R)/np.abs(RR))

                # G = np.log(np.abs(R * np.conj(RR) /
                #                   (RR * np.conj(RR) + 1e-2 * np.amax(np.abs(RR)))))

                pfit = np.polyfit(f, G, 1)

                G0[n + m * self.NumberOfElements, 0] = pfit[1]
                G1[n + m * self.NumberOfElements, 0] = -pfit[0]

                def xf(y):
                    return (xn-xm)*y/(2*d) + xm

                def yf(x):
                    return (x - xm)*2*d/(xn - xm)

                def xb(y):
                    return (xm-xn)*y/(2*d) + xn

                def yb(x):
                    return (x - xn)*2*d/(xm - xn)

                r = []

                pq = []

                for p in range(int(round(np.ceil(xm/dx))), int(round(np.floor(xf(d)/dx)))):

                    xp = dx*p

                    yp = yf(xp)

                    pq.append([p, int(np.floor(yp/dy))])

                    r.append(np.sqrt(xp**2 + yp**2))

                for q in range(Ny):

                    yq = dy*q

                    xq = xf(yq)

                    pq.append([int(np.floor(xq/dx)), q])

                    r.append(np.sqrt(xq**2 + yq**2))


                r = np.array(r)
                pq = np.array(pq)

                indr = np.argsort(r)

                pq = pq[indr,:]
                r = r[indr]


                r = np.diff(r)



                for i in range(pq.shape[0]-1):

                    B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Nx] = r[i]


                r = []

                pq = []

                for p in range(int(round(np.ceil(xf(d)/dx))), int(round(np.floor(xn/dx)))):

                    xp = dx*p

                    yp = yb(xp)

                    pq.append([p, int(np.floor(yp/dy))])

                    r.append(np.sqrt(xp**2 + yp**2))

                    # r.append(np.sqrt(xp**2 + yp**2))


                for q in range(Ny):

                    yq = dy*q

                    xq = xb(yq)

                    # print(yq)


                    pq.append([int(np.floor(xq/dx)), q])

                    # r.append(np.sqrt(xq**2 + yq**2))

                    r.append(np.sqrt(xq**2 + yq**2))




                r = np.array(r)
                pq = np.array(pq)

                indr = np.argsort(r)

                pq = pq[indr,:]
                r = r[indr]

                r = np.diff(r)

                # print(pq)
                # print(r)

                for i in range(pq.shape[0]-1):

                    # B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Ny] = r[i]

                    B[n + m * self.NumberOfElements, pq[i,0]+pq[i,1]*Nx] = r[i]






                # B[ n + m*Nx, :] =

                # B[n + m*Nx,:] = (np.exp(1j*kx*xm)*(np.exp(1j*(kx*(xn-xm) + ky*2*d)) - 1)/(kx*(xn-xm)/(2*d) + ky)).flatten()

                # B[n + m * self.NumberOfElements, :] = (np.exp(1j * kx * xm) * (np.exp(1j * (kx * 0.5 * (xn - xm) + ky * d)) - 1)) / (1j * (kx * 0.5 * (
                #     xn - xm) / d + ky)) - (np.exp(1j * kx * xn) * (np.exp(1j * (kx * 0.5 * (xm - xn) + ky * d)) - 1)) / (1j * (kx * 0.5 * (xm - xn) / d + ky))

        # indnz = (np.real(G0) > 0.).flatten()

        # G0 = np.nan_to_num(G0)
        #
        # G1 = np.nan_to_num(G1)

        #
        # G0 = G0[indnz]
        #
        # G1 = G1[indnz]

        # B = np.nan_to_num(B)

        # B = B[indnz,:]

        # print(np.amax(np.abs(B)))

        # A0 = np.linalg.lstsq(B, G0)[0].reshape(kxyshape)

        I0 = np.linalg.lstsq(B, G0, rcond=None)[0].reshape((Ny,Nx))

        I1 = np.linalg.lstsq(B, G1, rcond=None)[0].reshape((Ny,Nx))


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

        return I0, I1, G0, G1
