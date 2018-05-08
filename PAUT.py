import numpy as np
from numpy.fft import rfft, fftshift, ifft, ifftn, ifftshift, irfft
from Signal import ShiftSignal

def CalculateOffsets(angles, Tdelay, cw = 2.33, cs = 3.24):

    """ Calculates index offsets starting from first angle in angles as zero
        offset, using wedge and sample velocities cw and cs, refracted angles and wedge delay for
        first angle Tdelay """

    angles = angles*np.pi/180.
    anglesw = np.arcsin(cw*np.sin(angles)/cs)

    h = 0.5*cw*Tdelay*np.cos(anglesw[0])

    Offsets = h*np.tan(anglesw)

    return Offsets


class Sweep:

    def __init__(self, scans, samplingfreq, angles, pitch, numberofelements, c, indexoffsets=None, rangeoffsets=None):

        from copy import deepcopy

        self.Scans = deepcopy(scans)
        self.SamplingFrequency = samplingfreq
        self.Angles = angles*(np.pi/180.)
        self.Pitch = pitch
        self.NumberOfElements = numberofelements

        self.WaveSpeed = c

        self.IndexOffsets = indexoffsets
        self.RangeOffsets = rangeoffsets

        if self.IndexOffsets is not None:

            self.x0 = self.IndexOffsets

        if self.RangeOffsets is not None:

            self.x0 = self.x0 + self.RangeOffsets*np.sin(self.Angles)

            self.x0 = self.x0 - 0.5*(np.amin(self.x0)+np.amax(self.x0))
            self.y0 = self.RangeOffsets*np.cos(self.Angles)


    def FWtoRF(self,fc,BW):

        """ Uses centre frequency fc, and fractional bandwidth BW, to approximately
            undo full wave rectification and reproduce estimate of original RF signal.
            Modifies self.Scans """


        sigma = fc*BW/(2*np.sqrt(2*np.log(2)))

        Nt = self.Scans[0].shape[1]

        t = np.linspace(0.,Nt-1,Nt)/self.SamplingFrequency

        f = np.linspace(0., self.SamplingFrequency/2, int(np.floor(Nt/2) + 1))

        H = np.exp(-(f/(2*sigma))**2).reshape(1,-1)


        h = np.cos(2*np.pi*fc*t).reshape(1,-1)


        self.Scans = [irfft(rfft(s)*H)*h for s in self.Scans]



    def GetImage(self, ScanIndex, Resolution):

        from scipy.interpolate import griddata

        c = self.WaveSpeed

        a = np.abs(self.Scans[ScanIndex])

        r = np.linspace(0.,c*a.shape[1]/(2*self.SamplingFrequency), a.shape[1]).reshape(1,-1)

        ang = self.Angles.reshape(-1,1)

        if self.IndexOffsets is None:

            x = r*np.sin(ang)


            # y = r/np.cos(ang)

            y = r*np.cos(ang)

        else:

            x = r*np.sin(ang) + self.x0.reshape(-1,1)

            # y = r/np.cos(ang) + self.y0.reshape(-1,1)

            y = r*np.cos(ang) + self.y0.reshape(-1,1)

        xmin = np.min(x)
        xmax = np.max(x)

        xgrid = np.linspace(xmin,xmax,int(np.round(xmax-xmin)/Resolution))

        ymin = np.min(y)
        ymax = np.max(y)

        ygrid = np.linspace(ymin,ymax,int(np.round(ymax-ymin)/Resolution))

        xygrid = np.meshgrid(xgrid,ygrid)

        I = griddata((x.flatten(), y.flatten()), a.flatten(), (xygrid[0].flatten(), xygrid[1].flatten()), method='linear', fill_value=0., rescale = False)


        I = I.reshape(xygrid[0].shape)

        return I, (xmin,xmax,ymax,ymin)



    def GetWavenumberImage(self, ScanIndex, L, fband, Resolution, Npad=4):

        from scipy.interpolate import griddata

        c = self.WaveSpeed

        A = rfft(self.Scans[ScanIndex], int(Npad*self.Scans[ScanIndex].shape[1]))


        f = np.linspace(0.,self.SamplingFrequency/2,A.shape[-1]).reshape(1,-1)


        indf = (f>=fband[0])&(f<=fband[1])
        indf = np.array(indf).flatten()


        A = A[:,indf]

        w = 2*np.pi*f[:,indf]


        angrad = self.Angles.reshape(-1,1)


        kx = 2.*w*np.sin(angrad)/c + 0j

        ky = 2.*w*np.cos(angrad)/c + 0j

        # ky = 2.*np.sqrt((w/c)**2 - (kx/2)**2 +0j)

        # print(kx.shape)

        # print(self.x0.reshape(-1,1).shape)



        if self.IndexOffsets is None:


            Nx = int(np.round(L[0]/self.Pitch))

            kxgrid = 2*np.pi*np.linspace(-1/(2*self.Pitch),1/(2*self.Pitch), Nx)



        else:

            A = A*np.exp(-1j*kx*self.x0.reshape(-1,1))*np.exp(-1j*ky*self.y0.reshape(-1,1))

            p = np.amin(np.diff(self.x0))

            p = np.pi/np.amax(np.abs(kx))

            # p = self.Pitch

            Nx = int(np.round(L[0]/p))


            kxgrid = 2*np.pi*np.linspace(-1/(2*p), 1/(2*p), Nx)

            # kxgrid = 2*np.pi*np.linspace(-1/(2*self.Pitch),1/(2*self.Pitch), Nx)


        # print(np.amin(np.diff(self.x0)))


        # ky = ky[np.abs(np.real(ky))>=0.]

        # kxmax = np.amax(np.abs(kx))/2
        #
        # dx = np.pi/kxmax
        #
        # print(dx)
        #
        #
        # Nx = int(np.round(L[0]/dx))
        #
        # # kxgrid = 2*np.pi*np.linspace(-1/(2*dx), 1/(2*dx), Nx)
        #
        # kxgrid = np.linspace(-kxmax,kxmax, Nx)




        # Nx = int(np.round(L[0]/self.Pitch))

        # dy = c/(2*fband[1])

        dy = np.pi/np.amax(np.abs(ky))

        # print(dy)

        Ny = int(np.round(L[1]/(2*dy)))



        # kxgrid = 2*np.pi*np.linspace(-1/(2*self.Pitch),1/(2*self.Pitch), Nx) + 0j


        kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny) + 0j

        kgrid = np.meshgrid(kxgrid, kygrid)



        I = griddata((kx.flatten(), ky.flatten()), A.flatten(), (kgrid[0].flatten(),kgrid[1].flatten()), method = 'linear', fill_value=0+0j, rescale=False).reshape(kgrid[0].shape)


        Nx = int(np.round(L[0]/Resolution))

        xzeropad = np.zeros((I.shape[0],int(np.round((Nx - I.shape[1]-2)/2))), dtype=complex)


        I = ifft(ifftshift(np.hstack((xzeropad, I, xzeropad)), axes=(1,)))


        Ny = int(np.round(L[1]/(2*Resolution)))


        return fftshift(ifft(I,axis=0,n=2*Ny - 2),axes=(1,))


        # else:
        #
        #     kxgrid = 2*np.pi*np.linspace(0.,1/(2*self.Pitch), Nx)
        #
        #     xx = np.abs(self.IndexOffsets)
        #
        #     xx = (xx-np.amax(xx)).reshape(kx.shape)
        #
        #     A = A*np.exp(-2j*kx*xx)
        #
        #     kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny)
        #
        #     kgrid = np.meshgrid(kxgrid, kygrid)
        #
        #     I = griddata((kx.flatten(), ky.flatten()), A.flatten(), (kgrid[0].flatten(),kgrid[1].flatten()), method = 'cubic', fill_value=0+0j, rescale=True).reshape(kgrid[0].shape)
        #
        #     Nx = int(np.round(L[0]/(2*Resolution)))
        #
        #     Ny = int(np.round(L[1]/(2*Resolution)))
        #
        #     return ifftn(I,axes=(0,1),s=(2*Nx - 2, 2*Ny - 2))
