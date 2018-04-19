import numpy as np
from numpy.fft import *
from Signal import ShiftSignal


class Sweep:

    def __init__(self, scans, samplingfreq, angles, pitch, numberofelements, c, indexoffsets=None):

        from copy import deepcopy

        self.Scans = deepcopy(scans)
        self.SamplingFrequency = samplingfreq
        self.Angles = angles*(np.pi/180.)
        self.Pitch = pitch
        self.NumberOfElements = numberofelements

        self.WaveSpeed = c

        self.IndexOffsets = indexoffsets





        # if type(c) is float:
        #
        #     self.WaveSpeed = {'Wedge': c, 'Piece': c}
        #     self.IndexOffsets = np.zeros(len(self.Angles))
        #
        # elif type(c) is dict:
        #
        #     self.WaveSpeed = c
        #     self.IndexOffsets = indexoffsets



    def ZeroTime(self):

        """ Moves the zero time position to the centre of the aperture for all
        scans """

        L = self.Scans[0].shape[1]

        for i in range(len(self.Scans)):

            for n in range(len(self.Angles)):

                self.Scans[i][n,:] = ShiftSignal(self.Scans[i][n,:],-(self.NumberOfElements-1)*self.Pitch*np.sin(np.abs(self.Angles[n]))/self.WaveSpeed['Wedge'], self.SamplingFrequency)

                # d = int(np.round(self.SamplingFrequency*(self.NumberOfElements - 1)*self.Pitch*np.sin(np.abs(self.Angles[n]))/self.WaveSpeed['Wedge']))

                # print(d)
                #
                # print(self.Scans[i][n].shape)
                #
                # print(self.Scans[i][n,d::].shape)

                # s = np.zeros((len(self.Angles) , L))
                #
                # s[n,0:(L-d)] = self.Scans[i][n,d::]

                # self.Scans[i][n,:] = np.concatenate((self.Scans[i][n,d::],self.Scans[i][n,-1]*np.ones(d)))



    def GetImage(self, ScanIndex, Resolution):

        from scipy.interpolate import griddata

        c = self.WaveSpeed

        a = self.Scans[ScanIndex]

        r = np.linspace(0.,c*a.shape[1]/(2*self.SamplingFrequency), a.shape[1]).reshape(1,-1)

        ang = self.Angles.reshape(-1,1)

        x = r*np.sin(ang)
    # y = r/np.cos(angrad)

        y = r*np.cos(ang)

        xmin = np.min(x)
        xmax = np.max(x)

        xgrid = np.linspace(xmin,xmax,int(np.round(xmax-xmin)/Resolution))

        ymin = np.min(y)
        ymax = np.max(y)

        ygrid = np.linspace(ymin,ymax,int(np.round(ymax-ymin)/Resolution))

        xygrid = np.meshgrid(xgrid,ygrid)

        I = griddata((x.flatten(), y.flatten()), a.flatten(), (xygrid[0].flatten(), xygrid[1].flatten()), method='cubic', fill_value=0., rescale = True)


        I = I.reshape(xygrid[0].shape)

    # I[I>100.]=100.

    # I =

        return I

#

    def GetBackPropagationImage(self, ScanIndex, L, fband, Resolution):

        # from scipy.interpolate import interp1d

        c = self.WaveSpeed

        A = rfft(self.Scans[ScanIndex])

        # print(a.shape)

        # print(A.shape)

        f = np.linspace(0.,self.SamplingFrequency/2,A.shape[-1]).reshape(1,-1)

        indf = (f>=fband[0])&(f<=fband[1])
        indf = np.array(indf).flatten()

        ang = self.Angles.reshape(-1,1)


        # print(indf.shape)

        A = A[:,indf]

        A = A/np.cos(ang)

        w = 2*np.pi*f[:,indf]

        dy = c/(2*fband[1])

        Ny = int(np.round(L[1]/(2*dy)))

        kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny)

        ky = 2*w*np.cos(ang)/self.WaveSpeed['Piece']

        II = np.array([np.interp(kygrid, ky[i,:], A[i,:], 0+0j, 0+0j) for i in range(A.shape[0])])

        Ny = int(np.round(L[1]/(2*Resolution)))

        Nx = int(np.round(L[0]/Resolution))

        II = ifft(II,n=2*Ny - 2)

        y = np.linspace(0.,L[1],II.shape[1])

        x = y.reshape(1,-1)*np.tan(ang)

        xgrid = np.linspace(-L[0]/2, L[0]/2, Nx)

        print(y.shape)

        print(x.shape)
        print(II.shape)

        I = np.abs(np.array([np.interp(xgrid, x[:,i], II[:,i], 0.+0j, 0.+0j) for i in range(II.shape[1])]))

        # Nx = int(np.round(L[0]/Resolution))

        # return np.abs(ifftn(fftshift(I,axes=(1,)),axes=(1,0),s=(Nx,2*Ny - 2)))

        return I

    # return np.abs(ifftn(I, s=(2*Ny - 2, Ny), axes=(1,0)))

    def GetWavenumberImage(self, ScanIndex, L, fband, Resolution):

        from scipy.interpolate import griddata

        c = self.WaveSpeed

        A = rfft(self.Scans[ScanIndex], 4*self.Scans[ScanIndex].shape[1])

        # print(a.shape)

        # print(A.shape)

        f = np.linspace(0.,self.SamplingFrequency/2,A.shape[-1]).reshape(1,-1)


        indf = (f>=fband[0])&(f<=fband[1])
        indf = np.array(indf).flatten()

        # print(indf.shape)

        A = A[:,indf]

        w = 2*np.pi*f[:,indf]


        angrad = self.Angles.reshape(-1,1)


        kx = 2.*w*np.sin(angrad)/c

        ky = 2.*np.sqrt((w/c)**2 - (kx/2)**2 + 0+0j)

        ky = ky[np.abs(np.real(ky))>=0.]


        # ky = 2*w*np.cos(angrad)/c


        # AA = A/np.cos(angrad)

        Nx = int(np.round(L[0]/self.Pitch))

        dy = c/(2*fband[1])

        Ny = int(np.round(L[1]/(2*dy)))

        if self.IndexOffsets is None:


            kxgrid = 2*np.pi*np.linspace(-1/(2*self.Pitch),1/(2*self.Pitch), Nx)


            kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny)

            kgrid = np.meshgrid(kxgrid, kygrid)

            I = griddata((kx.flatten(), ky.flatten()), A.flatten(), (kgrid[0].flatten(),kgrid[1].flatten()), method = 'cubic', fill_value=0+0j, rescale=True).reshape(kgrid[0].shape)

                # I = I*np.exp(-1j*kgrid[0]*L[0])

            Nx = int(np.round(L[0]/Resolution))

            xzeropad = np.zeros((I.shape[0],int(np.round((Nx - I.shape[1]-2)/2))), dtype=complex)

                # I = fftshift(ifft(ifftshift(np.hstack((xzeropad, I, xzeropad)), axes=(1,))), axes=(1,))

            I = ifft(ifftshift(np.hstack((xzeropad, I, xzeropad)), axes=(1,)))


            Ny = int(np.round(L[1]/(2*Resolution)))

                # return np.abs(ifftn(fftshift(I,axes=(1,)),axes=(1,0),s=(Nx,2*Ny - 2)))

            return fftshift(ifft(I,axis=0,n=2*Ny - 2),axes=(1,))


        else:

            kxgrid = 2*np.pi*np.linspace(0.,1/(2*self.Pitch), Nx)

            xx = np.abs(self.IndexOffsets)

            xx = (xx-np.amax(xx)).reshape(kx.shape)

            A = A*np.exp(-2j*kx*xx)

            kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny)

            kgrid = np.meshgrid(kxgrid, kygrid)

            I = griddata((kx.flatten(), ky.flatten()), A.flatten(), (kgrid[0].flatten(),kgrid[1].flatten()), method = 'cubic', fill_value=0+0j, rescale=True).reshape(kgrid[0].shape)

            Nx = int(np.round(L[0]/(2*Resolution)))

            Ny = int(np.round(L[1]/(2*Resolution)))

            return ifftn(I,axes=(0,1),s=(2*Nx - 2, 2*Ny - 2))
