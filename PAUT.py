import numpy as np
from numpy.fft import rfft, fftshift, ifft, ifftn, ifftshift, irfft,ifft2
from Signal import ShiftSignal

# def CalculateOffsets(angles, Tdelay, cw = 2.33, cs = 3.24):
#
#     """ Calculates index offsets starting from first angle in angles as zero
#         offset, using wedge and sample velocities cw and cs, refracted angles and wedge delay for
#         first angle Tdelay """
#     angles = angles*np.pi/180.
#     anglesw = np.arcsin(cw*np.sin(angles)/cs)
#
#     h = 0.5*cw*Tdelay*np.cos(anglesw[0])
#
#     Offsets = h*np.tan(anglesw)
#
#     return Offsets
class Sweep:

    def __init__(self, scans, samplingfreq, angles, elements,c, pitch, wedgeparams=None, depthoffsets=None):

        """

            A class for rendering standard phased array sweep data.

        """

        from copy import deepcopy



        self.Scans = deepcopy(scans)
        self.SamplingFrequency = samplingfreq
        self.Angles = [np.deg2rad(a) for a in angles]
        # self.Pitch = pitch
        # self.NumberOfElements = numberofelements

        self.Elements = deepcopy(elements)
        self.WaveSpeed = c
        self.Pitch = pitch

        if depthoffsets is None:

            self.DepthOffsets = [np.zeros(len(self.Angles[i])) for i in range(len(self.Angles))]

        else:

            self.DepthOffsets = deepcopy(depthoffsets)

        if wedgeparams is None:

            self.GetContactIndexOffsets()




    def ProcessScans(self,Nclip):

        from scipy.signal import tukey, detrend

        for i in range(len(self.Angles)):

            a = self.Scans[i].swapaxes(1,2)

            Nt = a.shape[-1]

            a = detrend(tukey(Nt,Nclip/Nt)*a)

            a[:,:,0:Nclip] = 0.

            self.Scans[i] = a.swapaxes(2,1)


    def InterpolateAngles(self, newangles):

        from scipy.interpolate import interp1d

        for i in range(len(self.Angles)):

            newangles[i] = np.deg2rad(newangles[i])

            intp = interp1d(self.Angles[i], self.Scans[i],axis=0,fill_value='extrapolate')

            self.Scans[i] = intp(newangles[i])

            self.Angles[i] = newangles[i]

            self.DepthOffsets[i] = np.zeros(newangles[i].shape)


    def GetWedgeIndexOffsets(self, elements, wedgeangle, h, wedgeoffset=0., pitch=0.6, cw=2.33):

        """ Allows index offsets to be estimated from wedge parameters
        (h - height of first element, wedgeangle - angle of wedge,
        cw - wedge velocity, wedgeoffset - offset of wedge from center of imaging domain),
        and element pitch, as well as the start element and number of elements
        in each group:
        elements = [(group 1 start element, group 1 end element), ...]"""

        ioff = []

        wa = np.deg2rad(wedgeangle)


        for i in range(len(self.Angles)):

            aw = np.arcsin(cw*np.sin(self.Angles[i])/self.WaveSpeed)

            le = pitch*(0.5*(elements[i][1] - elements[i][0]) + elements[i][0])

            xe = le*np.cos(wa) + (h+le*np.sin(wa))*np.tan(aw)

            ioff.append(xe - wedgeoffset)

        self.IndexOffsets = ioff


    def GetContactIndexOffsets(self):


        self.IndexOffsets = []

        e0 = [self.Elements[i][0] for i in range(len(self.Elements))]
        e1 = [self.Elements[i][1] for i in range(len(self.Elements))]


        xarrmin = min(e0)*self.Pitch

        xarrmax = max(e1)*self.Pitch

        # xarrmean = 0.5*(xarrmax - xarrmin)

        R = 0.5*self.WaveSpeed*self.Scans[0].shape[1]/self.SamplingFrequency

        X = (xarrmax - xarrmin)*0.5


        Xmin = 0.

        Xmax = 0.

        def xcentre(x):

            return x - X - xarrmin


        for i in range(len(self.Angles)):

            Xmin = min([Xmin,xcentre(e0[i]*self.Pitch+R*np.amin(np.sin(self.Angles[i])))])

            Xmax = max([Xmax,xcentre(e1[i]*self.Pitch+R*np.amax(np.sin(self.Angles[i])))])

        self.xRange = (Xmin,Xmax)

        self.yRange = (0.,R)

        for i in range(len(self.Angles)):

            self.IndexOffsets.append(np.ones(self.Angles[i].shape)*(xcentre(np.mean(self.Pitch*np.array([e0[i],e1[i]])))))



    def CombineGroups(self):

        A = 0 +0j

        N = self.Scans[0].shape[1]

        for i in range(len(self.IndexOffsets)):

            AA = rfft(self.Scans[i],axis=1,n=2*N).swapaxes(0,2)

            w = 2*np.pi*np.linspace(0.,self.SamplingFrequency/2,AA.shape[1]).reshape(-1,1)

            AA = AA*np.exp(-2j*self.IndexOffsets[i].reshape(1,-1)*w*np.sin(self.Angles[i].reshape(1,-1))/self.WaveSpeed)

            A += AA

        self.Scans = [irfft(A.swapaxes(0,2),axis=1)[:,0:N,:]]

        self.Angles = [self.Angles[0]]

        self.IndexOffsets = [np.zeros(len(self.Angles[0]))]


    def GetPAUTImage(self, ScanIndex, Resolution):

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

        I = griddata((x.flatten(), y.flatten()), a.flatten(), (xygrid[0].flatten(), xygrid[1].flatten()), method='nearest', fill_value=0., rescale = False)




        I = I.reshape(xygrid[0].shape)

        return I, (xmin,xmax,ymax,ymin)


    # def GetAPAUTImage(self, ScanIndex, fband, Resolution, Npad=4, frequencydependent=False):
    #
    #     from scipy.interpolate import griddata
    #     from copy import deepcopy
    #
    #     c = self.WaveSpeed
    #
    #
    #     I = np.array([0+0j])
    #
    #     x = np.array([0.])
    #
    #     y = np.array([0.])
    #
    #
    #     for i in range(len(self.Angles)):
    #
    #
    #         A = rfft(self.Scans[i][:,:,ScanIndex], int(Npad*self.Scans[i][:,:,ScanIndex].shape[1]))
    #
    #
    #         f = np.linspace(0.,self.SamplingFrequency/2,A.shape[-1]).reshape(1,-1)
    #
    #
    #         indf = (f>=fband[0])&(f<=fband[1])
    #         indf = np.array(indf).flatten()
    #
    #
    #         A = A[:,indf]
    #
    #         w = 2*np.pi*f[:,indf]
    #
    #
    #         # angrad = self.Angles[i].reshape(-1,1)
    #
    #         a = deepcopy(self.Angles[i].reshape(-1,1))
    #
    #         aavg = a[int(len(a)/2)]
    #
    #         a = a - aavg
    #
    #         x0 = deepcopy(self.IndexOffsets[i])
    #
    #         x0avg = x0[int(len(x0)/2)]
    #
    #         x0 = x0 - x0avg
    #
    #         x0 = x0.reshape(-1,1)
    #
    #
    #         kx = 2.*w*np.sin(a)/c + 0j
    #
    #         ky = 2.*w*np.cos(a)/c + 0j
    #
    #
    #         dx = np.pi/np.amax(np.abs(kx))
    #
    #         dy = np.pi/np.amax(np.abs(ky))
    #
    #
    #
    #
    #         # if self.IndexOffsets[i] is not None:
    #
    #         if frequencydependent:
    #
    #             A = -(A*np.exp(-1j*kx*x0)*np.exp(-1j*ky*self.DepthOffsets[i].reshape(-1,1)))/(w**2)
    #
    #         else:
    #
    #             A = A*np.exp(-1j*kx*x0)*np.exp(-1j*ky*self.DepthOffsets[i].reshape(-1,1))
    #
    #
    #         R = c*(len(self.Scans[i][int(len(a)/2),:])/self.SamplingFrequency)/2.
    #
    #         l = 0.5*(self.Elements[i][1]-self.Elements[i][0])*self.Pitch
    #
    #         xx = np.arange(-l-x0[0]+R*np.sin(a[0]),l+x0[-1]+R*np.sin(a[-1]),dx)
    #
    #         yy = np.arange(0.,R,dy)
    #
    #
    #         Nx = int(np.round((np.amax(xx)-np.amin(xx))/dx))
    #
    #         Ny = int(np.round(R/(2*dy)))
    #
    #
    #         kxgrid = 2*np.pi*np.linspace(-1/(2*dx),1/(2*dx), Nx) + 0j
    #
    #         kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny) + 0j
    #
    #         kgrid = np.meshgrid(kxgrid, kygrid)
    #
    #         xx,yy = np.meshgrid(xx,yy)
    #
    #         xx = xx.flatten()
    #
    #         yy = yy.flatten()
    #
    #         x = np.concatenate((x,xx*np.cos(aavg) + yy*np.sin(aavg) + x0avg))
    #
    #         y = np.concatenate((y,-xx*np.sin(aavg) + yy*np.cos(aavg)))
    #
    #
    #         I = np.concatenate((I,ifftshift(ifft2(griddata((kx.flatten(), ky.flatten()), A.flatten() , (kgrid[0].flatten(),kgrid[1].flatten()), method = 'linear', fill_value=0+0j, rescale=False).reshape((Ny,Nx)),axes=(1,0),s=(Nx,2*Ny+1)),axes=(1,)).flatten()))
    #
    #
    #     X,Y = np.meshgrid(np.arange(self.xRange[0],self.xRange[1],Resolution), np.arange(self.yRange[0],self.yRange[1],Resolution))
    #
    #     print(x.shape)
    #     print(y.shape)
    #
    #     print(I.shape)
    #
    #
    #     I = griddata((x[1::],y[1::]), I[1::], (X.flatten(),Y.flatten()), method='linear', fill_value=0+0j).reshape(X.shape)
    #
    #     return I
    #
    #



    def GetAPAUTImage(self, ScanIndex, L, fband, Resolution, Npad=4, frequencydependent=False):

        from scipy.interpolate import griddata

        c = self.WaveSpeed

        B = np.array([0+0j])

        Ky = np.array([0+0j])

        Kx = np.array([0+0j])


        for i in range(len(self.Angles)):


            A = rfft(self.Scans[i][:,:,ScanIndex], int(Npad*self.Scans[i][:,:,ScanIndex].shape[1]))


            f = np.linspace(0.,self.SamplingFrequency/2,A.shape[-1]).reshape(1,-1)


            indf = (f>=fband[0])&(f<=fband[1])
            indf = np.array(indf).flatten()


            A = A[:,indf]

            w = 2*np.pi*f[:,indf]


            # angrad = self.Angles[i].reshape(-1,1)

            a = self.Angles[i].reshape(-1,1)


            kx = 2.*w*np.sin(a)/c + 0j

            ky = 2.*w*np.cos(a)/c + 0j




            # if self.IndexOffsets[i] is not None:

            if frequencydependent:

                A = -(A*np.exp(-1j*kx*self.IndexOffsets[i].reshape(-1,1))*np.exp(-1j*ky*self.DepthOffsets[i].reshape(-1,1)))/(w**2)

            else:

                A = A*np.exp(-1j*kx*self.IndexOffsets[i].reshape(-1,1))*np.exp(-1j*ky*self.DepthOffsets[i].reshape(-1,1))




            Kx = np.concatenate((Kx,kx.flatten()))

            Ky = np.concatenate((Ky,ky.flatten()))

            B = np.concatenate((B,A.flatten()))



        #
        Kx = Kx[1::]
        Ky = Ky[1::]

        B = B[1::]


        dx = np.pi/np.amax(np.abs(Kx))

        dy = np.pi/np.amax(np.abs(Ky))

        Nx = int(np.round(L[0]/dx))

        Ny = int(np.round(L[1]/(2*dy)))


        kxgrid = 2*np.pi*np.linspace(-1/(2*dx),1/(2*dx), Nx) + 0j

        kygrid = 2*np.pi*np.linspace(0,1/(2*dy), Ny) + 0j

        kgrid = np.meshgrid(kxgrid, kygrid)


        I = griddata((Kx, Ky), B , (kgrid[0].flatten(),kgrid[1].flatten()), method = 'linear', fill_value=0+0j, rescale=False).reshape(kgrid[0].shape)


        Nx = int(np.round(L[0]/Resolution))

        xzeropad = np.zeros((I.shape[0],int(np.round((Nx - I.shape[1]-2)/2))), dtype=complex)


        I = ifft(ifftshift(np.hstack((xzeropad, I, xzeropad)), axes=(1,)), axis=1)


        Ny = int(np.round(L[1]/(2*Resolution)))


        return fftshift(ifft(I,axis=0,n=2*Ny - 2),axes=(1,))
