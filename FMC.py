from numpy import *
from scipy.signal import detrend,hilbert
from functools import reduce
# from multiprocessing import Pool

# import numpy as np
#
# import tensorflow as tf



#
# def ContactDelays(x,y,Elements,Pitch,Velocity):
#
#     p = tf.constant(Pitch)
#     c = tf.constant(Velocity)
#     m = tf.constant(Elements,dtype=tf.float32,shape=(1,1,len(Elements)))
#     # fs = tf.constant(SamplingFrequency)
#
#     xy = np.array(np.meshgrid(np.linspace(x[0],x[1],x[2]),np.linspace(y[0],y[1],y[2]))).reshape((2,x[2]*y[2]))
#     xy = np.repeat(xy[:,:,np.newaxis],len(Elements),axis=2)
#
#     XY = tf.placeholder(tf.float32,shape=xy.shape)
#
#     T = tf.divide(tf.sqrt(tf.add(tf.squared_difference(tf.multiply(m,p),XY[0,:,:]),tf.square(XY[1,:,:]))),c)
#
#     # T = T[0,:,:]
#     #
#     # indT = tf.round(tf.multiply(fs,
#
#     with tf.Session() as sess:
#
#         T = sess.run(T,{XY:xy})[0,:,:]
#
#     return T
#
#
#
# def TFMImage(AScans,Delays,SamplingFrequency):
#
#     Sh = np.array(AScans.shape)
#
#     N = np.min(Sh)
#     # L = np.max(Sh)
#
#
#     A = tf.constant(np.moveaxis(AScans,np.argmax(Sh),0).reshape((np.max(Sh),np.min(Sh)**2)),dtype=tf.complex64)
#
#     T = tf.constant(Delays,dtype=tf.float32)
#
#     fs = tf.constant(SamplingFrequency)
#
#
#     # Tn = tf.constant(Delays.transpose(),dtype=tf.float32)
#
#     # print(Tm)
#     # print(Tn)
#
#     #
#     N = tf.constant(N,dtype=tf.int32)
#
#     Lxy = Delays.shape[0]
#
#     # I = np.zeros(Delays)
#
#
#     # I = tf.Variable(tf.zeros([Lxy],np.complex64))
#
#     # L = tf.constant(AScans.shape[2],dtype=tf.int32)
#
#     # M,N,L = AScans.shape
#
#
#
#
#     # Ixy = tf.placeholder(tf.int32,shape=(Lxy,))
#     #
#     # ixy = np.array(range(Lxy)).astype(int32)
#     #
#     # Lxy = tf.constant(Lxy,dtype=tf.int32)
#
#
#
#
#     with tf.Session() as sess:
#
#         for nn in range(np.min(Sh)**2):
#





    # I = tf.reduce_sum(A[:,:,tf.round(tf.multiply(fs,tf.add(tf.reshape(Tm[ixy,:],[1,N]),tf.reshape(Tn[:,ixy],[N,1]))))],[0,1])

    # I = tf.add(tf.reshape(Tm[ixy,:],[1,N]),tf.reshape(Tn[:,ixy],[N,1]))

    # TT = tf.gather(T,Ixy)

    # indT = tf.squeeze(tf.round(tf.multiply(fs,tf.add(tf.reshape(TT,[Lxy,N]),tf.reshape(TT,[N,Lxy])))))

    # indT = tf.round(tf.multiply(fs,tf.add(tf.reshape(TT,))

    # I = tf.reduce_sum(tf.gather(A,indT),0)

    # I = tf.reduce_sum(A[:,:,tf.round(tf.multiply(fs,tf.reshape(Tm[ixy,:],[1,N])))),[0,1])



    # I = tf.add(tf.reshape(Tm[ixy,:],[1,32]),tf.reshape(Tn[:,ixy],[32,1]))

    # I = tf.reshape(Tm[ixy,:],[1,32])

    # I = tf.gather_nd()


    # with tf.Session() as sess:

        # I = sess.run(I,{ixy:range(D)})

        # I = sess.run(indT,{Ixy:ixy})

        # I = sess.run(I,{ixy:range(D)})

    # return I

#
# def WedgeDelays(x,y,m,Pitch,Velocity,WedgeParameters={'Velocity':2.33,'Height':3.7,'Angle':31.5}):
#
#



class LinearCapture:

    def __init__(self,fs,scans,p,c,N,wedgeparams={'Angle':31.52,'Height':5.02,'Velocity':2.34},probedelays=None):

        self.SamplingFrequency = fs
        self.Pitch = p
        self.Velocity = c
        self.NumberOfElements = N


        if probedelays is None:

            self.ProbeDelays = zeros((N,N))

        else:

            self.ProbeDelays = probedelays

        self.AScans = scans.copy()

        if wedgeparams is not None:

            self.WedgeParameters = wedgeparams.copy()

        # self.ProcessScans()


    def ProcessScans(self):

        L = self.AScans[0].shape[2]

        self.AScans = [hilbert(detrend(self.AScans[i],bp=tuple(range(0,L+int(L/10),int(L/10))))) for i in range(len(self.AScans))]

    # def BackPropagationImage():

    # def ProcessScans(self,ProbeDelays=None):
    #
    #     from scipy.signal import detrend
    #     from numpy.fft import rfft,ifft
    #
    #     if self.WedgeParameters is not None:
    #
    #         Ngate = int(round(2*self.WedgeParameters['Height']/self.WedgeParameters['Velocity']*self.SamplingFrequency))
    #
    #         T = int(round(amax(ProbeDelays)*self.SamplingFrequency))
    #
    #         M,N,L = self.AScans[0].shape
    #
    #         d = ProbeDelays.reshape((M,N,1))
    #
    #         for i in range(len(self.AScans)):
    #
    #             x = zeros((M,N,L+T))
    #
    #             x[:,:,Ngate:L] = self.AScans[i][:,:,Ngate:L]
    #
    #             X = rfft(detrend(x,axis=-1,bp=range(0,L+T,int((L+T)/5))),axis=-1)
    #
    #             f = linspace(0,12.5,X.shape[2])
    #
    #             self.AScans[i] = ifft(X*exp(2*pi*1j*f*d),n=2*(L+T)-1)[:,:,0:L]
    #


    def GetContactDelays(self,xrng,yrng,c=None):

        if c is None:

            c = self.Velocity

        self.Delays = [[[sqrt( (x-n*self.Pitch)**2 + y**2)/c for y in yrng] for x in xrng] for n in range(self.NumberOfElements)]

        self.xRange = xrng.copy()

        self.yRange = yrng.copy()

    def GetWedgeDelays(self,xrng,yrng):

        from scipy.optimize import minimize

        p = self.Pitch
        h = self.WedgeParameters['Height']

        cw = self.WedgeParameters['Velocity']

        cphi = cos(self.WedgeParameters['Angle']*pi/180.)
        sphi = sin(self.WedgeParameters['Angle']*pi/180.)

        c = self.Velocity


        f = lambda x,X,Y,n: sqrt((h + n*p*sphi)**2 + (cphi*n*p - x)**2)/cw + sqrt(Y**2 + (X - x)**2)/c
        J = lambda x,X,Y,n: -(cphi*n*p - x)/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x)**2)) - (X - x)/(c*sqrt(Y**2 + (X - x)**2))


        self.Delays = [[[minimize(f,x0=0.5*abs(x-n*self.Pitch*cphi),args=(x,y,n),method='BFGS',jac=J).fun for y in yrng] for x in xrng] for n in range(self.NumberOfElements)]

        self.xRange = xrng.copy()
        self.yRange = yrng.copy()

    def GetAdaptiveDelays(self,cw,hrng,ScanIndex,depth):

        from scipy.optimize import minimize
        from scipy.interpolate import interp1d

        xrng = linspace(0,self.NumberOfElements-1,self.NumberOfElements)*self.Pitch

        self.GetContactDelays(xrng,hrng,cw)

        # I = [self.ApplyTFM(i) for i in range(len(self.AScans))]
        #
        # dh = hrng[1]-hrng[0]
        #
        # hgrid = [argmax(II,axis=0)*dh + hrng[0] for II in I]


        I = self.ApplyTFM(ScanIndex)

        dh = hrng[1]-hrng[0]

        hgrid = argmax(abs(I),axis=0)*dh + hrng[0]

        h = interp1d(xrng,hgrid,bounds_error=False)

        f = lambda x,X,Y,n: sqrt((x-n*self.Pitch)**2 + h(x)**2)/cw  + sqrt((X-x)**2 + (Y-h(x))**2)/self.Velocity

        hmin = mean(hgrid[hgrid>0])

        # print(mean(hgrid[hgrid>0]))
        # print(min(hgrid[hgrid>0]))

        yrng = linspace(hmin,hmin+depth[0],int(round(depth[0]/depth[1])))

        xrng = linspace(0,xrng[-1],int(round(xrng[-1]/depth[1])))

        self.Delays = [[[float(minimize(f,x0=0.5*abs(x-n*self.Pitch),args=(x,y,n),method='BFGS').fun) if y>=h(x) else nan for y in yrng] for x in xrng] for n in range(self.NumberOfElements)]

        self.xRange = xrng

        self.yRange = yrng

    def ApplyTFM(self,ScanIndex):

        IX = len(self.Delays[0])
        IY = len(self.Delays[0][0])

        L = self.AScans[ScanIndex].shape[2]

        # delaytype = (len(self.Delays) == len(self.AScans))

        def PointFocus(ix,iy,A):

            Nd = len(self.Delays)

            # if delaytype:
            #
            #     return reduce(lambda x,y: x+y, (A[m,n,int(round((self.Delays[ScanIndex][m][ix][iy] + self.Delays[ScanIndex][n][ix][iy] + self.ProbeDelays[m,n])*self.SamplingFrequency))] if ((type(self.Delays[ScanIndex][n][ix][iy]) is float) and type(self.Delays[ScanIndex][m][ix][iy]) is float) else 0.+0j for n in range(Nd) for m in range(Nd)))
            #
            # else:

            # return reduce(lambda x,y: x+y, (A[m,n,int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy] + self.ProbeDelays[m,n])*self.SamplingFrequency))] if ((type(self.Delays[n][ix][iy]) is float or float64) and type(self.Delays[m][ix][iy]) is float or float64) else 0.+0j for n in range(Nd) for m in range(Nd)))

            return reduce(lambda x,y: x+y, (A[m,n,int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy] + self.ProbeDelays[m,n])*self.SamplingFrequency))] if ( isfinite(self.Delays[m][ix][iy]) and isfinite(self.Delays[n][ix][iy]) and  int(round((self.Delays[m][ix][iy] + self.Delays[n][ix][iy] + self.ProbeDelays[m,n])*self.SamplingFrequency)) <= L) else 0.+0j for n in range(Nd) for m in range(Nd)))


        return array([PointFocus(ix,iy,self.AScans[ScanIndex]) for ix in range(IX) for iy in range(IY)]).reshape((IX,IY)).transpose()


#
#
#
#
#
#
# class MatrixCapture:
#
#     def __init__(self,fs,scans,p,c,N):
#
#         self.SamplingFrequency = fs
#         self.Pitch = p
#         self.Velocity = c
#         self.NumberOfElements = N
#
#         self.AScans = scans.copy()
#
#
#     def ProcessScans(self,gate,BPParams=(1.,8.)):
#
#         from scipy.signal import detrend,tukey,firwin,fftconvolve,hilbert
#
#         # AScans are windowed (to remove main bang + inter-element talk), band-passed, back-wall estimated, and reference taken from back-wall.
#         # BPParms - tuple (Low Frequency, High Frequency, )
#
#         dt = 1/self.SamplingFrequency
#
#
#         L = self.AScans.shape[-1]
#
#         N = self.NumberOfElements[0]*self.NumberOfElements[1]
#
#         T = gate
#
#         self.AScans[:,:,0:int(T/dt)] = 0.
#
#         self.AScans[:,:,int(T/dt)::] = detrend(self.AScans[:,:,int(T/dt)::],bp=[0,int((L-(T/dt))/3),L-(int(T/dt))])
#
#
#         h = firwin(L-1,[BPParams[0]/(self.SamplingFrequency*0.5),BPParams[1]/(self.SamplingFrequency*0.5)],pass_zero=False)
#
#
#         for m in range(N):
#             for n in range(N):
#
#                 self.AScans[m,n,:] = fftconvolve(self.AScans[m,n,:],h,mode='same')
#
#
#         self.AScans = hilbert(self.AScans,axis=2)
#
#
#
#     def GetDelays(self,xrng,yrng,zrng):
#
#         self.Delays = [[[[sqrt( (x-nx*self.Pitch)**2 + (y-ny*self.Pitch)**2 + z**2)/self.Velocity  for z in zrng] for y in yrng] for x in xrng] for ny in range(self.NumberOfElements[1]) for nx in range(self.NumberOfElements[0])]
#
#
#     def ApplyTFM(self):
#
#         IX = len(self.Delays[0])
#         IY = len(self.Delays[0][0])
#         IZ = len(self.Delays[0][0][0])
#
#         def PointFocus(ix,iy,iz):
#
#             Nd = len(self.Delays)
#
#             I = reduce(lambda x,y: x+y, (self.AScans[m,n,int(round((self.Delays[m][ix][iy][iz] + self.Delays[n][ix][iy][iz])*self.SamplingFrequency))] for n in range(Nd) for m in range(Nd)))
#
#             return I
#
#
#         self.TFMImage = array([PointFocus(ix,iy,iz) for ix in range(IX) for iy in range(IY) for iz in range(IZ)]).reshape((IX,IY,IZ))
