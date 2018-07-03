class BallScan:

    def __init__(self,File,Diameter,SamplingFrequency=100,PRF=100,Offset=4610,BallSpeed=720.,c=5.89):


        self.WaveSpeed = c

        self.File = File

        self.Offset = Offset

        self.BallSpeed = BallSpeed

        self.PRF = PRF

        self.SamplingFrequency = SamplingFrequency

        self.Offset = Offset

        self.Diameter = Diameter

        self.LoadScan()


    def LoadScan(self):

        from numpy import array,fromstring
        import re

        f = open(self.File, 'rb')

        l = f.readlines()

        f.close()


        N = int(l[1].decode('utf-8').strip('\n').split('\t')[1])-3

        b = l[5].decode('utf-8').strip('\n').split('\t')

        self.WingRange = (float(b[0]),float(b[1]),N)

        self.BallRange = (0,N*self.BallSpeed/self.PRF,N)

        a = [fromstring(ll,float,sep='\t') for ll in l[10:-2]]

        a = array(a)

        a = a.reshape((N,len(a[0])))


        self.AScans = a



    def ImageBall(self,R,dr):

        from numpy import array, linspace, meshgrid, hstack, floor, pi, sin, cos, arctan2, arccos
        from scipy.interpolate import griddata
        from scipy.signal import hilbert

        dsamp = max([int(floor(2*dr*self.SamplingFrequency/self.WaveSpeed)),1])

        Ip = abs(hilbert(self.AScans[:,self.Offset::dsamp],axis=-1))


        phi = linspace(self.WingRange[0],self.WingRange[1],self.WingRange[2])*(pi/180)
        theta = linspace(self.BallRange[0],self.BallRange[1],self.BallRange[2])*(pi/180)

        Ip = Ip.reshape((Ip.shape[-1],theta,phi))

        dtheta = theta[1]-theta[0]
        dphi = phi[1]-phi[0]

        r = linspace(0,R,Ip.shape[1])

        dtheta = theta[1]-theta[0]
        dphi = phi[1]-phi[0]
        dr = r[1]-r[0]


        # Ip = Ip.ravel()

        # xyz = array([[rr*cos(theta[i])*sin(phi[i]),rr*sin(theta[i])*sin(phi[i]),rr*cos(phi[i])] for i in range(len(phi)) for rr in r])

        l = linspace(-r[-1],r[-1],int(2*len(r)))

        x = l
        y = l
        z = l

        I = [[[Ip[int(round(sqrt(xx**2+yy**2+zz**2)/dr)),int(round(arccos(zz/sqrt(xx**2+yy**2+zz**2))/dth)),int(round(arctan2(yy,xx)/dphi))] for xx in x] for yy in y ] for zz in z]

        self.BallImage = I

        self.Coordinates = (x,y,z)

        # X,Y,Z = meshgrid(l,l,l)

        # I = zeros((len(l),len(l),len(l)))

        # for ix in range(len(x)):
        #     for iy in range(len(y)):
        #         for iz in range(len(z)):
        #
        #             ith = arccos(z[iz]/sqrt(x[ix]**2 + y[iy]**2 + z[iz]**2))
        #

        # S = X.shape
        # L = S[0]*S[1]*S[2]
        #
        # XYZ = hstack((X.reshape((L,1)),Y.reshape((L,1)),Z.reshape((L,1))))
        #
        # Ip = self.AScans.ravel()
        #
        # I = griddata(xyz,Ip,XYZ,fill_value=0.0)
        #
        # self.Image = I.reshape(S)
