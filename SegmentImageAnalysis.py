from numpy import *
from numpy.fft import *
from matplotlib.pylab import *
from misc import *
from scipy.ndimage import *
# from skimage.filters import *
import os
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import pickle
from scipy.misc import imresize

def ImgToArray(I):

    from numpy import sum as asum
    from numpy import amax,array,amin
    from numpy.linalg import norm

    bintodec = array([256**2,256,1])

    II = asum(bintodec*I[:,:,0:3],axis=-1)

    II = II/(norm(II.ravel()))

    # II[II==II[0,0]]=0
    #
    # II[II>0] = II[II>0]-amin(II[II>0])
    #
    # II = II*(II>0.)
    #
    # II = II/amax(II)

    return II.astype(float)

def ReNormalize(I):

    from numpy import sum as asum
    from numpy import amax,array,amin

    II = I.copy()

    II = II-amin(II[II>0])

    II = II*(II>=0.).astype(float)

    II = II/amax(II)

    return II

def GetImageList(pth):

    # Converts Folder of Images To Ordered List

    d = os.listdir(pth)

    d = [dd for dd in d if dd !='.DS_Store']

    if d[0].endswith('.png'):

        d.sort()

        l = [ ImgToArray(imread(pth+dd)) for dd in d]

    else:

        d = [int(dd) for dd in d]

        d.sort()

        l = [ ImgToArray(imread(pth+str(dd)+'/'+f)) for dd in d for f in os.listdir(pth+str(dd)) if f.endswith('.png')]



    return l


class Weld:

    def __init__(self,files,Region=[[5.,75.],[35.,105.]],filerange=None,RGBInd=0,Ext='.png',f0=5.,c=3.24,nclipframes=0):


        # fl = os.listdir(files)

        # I = GetImageList(files)

        if type(files) is list:

            I = [fl/norm(fl.ravel()) for fl in files]



        else:

                fl = os.listdir(files)


                I = [ ImgToArray(imread(files+f)) for f in fl if f.endswith(Ext)]


        # fcsv = [files+f for f in fl if f.endswith('.csv')]
        #
        # if len(fcsv)==1:
        #
        #     h = loadtxt(fcsv[0],delimiter=',')
        #
        #     r = int(round(len(h)/len(I)))
        #
        #     h = h[0::r]
        #
        #     l = min([len(h),len(I)])
        #
        #     h = h[0:l]
        #     I = I[0:l]
        #
        #     self.DisbondLengths = h

            # self.DisbondLengths = h[nclipframes+1:-nclipframes]
            # I = I[nclipframes+1:-nclipframes]


        self.RawImages = I.copy()
        self.Images = I.copy()

        # self.File = files.split('/')[-2]

        # self.FileNames = [f.strip(Ext) for f in fl if f.endswith(Ext)]

        Lxy = shape(I[0])

        self.WaveSpeed = c


        self.Period = ((Region[0][1]-Region[0][0])/Lxy[1],(Region[1][1]-Region[1][0])/Lxy[0],1.0)

        self.Region = vstack((array(Region),array([[0.,len(self.RawImages)*self.Period[2]]])))

        self.Pixels = (Lxy[0],Lxy[1])

        self.SubRegions = []

        # self.TrimSlices(5)

        # self.RegisterImages()

        self.AxialStack()

    # def GetRootPositions(self,rng=None):
    #
    #     from numpy import mean
    #
    #
    #     if rng is None:
    #
    #         imshow(self.AStack)
    #
    #         rng = ginput(n=0,timeout=0)
    #
    #         close()
    #
    #     ix0,iy0,ix1,iy1 = int(round(rng[0][0])),int(round(rng[0][1])),int(round(rng[1][0])),int(round(rng[1][1]))
    #
    #
    #     rpos = array([list(unravel_index(argmax(I[iy0:iy1+1,ix0:ix1+1]),I[iy0:iy1+1,ix0:ix1+1].shape)) for I in self.Images])
    #
    #     mu = mean(rpos,axis=0)
    #
    #     print(mu)
    #
    #     rpos = [(int(round(rpos[i,0]-mu[0])),int(round(rpos[i,1]-mu[1]))) for i in range(rpos.shape[0])]
    #
    #     self.RootPositions = rpos


    def PolyCropImages(self):

        I = []
        BI = []

        imshow(self.AStack)

        ind = ginput(n=0,timeout=0)

        close()

        for i in range(len(self.Images)):

            I.append(PolyCrop(self.Images[i],ind))

            try:

                BI.append(PolyCrop(self.BinaryImages[i],ind))

            except:

                pass

        self.Images = I

        try:

            self.BinaryImages=BI

        except:

            pass

        r = BoundingBox(ind)

        r[0,:] = r[0,:]*self.Period[0]+self.Region[0,0]
        r[1,:] = r[1,:]*self.Period[1]+self.Region[1,0]

        self.Region[0:2,0:2] = r

    def ResizeImages(self,factor):

        self.Images = [imresize(II,(int(II.shape[0]*factor),int(II.shape[1]*factor))) for II in self.Images]



    def RectangularCropImages(self):

        I = []

        imshow(self.AStack)

        ind = ginput(n=0,timeout=0)

        close()

        for II in self.Images:

            I.append(II[int(ind[0][1]):int(ind[1][1]),int(ind[0][0]):int(ind[1][0])])

        self.Images = I

        r = BoundingBox(ind)

        r[0,:] = r[0,:]*self.Period[0]+self.Region[0,0]
        r[1,:] = r[1,:]*self.Period[1]+self.Region[1,0]

        self.Region[0:2,0:2] = r

        self.AxialStack()


    def TrimSlices(self,n):

        self.Images = self.Images[n+1:-n]

    def ApplyGain(self,dB):

        from numpy import amax

        G = 10**(dB/20)

        Im = []

        for I in self.Images:

            A = amax(I)

            II = I*G

            II[II>A] = A

            Im.append(II)

        self.Images = Im

    def NormalizeSlices(self):

        from numpy import amax

        Im = [I/amax(I) for I in self.Images]

        self.Images = Im

    def RegisterImages(self,pts=None):

        from skimage.feature import register_translation


        from numpy import sum as asum
        #
        # Imavg = asum(dstack(self.Images),axis=2)/len(self.Images)

        if pts is None:

            imshow(self.AStack)

            ind = ginput(n=0,timeout=0)

            close()


        Isub = [ I[int(round(ind[0][1])):int(round(ind[1][1])),int(round(ind[0][0])):int(round(ind[1][0]))] for I in self.Images]

        Isubavg = asum(dstack(Isub),axis=2)/len(Isub)

        II = [ ShiftImage(self.Images[i],register_translation(Isubavg,Isub[i])[0]) for i in range(len(Isub))]

        self.Images = II







        # II = [ShiftImage(self.Images[i],pts[i]) for i in range(len(pts))]


        # for I in self.Images:
        #
        #     shift, error, diffphase = register_translation(Imavg, I)
        #
        #     II.append(real(ifftn(fourier_shift(fftn(I), shift))))


        self.Images = II

        self.AxialStack()

        self.HorizontalStack()

    def AxialStack(self,stcktype='max',stdthresh=0.):

        from numpy import amax,mean,std
        from numpy import sum as asum

        if stcktype=='max':

            self.AStack = amax(dstack(self.Images),axis=2)

        elif stcktype=='sumabovemean':

            A = moveaxis(dstack(self.Images),2,0)



            # self.AStack = self.AStack - mean(self.AStack,axis=2)

            # print(mean(self.AStack,axis=2).shape)

            A = asum(A*(A>mean(A,axis=0)+std(A,axis=0)*stdthresh),axis=0)
            self.AStack = A




    def GetBinaryImages(self):

        from skimage.filters import threshold_otsu
        # from misc import BimodalityCoefficient,AutoThreshold

        II = [(I>threshold_otsu(I)).astype(int) for I in self.Images]

        # BC = []
        #
        # SNR = []


        # for I in self.Images:
        #
        #     Im = I.copy()
            # Im = Im[Im>0]
            # thresh,snr = AutoThreshold(Im)
            # bc = BimodalityCoefficient(Im.ravel())
            #
            # if bc<5./9:
            #
            #     II.append(zeros(Im.shape,dtype=int))
            #
            # else:
            #
            #     thresh,snr = AutoThreshold(Im)
            #
            #
            #     II.append((Im>=thresh).astype(int))
            #
            #     BC.append(bc)
            # SNR.append(snr)


        # self.BimodalityCoefficient = array(BC)

        # snr1 = array([s[0] for s in SNR])
        # snr2 = array([s[1] for s in SNR])
        #
        # self.SNR = (snr1,snr2)


        self.BinaryImages = II



    def SmoothImages(self,FWHM):

        from skimage.filters import gaussian

        # sigma = sqrt(-8*log(0.5))/(self.WaveSpeed*bw)


        sigma = FWHM/2.35482

        sigma = sigma/self.Period[0]

        II = [ gaussian(I,sigma) for I in self.Images ]

        self.Images = II

    def DefineSubRegions(self,PreCrop=False):

        from skimage.measure import label,regionprops

        if PreCrop:

            self.RectangularCropImages()

        imshow(self.AStack)

        ind = ginput(n=0,timeout=0)

        close()

        self.SubRegions = []

        dx,dy,dz = self.Period

        for n in range(0,len(ind),2):

            xrng = (int(ind[n][0]),int(ind[n+1][0]))
            yrng = (int(ind[n][1]),int(ind[n+1][1]))

            Im = []
            thresh = []
            SNR = []
            props = []
            BI = []
            bc = []
            e = []
            std = []
            bsize = []

            for I in self.Images:

                II = I[yrng[0]:yrng[1],xrng[0]:xrng[1]]

                if any(II)>0.:

                    d = PrincipalImageMoments(II,dx,dy)

                    th,snr = AutoThreshold(II)

                    bi = II>th

                    rp = regionprops(label(bi),II)

                    # a = array([r.max_intensity for r in rp])

                    a = array([r.area for r in rp])


                    rp = rp[argmax(a)]

                    bcoeff = BimodalityCoefficient(II.ravel())

                    # rp = list(array(rp)[argsort(a)])



                    Im.append(II)
                    thresh.append(th)
                    SNR.append(snr)
                    props.append(rp)
                    BI.append(bi)
                    bc.append(bcoeff)

                    bsize.append((sqrt((dx*(rp.bbox[3]-rp.bbox[1]))**2+(dy*(rp.bbox[2]-rp.bbox[0]))**2) if bcoeff>0.6 else 0.))


                    e.append(d['Energy'])
                    std.append(sqrt(d['PrincipalValues']))

                else:

                    Im.append(II)
                    thresh.append(0.)
                    SNR.append([0.,0.])
                    props.append([])
                    BI.append(zeros(II.shape))
                    bc.append(0.)

                    bsize.append(0.)


                    e.append(0.)
                    std.append([0.,0.])



            SNR = array(SNR)
            std = array(std)

            self.SubRegions.append({'Images':Im, 'Thresholds':thresh, 'BinaryImages': BI, 'SNR1':list(SNR[:,0]), 'SNR2':list(SNR[:,1]), 'Properties':props, 'BimodalityCoefficient':list(bc), 'CornerCoords':(xrng[0],yrng[0]), 'Energy':e, 'MaxDeviation':list(std[:,1]), 'MinDeviation':list(std[:,0]), 'BinarySize':bsize})


    def SizeSubRegions(self):

        dx = self.Period[0]
        dy = self.Period[1]

        self.SubRegionSizes = [[ BlobSize(fr,dx,dy) for fr in sr ] for sr in self.SubRegions ]


    def RemoveSlices(self,Aspect=0.05):

        from numpy import floor,ceil,zeros

        imshow(self.HStack,aspect=Aspect)
        ind = ginput(n=0,timeout=0)

        close()

        z = zeros(self.Images[0].shape)

        for i in range(0,len(ind),2):

            i0 = int(floor(ind[i][0]))
            i1 = int(ceil(ind[i+1][0]))

            for j in range(i0,i1):

                self.Images[j] = z

        self.AxialStack()


    def SobelFilter(self):

        I = [sobel(I) for I in self.Images]

        self.Images = I


    def HorizontalStack(self,StackType='max'):

        from numpy import sum as asum,mean,amax
        # from skimage.filters import threshold_otsu

        self.HStack = zeros((shape(self.Images[0])[0],len(self.Images)))

        try:

            self.BinaryHStack = zeros((shape(self.BinaryImages[0])[0],len(self.Images)))

        except:

            pass

        if StackType is 'max':

            for i in range(len(self.Images)):

                self.HStack[:,i] = amax(self.Images[i],axis=1)

                try:

                	self.BinaryHStack[:,i] = amax(self.BinaryImages[i].astype('float'),axis=1)

                except:

                	pass

        elif StackType is 'average':

            for i in range(len(self.Images)):

                self.HStack[:,i] = mean(self.Images[i],axis=1)

                try:

                	self.BinaryHStack[:,i] = mean(self.BinaryImages[i].astype('float'),axis=1)

                except:

                	pass

    def VerticalStack(self,StackType='max'):

        from numpy import sum as asum,mean,amax
        # from skimage.filters import threshold_otsu

        self.VStack = zeros((shape(self.Images[0])[1],len(self.Images)))

        # self.BinaryHStack = zeros((shape(self.BinaryImages[0])[0],len(self.Images)))

        if StackType is 'max':

            for i in range(len(self.Images)):

                self.VStack[:,i] = amax(self.Images[i],axis=0)

        # self.BinaryHStack[:,i] = amax(self.BinaryImages[i].astype('float'),axis=1)

        elif StackType is 'average':

            for i in range(len(self.Images)):

                self.VStack[:,i] = mean(self.Images[i],axis=0)

        # self.BinaryHStack[:,i] = mean(self.BinaryImages[i].astype('float'),axis=1)

    def ResetImages(self):

        self.Images = self.RawImages.copy()

        self.AxialStack()
