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

parentdir = '/Users/jlesage/Dropbox/Eclipse/'

os.chdir(parentdir)


def ImgToArray(I):

    from numpy import sum as asum
    from numpy import amax,array

    bintodec = array([256**2,256,1])


    II = asum(bintodec*I[:,:,0:3],axis=-1)

    return II.astype(float)


def DisbondLinearClassifier(h,hh,size,N,niter,C_range,weights={1:5}):

	from sklearn import svm
	from sklearn import cross_validation
	from sklearn.grid_search import GridSearchCV
	from numpy import zeros,mean

	M = h.shape[0]

	n = floor(M/N).astype(int)

	yy = zeros((n,h.shape[1]))
	x = zeros((n,hh.shape[1]))

	for m in range(0,n):
		#
		# print(m*N)
		# print(m*N+N+1)

		yy[m,:] = mean(h[m*N:m*N+N+1,:],axis=0)

		x[m,:] = mean(hh[m*N:m*N+N+1,:],axis=0)


	# yy[n+1::,:] = mean(h[n*N+1::,:],axis=0)
	# x[n+1::,:] = mean(hh[n*N+1::,:],axis=0)
	Y = yy.copy()
	yy = yy.ravel()
	x = x.ravel()

	x = x.reshape((len(x),1))

	y = zeros((yy.shape)).astype(int)

	# y[(yy>=size[0])&(yy<=size[1])] = 1
	# y[yy>size[1]] = 2

	y[yy>size] = 1


	# skf = cross_validation.StratifiedKFold(y, n_folds=nfolds,shuffle=False, random_state=None)

	# param_grid = dict(C=C_range)
	# cv = cross_validation.StratifiedShuffleSplit(y, n_iter=niter, test_size=1/niter, random_state=42)
	# grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
	# grid.fit(x,y)
	#
	#
	# clf = svm.SVC(kernel='rbf',C=grid.best_params_['C'])
	# scores = grid.best_score_
	#


	param_grid = dict(C=C_range)
	cv = cross_validation.StratifiedShuffleSplit(y, n_iter=niter, test_size=1/niter, random_state=42)
	grid = GridSearchCV(svm.SVC(kernel='linear',class_weight=weights), param_grid=param_grid, cv=cv)
	grid.fit(x,y)


	clf = svm.SVC(kernel='linear',C=grid.best_params_['C'],class_weight=weights)
	scores = grid.best_score_

	return x,y,Y,clf,scores


def BlobSize(I,dx,dy,dBdrop=6):

    from numpy import linspace,cos,sin,amax,tan,mean,log10
    from skimage.measure import regionprops,profile_line,label
    from skimage.filters import threshold_otsu
    from matplotlib.pylab import plot,show
    from numpy import meshgrid,linspace,trapz,array,log,min
    from numpy.linalg import eig
    from matplotlib.pylab import imshow,show
    from misc import RMS

    # II = I.transpose()

    # Nx = I.shape[1]
    # Ny = I.shape[0]

    # y,x = meshgrid(linspace(0,Ny-1,Ny),linspace(0,Nx-1,Nx))

    # x = dx*linspace(0,Nx-1,Nx)
    # y = dy*linspace(0,Ny-1,Ny)

    th = threshold_otsu(I)

    n = I[I<th].ravel()

    r = regionprops(label(I>th),I)

    a = [r[i].max_intensity for i in range(0,len(r))]

    r = r[argmax(a)]

    r = regionprops(label(I>r.max_intensity*10**(-dBdrop/20)),I)

    bb = r[0].bbox

    psnr = 20*log10(r[0].max_intensity/RMS(n))

    print(psnr)

    if psnr<dBdrop:

        d = 0.0

    else:

        d = sqrt((dx*(bb[3]-bb[1]))**2 + (dy*(bb[2]-bb[0]))**2)





    # bb = r.bbox

    # th = r.orientation
    #
    # l = r.major_axis_length
    #
    # II = r.intensity_image
    #
    # II = II-min(abs(II))
    #
    #
    # Ny,Nx=II.shape
    #
    # x = dx*linspace(0,Nx-1,Nx)
    # y = dy*linspace(0,Ny-1,Ny)
    #
    # # print(Ny)
    # # print(Nx)
    #
    # # yy,xx = meshgrid(dy*linspace(0,Ny-1,Ny),dx*linspace(0,Nx-1,Nx))
    #
    # xx,yy = meshgrid(x,y)
    #
    #
    #
    # A = trapz(trapz(II,dx=dy),dx=dx)
    #
    # II = II/A
    #
    # mux = trapz(trapz(xx*II,dx=dx,axis=1),dx=dy)
    #
    #
    # muy = trapz(trapz(yy*II,dx=dy),dx=dx)
    #
    #
    # sigmax = trapz(trapz(II*(xx-mux)*(xx-mux),dx=dx,axis=1),dx=dy)
    #
    # sigmaxy = trapz(trapz(II*(xx-mux)*(yy-muy),dx=dx,axis=1),dx=dy)
    #
    # sigmay = trapz(trapz(II*(yy-muy)**2,dx=dy,axis=0),dx=dx)
    #
    # S = array([[sigmax,sigmaxy],[sigmaxy,sigmay]])
    #
    # w,v = eig(S)
    #
    # ie = abs(w).argsort()
    #
    # w = w[ie]
    # v = v[:,ie]
    #
    # lfwhm = sqrt(w[1])*2*sqrt(2*log(2))
    #
    #
    # d = {'MajorAxis':sqrt((l*cos(th)*dx)**2 + (l*sin(th)*dy)**2), 'BoundingBox': sqrt((dx*(bb[3]-bb[1]))**2 + (dy*(bb[2]-bb[0]))**2), 'FWHM':lfwhm}


    return d



def HoleSpectrum(a,xrng,yrng,res):

	from scipy.special import jn



	N = round((xrng[1]-xrng[0])/res)
	M = round((yrng[1]-yrng[0])/res)


	u = linspace(0,1/res,N)
	v = linspace(0,1/res,M)

	u,v = meshgrid(u,v)

	F = zeros((N,M),dtype='complex')

	for aa in a:

		F += (2*jn(1,aa[0]*2*pi*sqrt(u**2+v**2))/(2*pi*aa[0]*sqrt(u**2+v**2)))*exp(-2j*pi*aa[1]*u)*exp(-2j*pi*aa[2]*v)


	F[0,0] = len(a)*(1.+0j)
	f = ifft2(F)

	return f,F


def WedgeDiffractedField(f,l,L,h=5.0,phi=31.0,c=[2.3,(3.2,5.9)]):



	lmbd = c/f

	dy = lmbd/4

	N = FFTLengthPower2(round(L/dy))

	x = linspace(0,L,N)

	ky = 2*pi*linspace(-1/(2*dy),1/(2*dy),N)


	kx = sqrt((2*pi*f/c)**2-ky**2+0j)

	# kx[abs(imag(kx))>0]=0+0j

	kx,x = meshgrid(kx,x)

	P = sinc(l*ky)*exp(1j*kx*x)/(1j*l*kx)

	# P[]

	p = ifft(P)

	y = linspace(-L/2,L/2,N)

	return x,y,p

# def RectangularTransducerField(f,l,n,p,L,c=5.9):

def RectangularTransducerField(f,l,L,c=5.9):


	lmbd = c/f

	dy = lmbd/10

	N = FFTLengthPower2(round(L/dy))

	x = linspace(0,L,N)

	ky = 2*pi*linspace(-1/(2*dy),1/(2*dy),N)

	ky,x = meshgrid(ky,x)

	kx = sqrt((2*pi*f/c)**2-ky**2+0j)

	# P = sinc(l*ky)*exp(1j*kx*x)*exp(-1j*0*p*ky)/(1j*l*kx) + exp(-2j*f*10)*sinc(l*ky)*exp(1j*kx*x)*exp(1j*n*p*ky)/(1j*l*kx) + exp(-4j*f*20)*sinc(l*ky)*exp(1j*kx*x)*exp(1j*2*n*p*ky)/(1j*l*kx)

	P = sinc(l*ky/pi)*exp(1j*kx*x)/(1j*l*kx)


	p = ifftshift(ifft(ifftshift(P),axis=1))

	y = linspace(-L/2,L/2,N)

	return p,P


def FrequencyTFM(s,r,dt,a=100,b=100,l=0.6,c=5.9,alpha=0.05):

	lmbd = c/f

	N = FFTLengthPower2(len(s))

	S = rfft(s,N)

	R = rfft(r,N,axis=2)

	# S = S.reshape((1,1,N)))

	H = R/S

	f = linspace(0,1/(2*dt),floor(N/2)+1)

	y = linspace(-b/2,b/2,round(b/l))

	x = linspace(0,a,round(a/l))

	return x,y,H


def ThinPlateTOFTSimulate(l,d,A=0.1,T=5.,T0=2.,f0=5.,BW=2.,c=3.2):

	fmax = f0+10*BW/2

	dt = 1/(2*fmax)

	N = FFTLengthPower2(round(T/dt))

	f = linspace(0,1/(2*dt),N/2+1)

	a = -4*log(0.5)/(BW**2)

	X = exp(-a*(f-f0)**2)*exp(-2j*pi*T0*f)
	X = X/abs(max(X))

	T1 = (sqrt(l[0]**2+d[0]**2) + sqrt((l[1]-l[0])**2 + d[0]**2))/c

	T2 = 2*sqrt((l[1]/2)**2+d[1]**2)/c


	H = A*exp(-2j*pi*T1*f) + exp(-2j*pi*T2*f)

	Y = X*H

	x = ifft(2*X,N)
	y = ifft(2*Y,N)

	C = irfft(log(abs(H)))

	Cy = irfft(log(abs(Y)))

	df = f[1]-f[0]



	T = linspace(0.,1/(2*df),len(C))

	dT = T2-T1

	t = linspace(0.,len(y)*dt,len(y))


	return t,x,y,dT,T,Cy,C

class FMC:

	def __init__(self,f0=10.,dt=1/50.,BW=1.,c=5.9,atten=0.06,pitch=0.3,elsize=(0.6,10.),nel=32):

		self.SamplingPeriod = dt
		self.CentreFrequency = f0
		self.WaveSpeed ={'Piece': (3.24,5.9), 'Wedge': 2.3}
		self.Bandwidth = f0*BW
		self.Pitch = pitch
		self.ElementSize = elsize
		self.NumberOfElements = nel
		# self.Resolution = c/(2*(f0+0.5*self.Bandwidth))
		fmax = f0+0.5*BW
		# self.Resolution = (2*pitch,2*sqrt((fmax/c)**2-(1/pitch)**2))

		self.Attenuation = atten

	def LoadAScansFromText(self,flname,t0=0.0):

		# a = genfromtxt(flname,delimiter=',')

		a = loadtxt(flname,delimiter=',')

		# a = a[:,0:-2]

		a = a.reshape((self.NumberOfElements,self.NumberOfElements,shape(a)[1]))

		# a = a.reshape((self.NumberOfElements,self.NumberOfElements,int(round(shape(a)[1]/self.NumberOfElements))),order='F')


		self.Time = linspace(0.,self.SamplingPeriod*shape(a)[2],shape(a)[2])

		a[:,:,self.Time<=t0] = 0.0



		self.RawAScans = [a]
		self.AScans = [a.copy()]


	def LoadAScansFromPickle(self,flname):

		d = pickle.load(open(flname,'rb'))

		self.RawAScans = d['AScans']
		self.SamplingPeriod = 1/d['SamplingFrequency']

		self.AScans = d['AScans'].copy()


	def ApplyWindow(self,Gate,alpha):

		from scipy.signal import tukey, detrend

		Gate = array(Gate)/self.SamplingPeriod
		Gate = Gate.astype(int)

		w = tukey(Gate[1]-Gate[0],alpha).reshape((1,Gate[1]-Gate[0]))

		A = []


		for aa in self.AScans:

			a = detrend(aa[:,:,Gate[0]:Gate[1]],axis=-1,bp=((0,100)))

			z = zeros(aa.shape)

			z[:,:,Gate[0]:Gate[1]] = w*a

			A.append(z)

		self.AScans = A


	def ProcessAScans(self,func):

		A = []

		for a in self.AScans:

			A.append(func(a))

		self.AScans = A


	# def WavenumberImage(self,Ly,pixels,N=32):
	#
	# 	from numpy import sum as asum
	# 	from numpy import tile
	# 	from numpy import meshgrid,hstack,delete,isnan,vstack
	# 	from scipy.interpolate import interp1d, griddata
	# 	from misc import FFTLengthPower2
	#
	# 	dy = ((1/(self.WaveSpeed*self.SamplingPeriod))**2-(1/self.Pitch)**2)**(-1/2)
	#
	# 	print(dy)
	# 	dy = self.Pitch
	# 	# nx = int(round(Lx/self.Pitch))
	# 	ny = FFTLengthPower2(int(round(Ly/dy)))
	# 	nx = ny
	# 	# nx0 = int(round(nx/2))-self.NumberOfElements
	# 	nx0 = int(round((nx-self.NumberOfElements)/2))
	# 	nt = self.RawAScans.shape[2]
	#
	# 	# U = zeros((nx,nx,nt/2+1),dtype=complex)
	# 	U = zeros((nx,nx,int(floor(nt/2)+1)),dtype=complex)
	#
	# 	# U = zeros((nx,nx,nt),dtype=complex)
	#
	#
	#
	# 	# U[nx0:nx0+self.NumberOfElements,nx0:nx0+self.NumberOfElements,:] = conj(rfft(self.RawAScans,axis=2))
	#
	# 	U[nx0:nx0+self.NumberOfElements,nx0:nx0+self.NumberOfElements,:] = rfft(self.RawAScans,axis=2)
	#
	# 	# U[nx0:nx0+self.NumberOfElements,nx0:nx0+self.NumberOfElements,:] = fft(self.RawAScans,axis=2)
	#
	#
	#
	# 	# U[nx0:nx0+self.NumberOfElements,nx0:nx0+self.NumberOfElements,:] = conj(rfft(self.RawAScans,axis=2))
	#
	# 	U[:,:,0] = 0+0j
	#
	#
	# 	U = fft2(U,axes=(0,1))
	# 	# U[:,:,0] = 0+0j
	# 	# U = fftshift(U,axes=(0,1))
	#
	# 	ku = 2*pi*linspace(0,1/self.Pitch,nx)
	# 	# kx = 2*pi*linspace(-1/(2*self.Pitch),1/(2*self.Pitch),nx)
	# 	# ky = 2*pi*linspace(0.0,1/(2*dy),ny/2)
	#
	# 	# ky = 2*pi*linspace(0,1/(2*dy),ny/2+1)
	#
	# 	# ky = 2*pi*linspace(-1/(2*dy),1/(2*dy),ny)
	#
	# 	ky = 2*pi*linspace(0,1/(2*dy),ny)
	#
	# 	# ky = 2*pi*linspace(0,1/(2*dy),floor(ny/2)+1)
	#
	#
	#
	# 	# k = 2*pi*linspace(0,1/self.SamplingPeriod,U.shape[2])/self.WaveSpeed
	#
	# 	k = 2*pi*linspace(0,1/(2*self.SamplingPeriod),U.shape[2])/self.WaveSpeed
	#
	# 	# k = 2*pi*linspace(1/(2*self.SamplingPeriod),1/self.SamplingPeriod,U.shape[2])/self.WaveSpeed
	#
	#
	# 	# Kv,K = meshgrid(ku,k)
	#
	# 	K,Kv = meshgrid(k,ku)
	#
	#
	# 	# K = tile(k.reshape((1,1,len(k))),(shape(Kv)[0],shape(Kv)[1],1))
	#
	# 	# KKu = Ku.flatten()
	# 	# # Kxv = Kxv.reshape((len(Kxv),1))
	# 	#
	# 	# KK = K.flatten()
	# 	# Kv = Kv.reshape((len(Kv),1))
	#
	# 	Ky,Kx = meshgrid(ky,ku)
	#
	# 	F = zeros((len(ku),len(ky),len(ku)),dtype=complex)
	#
	# 	for m in range(N):
	#
	# 		FF = -(4*pi)**2*sqrt(K**2-Kv**2)*sqrt(K**2-(ku[m])**2)*U[m,:,:]
	#
	# 		# FF = -U[m,:,:]
	#
	# 		FF = FF[~isnan(FF)].flatten()
	#
	# 		kx = ku[m] + Kv
	#
	# 		ky = sqrt(K**2-ku[m]**2) - sqrt(K**2 - Kv**2)
	#
	# 		kx = kx.flatten()
	#
	# 		kx = kx.reshape((len(kx),1))
	#
	# 		ky = ky.flatten()
	#
	# 		ky = ky.reshape((len(ky),1))
	#
	# 		# kky = sqrt(Kv**2-kx[m]**2)+sqrt(Kv**2-Kxv**2)
	#
	# 		# kkynan = isnan(kky)
	# 		#
	# 		# kky = kky[~kkynan]
	# 		# kky = kky.reshape((len(kky),1))
	# 		#
	# 		# kkx = kkx[~kkynan]
	# 		# kkx = kkx.reshape((len(kkx),1))
	#
	#
	# 		pts = hstack((kx,ky))
	#
	# 		pts = array([pts[i,:] for i in range(shape(pts)[0]) if all(~isnan(pts[i,:]))])
	#
	#
	# 		# F[:,:,m]=griddata(pts,FF,(kxg,kyg),method='nearest')
	# 		# F[:,:,m]=griddata(pts,FF,(kxg,kyg),fill_value=0+0j,method='cubic')
	#
	# 		F[:,:,m]=griddata(pts,FF,(Kx,Ky),fill_value=0+0j,method='linear')
	#
	#
	#
	#
	#
	#
	# 	f = asum(F,axis=2).transpose()
	#
	#
	# 	# npadx = pixels[0]-f.shape[1]
	# 	# nxhalf = int(floor(f.shape[1]/2))
	#
	# 	# npady = pixels[1]-f.shape[0]
	# 	# nyhalf = int(floor(f.shape[0]/2))
	#
	# 	# f = hstack((f[:,0:nxhalf],zeros((f.shape[0],npadx),dtype=complex),f[:,nxhalf::]))
	#
	# 	# f = vstack((f[0:nyhalf,:],zeros((npady,f.shape[1]),dtype=complex),f[nyhalf::,:]))
	# 	#
	#
	#
	#
	# 	self.ImageStack = ifft2(f,(f.shape[1],max(array([pixels[1],f.shape[0]]))),axes=(1,0))

		# self.ImageStack = ifft2(f)

		# self.ImageStack = irfft(ifft(f,axis=1),pixels[1],axis=0)


	# def TimeReversalImage(self):
	#
	#

	# def WavenumberImage(self,N):
	#
	#
	# 	foreach a in self.AScans:
	#
	# 		A = fftn(a,axis=(0,1))
	# 		A = rfft(A,axis=-1)
	#
	# 		A = fftshift(A,axis=0)
	#
	# 		trind = int(round(A.shape[0]/2))-N:int(round(A.shape[0]/2))+N)
	#
	# 		kxt = 2*pi*linspace(-1/(self.Pitch),1/(self.Pitch),A.shape[0])
	# 		kxt = kxt[trind]
	#
	# 		A = A[trind,:,:]
	#
	# 		kxr = 2*pi*fftshift(linspace(-1/(self.Pitch),1/(self.Pitch),A.shape[1]))
	#
	# 		w = 2*pi*linspace(0,1/(2*self.SamplingPeriod),A.shape[2])
	#
	# 		kzr = sqrt((w/c)**2-kxr**2)
	#
	# 		kzt = sqrt((w/c)**2-kxt**2)
	#
	# 		F = []
	#
	#

	def AngularSweep(self,angles,ind,WedgeAngle=0):

		from numpy import sum as asum,arcsin,ceil,dstack,zeros

		WedgeAngle = WedgeAngle*(pi/180)

		# A = rfft(self.RawAScans[ind],n=self.RawAScans.shape[2]*2,axis=2)

		# x = self.Pitch*linspace(self.NumberOfElements-1,0,self.NumberOfElements)

		x = self.Pitch*linspace(-self.NumberOfElements/2,self.NumberOfElements/2,self.NumberOfElements)

		# f = linspace(0,1/(2*self.SamplingPeriod),A.shape[2])
		#
		# f,x = meshgrid(f,x)

		if WedgeAngle>0:

			T = lambda th: sin(arcsin(sin(pi*th/180)*self.WaveSpeed['Wedge']/self.WaveSpeed['Piece'][0]) - WedgeAngle)/self.WaveSpeed['Wedge']


		else:

			T = lambda th: sin(th*pi/180)/self.WaveSpeed['Piece'][1]

		# Tmax = x[-1]*max([abs(T(an)) for an in angles])

		# print(Tmax)


		# a = []

		# N = round(ceil(Tmax/self.SamplingPeriod))
		#
		# print(N)


		A = rfft(self.AScans[ind],axis=2)

		# L = A.shape[2]

		# A = hstack((zeros((N,self.NumberOfElements,L)),A,zeros((N,self.NumberOfElements,L))))
		# A = vstack((zeros((self.NumberOfElements,N,L)),A,zeros((self.NumberOfElements,N,L))))

		# z = zeros((self.NumberOfElements,self.NumberOfElements,N),dtype=complex)
		#
		# A = dstack((z,A,z))

		f = linspace(0,1/(2*self.SamplingPeriod),A.shape[2])

		f,x = meshgrid(f,x)

		b = 2*pi*1j*f*x

		# M = self.RawAScans[ind].shape[2]

		M = 2*A.shape[2]-2

		a = zeros((len(angles),len(angles),M),dtype=complex)


		for ta in range(len(angles)):
			for ra in range(len(angles)):

				# aa = ifft(asum(asum(A*exp(T(angles[ta])*b),axis=1)*exp(-b*T(angles[ra])),axis=0),M)

				aa = asum(A*exp(-b*T(angles[ra])),axis=1)
				aa = asum(aa*exp(b*T(angles[ta])),axis=0)

				aa = ifft(aa,M)


				a[ta,ra,:] = aa[0:M]



			# a.append(ifft(asum(asum(A*exp(T(an)*b),axis=1)*exp(-b*T(an)),axis=0),2*A.shape[2]-2))


		return a


	def TFMImage(self):

		A = self.AScanSpectra



		I = zeros((Ny,Nx,Nf),dtype='complex')


		self.Image = I


# class FMCImage:
#
# def __init__(self,files,Region=[[0.,15.],[0.,25.]],filerange=None,RGBInd=0,Ext='.png',f0=5.,c=3.24):
#
#
# I = []
#
#
# if os.path.isdir(files):
#
# os.chdir(files)
# fl=[f for f in os.listdir(os.curdir) if f.endswith(Ext)]
#
# try:
# hfl=[f for f in os.listdir(os.curdir) if f.endswith('.csv')]
# h = loadtxt(hfl[0],delimiter=',')
#
# # print(len(h))
#
# except:
# pass
#
# I = [ImgToArray(imread(files+f)) for f in fl]
#
#
# else:
#
# I = imread(files)
#
#
# if filerange!=None:
#
# I = [I[fr] for fr in filerange]
#
#
#
# self.RawImages = I.copy()
# self.Images = I.copy()
#
# try:
#
# ratio = len(h)/len(I)
#
# self.DisbondLengths = h[0::int(round(ratio))]
#
# if len(self.DisbondLengths)>len(I):
#
# self.DisbondLengths = self.DisbondLengths[0:len(I)]
#
# else:
#
# self.RawImages = self.RawImages[0:len(self.DisbondLengths)]
# self.Images = self.Images[0:len(self.DisbondLengths)]
#
#
#
# except:
#
# pass
#
# self.File = files.split('/')[-2]
#
# Lxy = shape(I[0])
#
# self.WaveSpeed = c
#
# self.CentreFrequency = f0
#
# self.Period = ((Region[0][1]-Region[0][0])/Lxy[1],(Region[1][1]-Region[1][0])/Lxy[0],1.0)
#
# self.Region = vstack((array(Region),array([[0.,len(self.RawImages)*self.Period[2]]])))
#
# self.Pixels = (Lxy[0],Lxy[1])
#
# self.Resolution = c/(2*f0)
#
# self.SubRegions = []
#
#
#
# os.chdir(parentdir)
#
# def ZeroOffset(self):
#
# self.Images = []
#
# for I in self.RawImages:
#
# II = fft2(I)
# II[0,0] = 0+0j
#
# self.Images.append(real(ifft2(II)))
#
#
# def CropImages(self,XRange,YRange):
#
# I = []
#
# for II in self.Images:
#
# I.append(II[YRange[0]:YRange[1],XRange[0]:XRange[1]])
#
# self.Images = I
#
# def PolyCropImages(self):
#
# I = []
# BI = []
#
# imshow(self.AStack)
#
# ind = ginput(n=0,timeout=0)
#
# close()
#
# for i in range(len(self.Images)):
#
# I.append(PolyCrop(self.Images[i],ind))
#
# try:
#
# BI.append(PolyCrop(self.BinaryImages[i],ind))
#
# except:
#
# pass
#
# self.Images = I
#
# try:
#
# self.BinaryImages=BI
#
# except:
#
# pass
#
# r = BoundingBox(ind)
#
# r[0,:] = r[0,:]*self.Period[0]+self.Region[0,0]
# r[1,:] = r[1,:]*self.Period[1]+self.Region[1,0]
#
# self.Region[0:2,0:2] = r
#
#
#
# def RectangularCropImages(self):
#
# I = []
#
# imshow(self.AStack)
#
# ind = ginput(n=0,timeout=0)
#
# close()
#
# for II in self.Images:
#
# I.append(II[int(ind[0][1]):int(ind[1][1]),int(ind[0][0]):int(ind[1][0])])
#
# self.Images = I
#
# r = BoundingBox(ind)
#
# r[0,:] = r[0,:]*self.Period[0]+self.Region[0,0]
# r[1,:] = r[1,:]*self.Period[1]+self.Region[1,0]
#
# self.Region[0:2,0:2] = r
#
#
#
#
# def RegisterImages(self):
#
# from numpy import sum as asum
#
# Imavg = asum(dstack(self.Images),axis=2)/len(self.Images)
#
# II = []
#
# for I in self.Images:
#
# shift, error, diffphase = register_translation(Imavg, I)
#
# II.append(real(ifftn(fourier_shift(fftn(I), shift))))
#
#
# self.Images = II
#
#
# def AxialStack(self):
#
# from numpy import amax
#
# self.AStack = amax(dstack(self.Images),axis=2)
#
#
# def Threshold(self,nstd):
#
# from skimage.filters import threshold_otsu
#
# II = []
#
# for I in self.Images:
#
# thresh = threshold_otsu(I)
#
# Imean = mean(I[I<thresh].ravel())
# Istd = std(I[I<thresh].ravel())
#
# # I[I<Imean+nstd*Istd]=0.0
#
# III = I.copy()
#
# III[I<Imean+nstd*Istd] = 0.0
#
#
# II.append(III)
#
#
# self.Images = II
#
#
# def GetBinaryImages(self):
#
# # from skimage.filters import threshold_otsu
# from misc import BimodalityCoefficient,AutoThreshold
#
# II = []
#
# BC = []
#
# SNR = []
#
#
#
#
#
# for I in self.Images:
#
# Im = I.copy()
# # Im = Im[Im>0]
# # thresh,snr = AutoThreshold(Im)
# bc = BimodalityCoefficient(Im.ravel())
#
# if bc<5./9:
#
# II.append(zeros(Im.shape,dtype=int))
#
# else:
#
# thresh,snr = AutoThreshold(Im)
#
#
# II.append((Im>=thresh).astype(int))
#
# BC.append(bc)
# # SNR.append(snr)
#
#
# self.BimodalityCoefficient = array(BC)
#
# # snr1 = array([s[0] for s in SNR])
# # snr2 = array([s[1] for s in SNR])
# #
# # self.SNR = (snr1,snr2)
#
#
# self.BinaryImages = II
#
#
#
# def SmoothImages(self,FWHM):
#
# from skimage.filters import gaussian
#
# # sigma = sqrt(-8*log(0.5))/(self.WaveSpeed*bw)
#
# sigma = FWHM/2.35482
#
# II = [ gaussian(I,sigma) for I in self.Images ]
#
# self.Images = II
#
#
# def DefineSubRegions(self):
#
# imshow(self.AStack)
#
# ind = ginput(n=0,timeout=0)
#
# close()
#
# self.SubRegions = [[ I[int(ind[n][1]):int(ind[n+1][1]),int(ind[n][0]):int(ind[n+1][0])] for I in self.Images] for n in range(0,len(ind),2)]
#
#
# def SizeSubRegions(self):
#
# dx = self.Period[0]
# dy = self.Period[1]
#
# s = [[ BlobSize(fr,dx,dy) for fr in sr ] for sr in self.SubRegions ]
#
# self.SubRegionSizes = s
#
#
# def RemoveSlices(self,Aspect=0.05):
#
# from numpy import floor,ceil,zeros
#
# imshow(self.HStack,aspect=Aspect)
# ind = ginput(n=0,timeout=0)
#
# close()
#
# z = zeros(self.Images[0].shape)
#
# for i in range(0,len(ind),2):
#
# i0 = int(floor(ind[i][0]))
# i1 = int(ceil(ind[i+1][0]))
#
# for j in range(i0,i1):
#
# self.Images[j] = z
#
#
# def SobelFilter(self):
#
#
# II = []
#
# for I in self.Images:
#
# II.append(sobel(I))
#
# self.Images = II
#
#
#
# def SVDDecompose(self,Truncation,ZeroMinValue=True):
#
# self.SVD = []
#
# I = []
#
# for i in self.Images:
#
# S = svd(i,full_matrices=False)
#
# sv = zeros(len(S[1]))
#
# if type(Truncation) is range:
#
# sv[Truncation] = S[1][Truncation]
#
# elif type(Truncation) is float:
#
# SSqrsTot = dot(S[1].reshape((1,len(S[1]))),S[1].reshape((len(S[1]),1)))
#
# s = 1
#
# SSqrsRatio = 0.0
#
# while SSqrsRatio<=Truncation:
#
# 	SSqrsRatio = dot(S[1][0:s].reshape((1,len(S[1][0:s]))),S[1][0:s].reshape((len(S[1][0:s]),1)))/SSqrsTot
#
# 	s+=1
#
# sv[0:s] = S[1][0:s]
#
# II = dot(S[0],dot(diag(sv),S[2]))
#
# if ZeroMinValue:
#
# I.append(II-amin(II))
#
# else:
#
# I.append(II)
#
# self.SVD.append(S)
#
# self.Images = I
#
# def HorizontalStack(self,StackType='max'):
#
# from numpy import sum as asum,mean,amax
# # from skimage.filters import threshold_otsu
#
# self.HStack = zeros((shape(self.Images[0])[0],len(self.Images)))
#
# try:
#
# self.BinaryHStack = zeros((shape(self.BinaryImages[0])[0],len(self.Images)))
#
# except:
#
# pass
#
# if StackType is 'max':
#
# for i in range(len(self.Images)):
#
# self.HStack[:,i] = amax(self.Images[i],axis=1)
#
# try:
#
# 	self.BinaryHStack[:,i] = amax(self.BinaryImages[i].astype('float'),axis=1)
#
# except:
#
# 	pass
#
# elif StackType is 'average':
#
# for i in range(len(self.Images)):
#
# self.HStack[:,i] = mean(self.Images[i],axis=1)
#
# try:
#
# 	self.BinaryHStack[:,i] = mean(self.BinaryImages[i].astype('float'),axis=1)
#
# except:
#
# 	pass
#
# def VerticalStack(self,StackType='max'):
#
# from numpy import sum as asum,mean,amax
# # from skimage.filters import threshold_otsu
#
# self.VStack = zeros((shape(self.Images[0])[1],len(self.Images)))
#
# # self.BinaryHStack = zeros((shape(self.BinaryImages[0])[0],len(self.Images)))
#
# if StackType is 'max':
#
# for i in range(len(self.Images)):
#
# self.VStack[:,i] = amax(self.Images[i],axis=0)
#
# # self.BinaryHStack[:,i] = amax(self.BinaryImages[i].astype('float'),axis=1)
#
# elif StackType is 'average':
#
# for i in range(len(self.Images)):
#
# self.VStack[:,i] = mean(self.Images[i],axis=0)
#
# # self.BinaryHStack[:,i] = mean(self.BinaryImages[i].astype('float'),axis=1)
#
#
#
# def ResetImages(self,ZeroDC=False):
#
# self.Images = self.RawImages.copy()
#
# if ZeroDC:
# self.ZeroOffset()
