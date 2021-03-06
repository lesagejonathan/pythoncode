def iseven(n):
    return n%2==0

def pkfind(x,y,n):
	from matplotlib.pyplot import ginput,plot,close,xlabel,ylabel,grid
	from numpy import zeros, array
	close('all')
	plot(x,y)
	grid(True)
	rng=ginput(2*n)
	close()
	xmax=zeros(n)
	ymax=zeros(n)
	for i in range(n):
		x1=rng[2*i][0]
		x2=rng[2*i+1][0]
		if x1<x2:
			xrng=(x>=x1)&(x<=x2)
		elif x2<x1:
			xrng=(x>=x2)&(x<=x1)
		xmax[i]=x[xrng][y[xrng].argmax()]
		ymax[i]=y[xrng].max()
	indmax=(xmax-x[0])/abs(abs(x[1])-abs(x[0]))
	return xmax,ymax

def cutwvfrm(t,x,zerooffset=False,window=('tukeywin',0.5)):
	import numpy as np
	from matplotlib.pyplot import ginput, plot, close
	plot(t,x)
	tx=ginput(2)
	close()
	tx1=tx[0]
	tx2=tx[1]
	if (tx1[0]<tx2[0]):
		t1=tx1[0]
		t2=tx2[0]
	elif (tx1[0]>tx2[0]):
		t1=tx2[0]
		t2=tx1[0]
	ind=(t>=t1)&(t<=t2)
	X=x[ind]
	if window[0]=='tukeywin':
		X=X*tukeywin(X.size,window[1])
	elif window[0]=='gausswin':
		X=X*gausswin(X.size,window[1])
	elif window[0]=='expwin':
		X=X*expwin(X.size,window[1])
	if zerooffset==False:
		T=t[ind]
	elif zerooffset==True:
		dt=abs(abs(t[1])-abs(t[0]))
		T=np.linspace(0.0,dt*X.size,X.size)
	return T,X,ind

def freqs(N,dt):
	from numpy import linspace
	f=linspace(0.,1/(2*dt),N/2+1)
	return f

def loaddata(pth):
	import numpy as np
	import cPickle as cp
	fltype=pth[-3::]
	if (fltype=='csv')|(fltype=='isf'):
		x=np.loadtxt(pth,delimiter=',',unpack=True)
	elif fltype=='dat':
		x=cp.load(pth,'rb')
	elif fltype=='npy':
		x=np.load(pth)
	else:
		x=np.loadtxt(pth,unpack=True)
	return x

def savedata(pth,x):
	import numpy as np
	import os
	import scipy.io as sio
	if type(pth)==tuple:
		fltype=pth[-1][-3::]
	else:
		fltype=pth[-3::]
	if (fltype=='csv')|(fltype=='isf'):
		np.savetxt(pth,x,delimiter=',')
	elif (fltype=='mat'):
		# for this kind of save, pth must be a tuple (pthtofile,filname) and x must be a dict {'varname':var}
		os.chdir(pth[0])
		sio.savemat(pth[1],x)
	elif (fltype=='npy'):
		np.save(pth,x)
	else:
		np.savetxt(pth,x)

def asignal(x):
	import numpy.fft as nf
	import numpy as np
	X=2*nf.fft(x)
	X[np.ceil(X.size/2)::]=0
	xa=nf.ifft(X)
	return(xa)


def tukeywin(N,alpha):
	import numpy as np
	n=np.linspace(0,alpha*N/2,alpha*N/2)
	w1=np.sin(np.pi/(alpha*N)*n)
	w2=np.ones(N-2*n.size)
	w3=np.cos(np.pi/(alpha*N)*n)
	w=np.hstack((w1,w2,w3))
	return w

def gausswin(N,stddev):
	import numpy as np
	n=np.linspace(0,N,N)
	w=np.exp(-((n-np.floor(N/2))**2)/(2*stddev**2))
	return w

def expwin(N,decay):
	import numpy as np
	n=np.linspace(0,N,N)
	w=np.exp(-decay*n)
	return w

def moving_average(a, n=3):
    import numpy as np
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def window_frequency_domain(x, dt, fc, bw, debug=False):
    from numpy import hanning, zeros, array
    from numpy.linalg import norm
    from numpy.fft import rfft, irfft
    from matplotlib.pyplot import plot, figure
    N = len(x)
    X = rfft(x)
    fc = fc*dt*N
    bw = int(bw*dt*N)
    window = zeros(len(X))
    window[fc-bw:fc+bw] = hanning(2*bw)
    if debug:
        figure()
        plot(window*0.3)
        plot(X/max(X)) # normalize to 1
    X = window*X
    return irfft(X)

def corr(x1,x2,dt,off=0.):
	import numpy as np
	Noff=int(off/dt)
	x2=np.hstack((np.zeros(Noff),x2))
	x1=np.hstack((x1,np.zeros(x2.size-x1.size)))
	x12=np.correlate(x1,x2,'full')
	t=np.linspace(-x12.size*dt/2,x12.size*dt/2,x12.size)
	return t,x12


def beamfreqs(E,h,rho,l,bc):
	import numpy as np
	if bc=='cf':
		B=np.array([1.875104,4.694091,7.854757,10.995541])
	elif bc=='ff':
		B=np.array([4.730041,7.853205,10.995608,14.137165])
	s=(B**2/(4*np.pi))*h*np.sqrt(E/(3*rho*l**4))
	return s

def barfreqs(n,E,rho,l):
	import numpy as np
	s=(n*np.sqrt(E/rho))/(2*l)
	return s

def ftrans(x,dt,frange=None):
	from numpy.fft import rfft
	f=freqs(x.size,dt)
	X=rfft(x)
	if type(frange)==tuple:
		X=X[(f>=frange[0])&(f<=frange[1])]
		f=f[(f>=frange[0])&(f<=frange[1])]
	return f,X

def pltft(x,dt,pltype='mag',frange=None):
	import numpy as np
	import matplotlib.pyplot as plt
	f,X=ftrans(x,dt)
	if pltype=='mag':
		plt.plot(f,np.abs(X/len(X)))
		plt.xlabel(r'$f$')
		plt.ylabel(r'$|X (\omega)|$')
		if type(frange)==tuple:
			plt.xlim(frange)
		plt.show()
	elif pltype=='phase':
		plt.plot(f,np.degrees(np.unwrap(np.arctan(np.imag(X)/np.real(X)))))
		plt.xlabel(r'$f$')
		plt.ylabel(r'$\angle X (\omega)$')
		if type(frange)==tuple:
			plt.xlim(frange)
		plt.show()


def hpfilter(x,dt,fc,fp):
	import numpy as np
	import numpy.fft as nf
	X=nf.rfft(x)
	N=X.size
	df=1/(dt*x.size)
	Np=int((N*df-fp)/df)
	Ntr=int((fp-fc)/df)
	Nzp=N-Np-Ntr
	ntr=np.linspace(0,Ntr,Ntr)
	Hp=np.ones(Np)
	Htr=np.sin(np.pi*ntr/(2*Ntr))
	Hzp=np.zeros(Nzp)
	xf=nf.irfft(X*np.hstack((Hzp,Htr,Hp)))
	return xf

def lpfilter(x,dt,fp,fc):
	import numpy as np
	import numpy.fft as nf
	X=nf.rfft(x)
	N=X.size
	df=1/(dt*x.size)
	Np=int(fp/df)
	Ntr=int((fc-fp)/df)
	Nzp=N-Np-Ntr
	ntr=np.linspace(0,Ntr,Ntr)
	Hp=np.ones(Np)
	Htr=np.cos(np.pi*(ntr/(2*Ntr)))
	Hzp=np.zeros(Nzp)
	xf=nf.irfft(X*np.hstack((Hp,Htr,Hzp)))
	return xf

def bpfilter(x,dt,fc1,fp1,fp2,fc2):
	import numpy as np
	import numpy.fft as nf
	from matplotlib.pyplot import plot
	X=nf.rfft(x)
	N=X.size
	f=freqs(N,dt)
	df=1/(dt*x.size)
	Np=int((fp2-fp1)/df)
	Ntr1=int((fp1-fc1)/df)
	Ntr2=int((fc2-fp2)/df)
	Nzp1=int(fc1/df)
	Nzp2=N-Nzp1-Ntr1-Np-Ntr2
	ntr1=np.linspace(0,Ntr1,Ntr1)
	ntr2=np.linspace(0,Ntr2,Ntr2)
	Hp=np.ones(Np)
	Htr1=np.sin(np.pi*(ntr1/(2*Ntr1)))
	Htr2=np.cos(np.pi*(ntr2/(2*Ntr2)))
	Hzp1=np.zeros(Nzp1)
	Hzp2=np.zeros(Nzp2)
	xf=nf.irfft(X*np.hstack((Hzp1,Htr1,Hp,Htr2,Hzp2)))
	return xf

def resfreqs(x,dt,n,frange=None):
	from numpy import abs
	f,X=ftrans(x,dt,frange)
	fres,Xres=pkfind(f,abs(X),n)
	return fres

def dispersion(x,t,d,fc1,fp1,fp2,fc2,alpha=1):
	from matplotlib.pyplot import ginput, plot, close, grid
	from numpy.fft import rfft,fft
	from numpy import finfo, zeros, hstack, unwrap, double, pi, angle, arctan2, imag, real
	close('all')
	eps=finfo(double).tiny
	fc1=1e6*fc1
	fp1=1e6*fp1
	fp2=1e6*fp2
	fc2=1e6*fc2
	dt=abs(t[-1]-t[-2])
	x=x-x[0]
	plot(t,x)
	grid(True)
	tx=ginput(4)
	close()
	x1=x[(t>=tx[0][0])&(t<=tx[1][0])]
	x2=x[(t>=tx[2][0])&(t<=tx[3][0])]
	toff=t[(t>=tx[1][0])&(t<=tx[2][0])]
	N1=len(x1)
	N2=len(x2)
	N3=len(toff)
	x1=hstack((x1*tukeywin(N1,alpha),zeros(N2+N3)))
	x2=hstack((zeros(N1+N3),x2*tukeywin(N2,alpha)))
	x1=bpfilter(hstack((x1*tukeywin(N1,alpha),zeros(N2+N3))),dt,fc1,fp1,fp2,fc2)
	x2=bpfilter(hstack((zeros(N1+N3),x2*tukeywin(N2,alpha))),dt,fc1,fp1,fp2,fc2)
	H=-rfft(x2)/rfft(x1)
	f=freqs(len(x1),dt)
	phi=unwrap(angle(H));
	phi=phi[(f>=fp1)&(f<=fp2)]
	f=f[(f>=fp1)&(f<=fp2)]
	c=-(4.*d*1e-3*pi*f)/phi
	f=1e-6*f[(f>=fp1)&(f<=fp2)]
	return f,c,phi

def moments(y,x=None,centre='mean'):

    from numpy import average,array,linspace,trapz
    from numpy.linalg import norm

    if x is None:

        x=linspace(0.,len(y)-1,len(y))

    m=[]

    m.append(trapz(y,x))
    y=y/m[0]

    if centre=='mean':
        m.append(trapz(x*y,x))
    elif centre=='mode':
        m.append(x[abs(y).argmax()])
    elif (type(centre)==float) or (type(centre)==int):
        m.append(centre)


    m.append(trapz(y*(x-m[1])**2,x))
    m.append(trapz(y*(x-m[1])**3,x)/(m[2]**1.5))
    m.append(trapz(y*(x-m[1])**4,x)/(m[2]**2))

    return m

def localmax(x):

    from numpy import diff,sign

    indmax=(diff(sign(diff(x))) < 0).nonzero()[0] + 1

    return indmax


def localmin(x):

    from numpy import diff,sign

    indmin=(diff(sign(diff(x))) > 0).nonzero()[0] + 1 # local min


    return indmin

def PeakLimits(x,indmax,db):

    from numpy import log10

    indleft = indmax
    indright = indmax

    indrightmax = len(x)

    xx=x.copy()

    xx=xx/xx[indmax]

    DB=0.

    while DB>db:

        indleft -=1

        if indleft == 0:

        	indleft = 0

        	break

        else:

        	DB=20*log10(xx[indleft])


    DB=0.

    while DB>db:

        indright +=1

        if indright==indrightmax:


        	indright = indrightmax

        	break

        else:

        	DB=20*log10(xx[indright])

    return indleft, indright


def SpacedArgmax(x,N,space):

    i=[]
    xx=x.copy()

    for n in range(N):

        I=xx.argmax()

        xx[I-space/2:I+space/2]=0.

        i.append(I)

    i.sort()

    return i

#def H1Pairs(x,dt,frng,pairs):
#    from numpy.fft import rfft
#    from numpy import linspace,angle,conj
#


def H1(x,y,dt,frng,uwphase=True,nfreqs='All'):
    """
    Returns a frequency vector, transfer function between x and y, and the wrapped phase of the transfer
    function
    """

    from numpy import zeros,conjugate,linspace, angle, unwrap, array, arange
    from numpy.fft import rfft

    # X = rfft(x,int(1/(df*dt)))
   #  Y = rfft(y,int(1/(df*dt)))
    X=rfft(x)

    Xc=conjugate(X)

    H=[]
    phi=[]
    f=linspace(0.,1/(2*dt),len(X))


    df=f[1]-f[0]
    if1=NearestValue(f,frng[0])
    if2=NearestValue(f,frng[1])

    if nfreqs is 'All':

        find=arange(if1,if2,1)

    else:

        find=linspace(if1,if2,nfreqs).astype(int)


    for i in range(len(y)):

        Y=rfft(y[i])
        HH=(Y*Xc)/(X*Xc)

        pphi=angle(HH)

        if uwphase:
            pphi=unwrap(angle(HH))


        H.append(HH[find])
        phi.append(pphi[find])

    f=f[find]

    f=array(f)
    H=array(H)
    phi=array(phi)

    return f,H,phi


def ZeroCrossings(x):

    from numpy import sign, where, array

    zc=array(where((sign(x[0:-1])*sign(x[1::]))<0.))

    zc=zc[0,:]

    return zc

def EchoSeparate(x,N,db=-40,ws=0.1):

    from numpy import zeros,array
    from scipy.signal import hilbert

    xa=hilbert(x.copy())
    Xa=abs(xa)

    ipks=LocalMax(Xa)

    pks=Xa[ipks]
    p=pks.copy()
    pmin=pks.min()
    ind=[]
    i=p.argmax()
    ind.append(ipks[i])
    p[i]=pmin
    il,ir=PeakLimits(Xa,ipks[i],db)
    ispace=round((ir-il)/2)
    n=0

    while len(ind)<N:

        i=p.argmax()
        ii=ipks[i]
        if all([abs(I-ii)>ispace for I in ind]):
            ind.append(ii)
            n+=1

        p[i]=pmin

    ind.sort()
    i0=il.copy()

    x=x.copy()

    xe=[]
    xx=zeros(len(x))
    xx[il:ir]=x[il:ir]*tukeywin(ir-il,ws)
    xe.append(xx)

    for n in range(1,N):

        il,ir=PeakLimits(Xa,ind[n],db)
        xx=zeros(len(x))
        xx[il:ir]=x[il:ir]*tukeywin(ir-il,ws)

        xe.append(xx)

    xe=array(xe)
    xe=xe[:,i0::].transpose()

    return xe

def NearestValue(vec,val):

    return abs(vec-val).argmin()

def NPeakLimits(x,N,frac,nextzero=True):

    from scipy.signal import hilbert


    zc=ZeroCrossings(x)

    while True:

        ind=[]
        xr=abs(hilbert(x))


        for n in range(N):

            try:

                il,ir=PeakLimits(xr,xr.argmax(),frac)

            except:

                frac=frac*1.01
                break



            xr[il:ir]=0.

            if nextzero is True:

                il=zc[NearestValue(zc,il)]
                ir=zc[NearestValue(zc,ir)]

            ind.append((il,ir))


        if len(ind)==N:

            break


    ind.sort()

    return ind


def DecayConstant(x,N,md):

    from numpy import log, polyfit, array, diff

    ind,A=Delays(x,N,mindelay=md)
    print(diff(ind))

    dc=-1*polyfit(array(range(N)),log(abs(A)),1)[0]

    return dc

def DiffCentral(x):

    from numpy import zeros,diff

    dx=zeros(x.shape)

    dx[0]=2*(x[1]-x[0])
    dx[-1]=2*(x[-1]-x[-2])

    for i in range(1,len(dx)-1):

        dx[i]=x[i+1]-x[i-1]

    return dx

def LocalMax(x):

    from numpy import diff,sign

    indmax=(diff(sign(DiffCentral(x))) < 0).nonzero()[0] + 1

    return indmax

def binary_search(a, x, lo=0, hi=None):   # can't use a to specify default for hi
    from bisect import bisect_left

    hi = hi if hi is not None else len(a) # hi defaults to len(a)
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end

def AmplitudeDelayPhase(x,N,dt,scale=1,db=-40,ws=0.01, debug=False):
    from numpy import correlate,array,angle,zeros,real,imag, mean, argmax, linspace
    from numpy.linalg import norm
    from scipy.signal import hilbert
    from matplotlib.pyplot import plot, figure, show
    from bisect import bisect_left, bisect_right

    x = x - mean(x)
    X=x.copy()
    xa=abs(hilbert(X))
    try:
        il,ir=PeakLimits(xa,xa.argmax(),db)
    except IndexError:
        print("Unable to detect reference pulse")
        return [0 for i in range(0, N)],[0 for i in range(0, N)],[0 for i in range(0, N)]

    x1=zeros(len(X))
    win=tukeywin(ir-il,ws)
    x1[il:ir]=X[il:ir]*win
    x2=x.copy()/norm(x1)
    x1=x1/norm(x1)

    xa=ACorrelate(x1,x2,M=scale)
    xa=xa[abs(xa).argmax()::]
    Xa=abs(xa)

    if debug:
        # figure()
        plot(dt*linspace(0,len(Xa)-1,len(Xa)),Xa)
        show()
    width = int((ir-il))
    candidates = LocalMax(Xa)
    candidates.sort()
    inds = []
    for i in range(0, len(candidates)):
        lesser = 0
        greater = len(Xa)
        if candidates[i]-width > 0:
            lesser = candidates[i]-width
        if candidates[i]+width < len(Xa):
            greater = candidates[i]+width
        local_candidates = candidates[bisect_left(candidates, lesser):bisect_right(candidates, greater)]
        local_peaks = [Xa[local_candidate] for local_candidate in local_candidates]
        if max(local_peaks) == Xa[candidates[i]]:
            inds.append(candidates[i])
    if len(inds) > N:
        inds = sorted(inds, key=lambda ind: Xa[ind])[-1*N:]
    inds.sort()
    T=(dt/scale)*array(inds)
    A=Xa[inds]
    phi=angle(xa[inds])

    return A,T,phi

def ACorrelate(x,y,M=1):
    from numpy.fft import fft,ifft,fftshift
    from numpy import conj,zeros,array,max,hstack, argmax
    from matplotlib.pyplot import plot

    N = 1 << (argmax([max(array([len(x),len(y)])) & (1<<i) for i in range(0, 32)]) + 1)
    # N=max([len(x),len(y)])
    X=fft(x, 2*N)
    Y=fft(y, 2*N)

    Cyx=2*Y*conj(X)
    Cyx[0]=Cyx[0].copy()/2
    Cyx[N]=Cyx[N].copy()/2
    Cyx[N+1::]=0
    Cyx=hstack((Cyx.copy(),zeros(2*N*(M-1))))

    cyx=fftshift(ifft(Cyx))*M

    return cyx

def DSTFT(x,y,alpha,NFFT):

    from numpy.fft import rfft
    # from scipy.signal import tukey
    from numpy import array,conj

    N=len(x)

    w = tukeywin(N,alpha)

    X = rfft(w*x,n=NFFT)
    Xc = conj(X)

    yw = array([w*y[i:i+N] for i in range(len(y)-N)])

    H = (rfft(yw,n=NFFT,axis=1)*Xc)/(X*Xc)

    return H

def FFTLengthPower2(N):

    from numpy import ceil,log

    if N>0:

        NN = int(2**(ceil(log(N)/log(2))))

    else:

        NN = int(0)

    return NN

def CircularDeconvolution(x,y,dt,fmax,beta=None):

    from numpy.fft import rfft, irfft, fftshift, fft, ifft, fftfreq, ifftshift
    from numpy import floor,zeros,hstack,conj,pi,exp,linspace,ones,real,pi
    from scipy.signal import blackmanharris,kaiser,hann
    from matplotlib.pyplot import plot, show

    # mx = abs(x).argmax()
    # my = abs(y).argmax()
    # Nx = len(x)
   #  Ny = len(y)
    Nmin = len(x)+len(y)-1


    N = FFTLengthPower2((len(x)+len(y)-1))



    # X = rfft(hstack((x[mx::],zeros(abs(Ny-Nx)),x[0:mx-1])))
    X = fft(x,n=N)
    Sxx = fftshift(X*conj(X))
    # Y = rfft(y[m::],n=Ny)
    # Y = rfft(hstack((y[mx::],y[0:mx-1])))
    Y = fft(y,n=N)
    Syx = fftshift(Y*conj(X))

    f = fftshift(fftfreq(N,dt))

    fpass = [(f>=-fmax)&(f<=fmax)]

    H = Syx[fpass]/Sxx[fpass]



    if beta is None:

        H = hann(len(H))*H

    else:

        H = kaiser(len(H),beta)*H


    HH = zeros(N)+0.0*1j

    HH[fpass] = H.copy()

    H = ifftshift(HH)

    h = real(ifft(H))

    return h[0:Nmin], H

def WienerDeconvolution(x,y,Q=0.01,Nf=5):

    from numpy.fft import rfft, irfft
    from numpy import conj

    N = FFTLengthPower2(Nf*(len(x)+len(y)-1))

    X=rfft(x,n=N)
    Xc=conj(X)
    Sxx=X*Xc
    Q=Q*(abs(Sxx).max())
    Y=rfft(y,n=N)

    H = Y*Xc/(Sxx+Q)

    h = irfft(H)

    return h

def SparseDeconvolution(x,y,p,rtype='omp'):

    from numpy import zeros, hstack, floor, array, shape, sign
    from scipy.linalg import toeplitz, norm
    from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso

    xm = x[abs(x).argmax()]

    # x = (x.copy())/xm
    x = (x.copy())/xm
    x = x/norm(x)

    y = (y.copy())/xm

    Nx=len(x)
    Ny=len(y)

    X = toeplitz(hstack((x,zeros(Nx+Ny-2))),r=zeros(Ny+Nx-1))

    Y = hstack((zeros(Nx-1),y,zeros(Nx-1)))

    if (rtype=='omp')&(type(p)==int):

        model = OrthogonalMatchingPursuit(n_nonzero_coefs=p,normalize=True)

    elif (rtype=='omp')&(p<1.0):

        model = OrthogonalMatchingPursuit(tol=p,normalize=True)


    elif (rtype=='lasso'):

        model = Lasso(alpha=p)


    model.fit(X,Y)

    h = model.coef_
    b = model.intercept_

    r = Y-b
    r = r[int(len(x)/2)-1:int(len(x)/2)-1+len(y)]

    h = h[int(len(x)/2)-1:int(len(x)/2)-1+len(y)]

    return r,h


def Deconvolution(x,y,dt,frng,Nscale=5):

    from numpy.fft import rfft,irfft
    from numpy import conj, zeros, hstack, array, linspace, sin, cos, ones, pi

    ftrans = min([(frng[1]-frng[0]),(frng[3]-frng[2])])
    fs = 1/(2*dt)

    # print(ftrans)
  #   print(fs)


    N = FFTLengthPower2(Nscale*max([round(fs/ftrans),len(y)+len(x)-1]))

    X = rfft(x,n=N)
    # X = X[0:N/2]
    Xc = conj(X)
    Y = rfft(y,n=N)
    # Y = Y[0:N/2]

    Sxx = X*Xc
    Syx = Y*Xc

    f = linspace(0.,1/(2*dt),len(Sxx))
    df = f[1]-f[0]

    f1 = (f>=frng[0])&(f<=frng[1])
    f2 = (f>frng[1])&(f<frng[2])
    f3 = (f>=frng[2])&(f<=frng[3])

    F = zeros(len(Sxx))

    F[f1] = sin((pi/(2*(frng[1]-frng[0])))*(f[f1]-frng[0]))**2

    F[f2] = ones(len(f2.nonzero()[0]))

    F[f3] = cos((pi/(2*(frng[3]-frng[2])))*(f[f3]-frng[2]))**2

    H = (Syx/Sxx)*F

    h = irfft(H)


    return h

def GaussianWindow(x,Apply=True):

    from scipy.signal import hilbert
    from numpy import linspace,exp

    m = moments(abs(hilbert(x)),centre='mode')

    n = linspace(0,len(x)-1,len(x))

    w = exp(-(1/(2*m[2]))*(n-m[1])**2)

    if Apply==True:

        w = x*w


    return w

def stft(x, fs, framesz, hop):
	from scipy import hanning
	from scipy import array
	from scipy import fft

	framesamp = int(framesz*fs)
	hopsamp = int(hop*fs)
	w = hanning(framesamp)
	X = array([fft(w*x[i:i+framesamp]) for i in range(0, len(x)-framesamp, hopsamp)])

	return X

def PeakSharpen(x,Sigma,a,b):

	from scipy.ndimage.filters import gaussian_filter1d
	from scipy.signal import savgol_filter

	xs = gaussian_filter1d(x,Sigma)
	xs = xs+a*savgol_filter(xs,3,2,deriv=2)+b*savgol_filter(xs,5,4,deriv=2)

	return xs

def dBDrop(x,dB):

	from numpy import where,amax

	ind = where(x>=(amax(x))*(10**(dB/20)))[0]


	return ind[-1]-ind[0]

def CoeffDet(p,x,y):

	from numpy import polyfit,polyval,mean,inner

	e = (y-polyval(p,x))
	# e = e.reshape((1,len(e)))
	E = y-mean(y)
	# E = E.reshape((1,len(E)))

	R2 = 1-inner(e,e)/inner(E,E)

	return R2

def BimodalityCoefficient(x):

    from scipy.stats import kurtosis, skew

    if any(x>0.0):

        n = len(x)
        m3 = skew(x)
        m4 = kurtosis(x)

        b = (m3**2 + 1)/(m4+3*(((n-1)**2)/((n-2)*(n-3))))


    else:

        b=0.0

    return b

def Align(x,y,shifts):

    from numpy.fft import fft,ifft
    from numpy import linspace,real,conj,exp,pi,argmax

    s = linspace(0,1,len(y))
    X = fft(x)
    Y = fft(y)

    c = real(ifft(conj(Y)*X))[shifts]

    sh = shifts[argmax(abs(c))]

    xx = real(ifft(X*exp(1j*2*pi*sh*s)))

    return xx,sh

def ApplyShift(x,shift):

    from numpy.fft import fft,ifft
    from numpy import linspace,real,exp,pi

    X = fft(x)
    s = linspace(0,1,len(X))

    return real(ifft(X*exp(2*pi*1j*s*shift)))

# def ChunkAverage(x,N):
#
#     from numpy import mean,floor,zeros
#
#     M = floor(x.shape[0]/N).astype(int)
#
#     X = zeros((M,x.shape[1]))
#
#     for n in range(0,M):
#
#
#         X[n,:] = mean(x[n*N:n*N+N,:],axis=0)
#
#
#     return X

def ChunkAverage(x,N):

    from numpy import mean

    M = int(len(x)/N)*N

    X = [mean(x[n:n+N]) for n in range(0,M,N)]

    return X

def PolyCrop(Img,ind):

    from matplotlib.path import Path
    from numpy import array,dstack,amax,meshgrid,linspace,hstack


    # Istack = dstack(array(I)
    # imshow(Img)
    #
    # ind = ginput(n=0,timeout=0)
    #
    # close()

    ind = array(ind)
    ind = ind.astype('int')

    I = Img.copy()

    I = I[ind[:,1].min():ind[:,1].max(),ind[:,0].min():ind[:,0].max()]

    ind[:,1] = ind[:,1]-ind[:,1].min()
    ind[:,0] = ind[:,0]-ind[:,0].min()

    p = Path(ind)

    Ish = I.shape

    ix,iy = meshgrid(linspace(0,Ish[1]-1,Ish[1]),linspace(0,Ish[0]-1,Ish[0]))

    sel = hstack((ix.reshape((Ish[1]*Ish[0],1)),iy.reshape((Ish[0]*Ish[1],1))))

    mask = p.contains_points(sel).reshape(Ish)

    I = I*mask.astype(int)

    return I

def ClosestIndex(x,val):
    from numpy import array

    return abs(array(x)-val).argmin()

def ClosestValue(x,val):

    return x[ClosestIndex(x,val)]

def GaussianBP(x,dt,fband):

    from numpy.fft import rfft, irfft
    from numpy import linspace,log,exp

    X = rfft(x,axis=-1)

    f = linspace(0,1/(2*dt),X.shape[-1])

    a = -4*log(0.5)/(fband[1]-fband[0])**2

    W = exp(-a*(f-(fband[1]+fband[0])/2)**2).reshape((1,len(f)))

    return irfft(W*X,axis=-1)

def BoundingBox(ind):

    from numpy import array

    Ind = array(ind)

    return array([[Ind[:,0].min(),Ind[:,0].max()],[Ind[:,1].min(),Ind[:,1].max()]])

def  AutoThreshold(Img,nbins=256):

    from skimage.filters import threshold_otsu
    from numpy import histogram,sum,mean,linspace,std,sqrt

    thr = threshold_otsu(Img)


    h = histogram(Img.ravel(),bins=nbins)

    p = h[0]/sum(h[0])

    i = linspace(mean([h[1][0],h[1][1]]),mean([h[1][-2],h[1][-1]]),len(p))

    i1 = i[i<thr]
    i2 = i[i>=thr]

    p1 = p[i<thr]
    p2 = p[i>=thr]

    w1 = sum(p1)
    w2 = sum(p2)

    mu1 = sum(p1*i1)/w1
    mu2 = sum(p2*i2)/w2

    vb = w1*w2*(mu1-mu2)**2

    mu = sum(p*i)

    v = sum(p*(i-mu)**2)

    s = vb/(v-vb)

    n = Img[Img<thr].flatten()

    sig = Img[Img>=thr].flatten()

    musig = mean(sig)
    mun = mean(n)

    cv=(musig-mun)/std(n)


    return thr,(s,cv)

def BytesToFloat(x,depth,dB=0):

    from numpy import array


    converter = {}

    converter['8'] = lambda x : array([xx-2**7 for xx in x]).astype(float)*(1/(2**8*10**(dB/20)))

    converter['16'] = lambda x : array([x[i]+x[i+1]*256 - 2**15 for i in range(0,len(x),2)]).astype(float)*(1/(2**16*10**(dB/20)))

    return converter[str(depth)](x)

def GaussianBlob(Nx,Ny,dx,dy,lx,ly,xc,yc,theta):

    from numpy import meshgrid,linspace,pi,cos,sin,exp

    x,y = meshgrid(linspace(0,Nx*dx,Nx),linspace(0,Ny*dy,Ny))

    theta = theta*(pi/180)

    sx = lx/2.35482
    sy = ly/2.35482

    a = 0.5*((cos(theta)/sx)**2 + (sin(theta)/sy)**2)

    b = 0.25*sin(2*theta)*(-1/sx**2 + 1/sy**2)

    c = 0.5*((sin(theta)/sx)**2 + (cos(theta)/sy)**2)

    z = exp(-(a*(x-xc)**2 +2*b*(x-xc)*(y-yc) + c*(y-yc)**2))

    return z

def RectifiedNoise(N,sigma,mu=0):

    from numpy.random import randn
    from scipy.signal import hilbert

    n = abs(hilbert(sigma*randn(N[0]*N[1])+mu))

    return n.reshape((N[0],N[1]))

def RMS(x):

    from numpy import sqrt,mean

    return sqrt(mean(x**2))

def ImageSNR(I,thresh=None,kind='mean'):

    from numpy import mean,log10,amax
    from skimage.filters import threshold_otsu

    if thresh==None:

        th = threshold_otsu(I)


    s = I[I>=th].ravel()
    n = I[I<th].ravel()

    snr={'mean':20*log10((mean(s)-mean(n))/RMS(n)),'peak':20*log10((amax(s)-mean(n))/RMS(n))}

    return snr[kind]

def UnravelJagged(x):

    from numpy import array

    X = array([xx[i] for xx in x for i in range(len(xx))])

    return X

def PrincipalImageMoments(I,dx,dy):

    from numpy import meshgrid,sum,linspace,array,argsort
    from numpy.linalg import eig

    Ny,Nx = I.shape

    x,y = meshgrid(dx*linspace(0,Nx-1,Nx),dy*linspace(0,Ny-1,Ny))

    A=dx*dy*sum(sum(I))

    In = I/A

    xc = dx*dy*sum(sum(In*x,axis=1))

    yc = dx*dy*sum(sum(In*y,axis=0))

    sxx = dx*dy*sum(sum(In*(x-xc)*(x-xc),axis=1))

    sxy = dx*dy*sum(sum(In*(x-xc)*(y-yc),axis=1),axis=0)

    syy = dx*dy*sum(sum(In*(y-yc)*(y-yc),axis=0))

    V = array([[sxx,sxy],[sxy,syy]])

    w,v = eig(V)

    iord = argsort(abs(w))

    v = v[:,iord]

    w = w[iord]

    d = {'Energy':A,'Centroid':(xc,yc),'PrincipalValues':w,'PrincipalAxes':v}

    return d

def ShiftImage(I,pt):

    from numpy.fft import fftn,ifftn
    from numpy import zeros,real
    from scipy.ndimage import fourier_shift

    Np0 = abs(int(pt[0]))
    Np1 = abs(int(pt[1]))

    NI0,NI1 = I.shape

    Izp = zeros((2*(Np0+1)+NI0,2*(Np1+1)+NI1))
    Izp[Np0+1:Np0+1+NI0,Np1+1:Np1+1+NI1] = I

    II = real(ifftn(fourier_shift(fftn(Izp), pt)))

    II = II[Np0+1:Np0+1+NI0,Np1+1:Np1+1+NI1]

    return II

def CheckBounds(x,bndlist):

    return all([(x[i]>=bndlist[i][0])&(x[i]<=bndlist[i][1]) for i in range(len(x))])

def SpikingDeconvolution(x,N,alpha=1.,eps=0.01):

    from numpy import log,real,conj,exp,floor,concatenate,zeros
    from numpy.fft import fft,ifft,fftshift,ifftshift
    from matplotlib.pylab import plot,show
    from scipy.signal import tukey

    # X = fft(concatenate((zeros(len(x)),x)))

    X = fft(x)
    #
    R = X*conj(X)

    # R = fft(correlate(x,x))

    # plot(ifft(R))
    # show()

    U = log(R)


    UU = ifft(U)

    # plot(real(UU))
    #
    # show()


    i0 = int(floor(len(UU)/2))+1
    UU[1:i0] = 2*UU[1:i0]

    UU[i0::] = 0.+0j

    # W = exp(real(fft(UU))/2)

    W = exp(fft(UU)/2)


    w = fftshift(real(ifft(W)))

    print(int(N/2))

    a = i0-int(N/2)

    w = w[a:a+N]*tukey(N,alpha)

    # # X = fft(concatenate((zeros(int(N/2)),x,zeros(int(N/2)))))
    #
    # X = fft(concatenate((x,zeros(N-1))))
    #
    #
    # w = fftshift(concatenate((zeros(int((len(X)-N)/2)),w,zeros(int((len(X)-N)/2)+1))))
    #
    # # plot(w)
    # # show()
    #
    # # W = fft(w)
    #
    # # xx = concatenate((zeros(int(N/2)),x,zeros(int(N/2))))
    # #
    # # plot(xx)
    # #
    # # plot(concatenate((w,zeros(len(xx)-len(w)))))
    # #
    # # show()
    #
    # W = fft(w)
    #
    # H = X/(W+eps*max(abs(W)))
    #
    # H[0] = 0+0j
    #
    #
    # # h = real(ifft(H))[int(N/2):int(N/2)+len(x)]
    #
    # h = real(ifft(H))



    return w


def MatMultiply(a):

    from numpy.linalg import dot
    from functools import reduce

    return reduce(lambda x,y: dot(y,x), (aa for aa in a))

def BeamSize(f,c,D,F):

    N = (D**2)/(4*(c/f))

    SF = F/N

    D6dB = 0.2568*D*SF

    Fz = (N*SF**2)*(2/(1+0.5*SF))

    return D6dB, Fz

# def LambDispersion(h,f,cL,cT,krange,mode='Antisymmetric'):
#
#     from scipy.optimize import brentq
#     from numpy import tan,sqrt,pi
#
#     # def p(k):
#     #
#     #     return sqrt(((2*pi*f/cL)**2 - k**2)+0j)
#     #
#     # def q(k):
#     #
#     #     return sqrt(((2*pi*f/cT)**2 - k**2)+0j)
#
#     # p = lambda k: sqrt(((2*pi*f/cL)**2 - k**2)+0j)
#     #
#     # q = lambda k: sqrt(((2*pi*f/cT)**2 - k**2)+0j)
#
#
#     # if mode == 'Antisymmetric':
#
#     if mode =='Antisymmetric':
#
#         def g(k):
#
#             p = sqrt(((2*pi*f/cL)**2 - k**2)+0j)
#             q = sqrt(((2*pi*f/cT)**2 - k**2)+0j)
#
#             return (4*p*q*k**2)*tan(q*h) + tan(p*h)*(q**2 - k**2)**2
#
#     elif mode =='Symmetric':
#
#         def g(k):
#
#             p = sqrt(((2*pi*f/cL)**2 - k**2)+0j)
#             q = sqrt(((2*pi*f/cT)**2 - k**2)+0j)
#
#
#             return (4*p*q*k**2)*tan(p*h) + tan(q*h)*(q**2 - k**2)**2
#
#     return brentq(g,krange[0],krange[1])


def LambDispersion(h,k,f,cL,cT,mode='Antisymmetric'):

    from scipy.optimize import brentq
    from numpy import tan,sqrt,pi,real,abs

    # def p(k):
    #
    #     return sqrt(((2*pi*f/cL)**2 - k**2)+0j)
    #
    # def q(k):
    #
    #     return sqrt(((2*pi*f/cT)**2 - k**2)+0j)

    # p = lambda k: sqrt(((2*pi*f/cL)**2 - k**2)+0j)
    #
    # q = lambda k: sqrt(((2*pi*f/cT)**2 - k**2)+0j)


    # if mode == 'Antisymmetric':

    k = k.reshape(-1,1)

    f = f.reshape(1,-1)

    p = sqrt(((2*pi*f/cL)**2 - k**2)+0j)
    q = sqrt(((2*pi*f/cT)**2 - k**2)+0j)

    h = h/2.


    if mode =='Antisymmetric':

        return abs((4*p*q*k**2)*tan(q*h) + tan(p*h)*(q**2 - k**2)**2)


    elif mode =='Symmetric':


        return abs((4*p*q*k**2)*tan(p*h) + tan(q*h)*(q**2 - k**2)**2)

    else:

        return None

# def EstimateWedgeParameters(x0,p,n,m,Tmn,dmn,cw=2.33):
#
#     from numpy import cos,sin,pi,array,arcsin
#     from scipy.optimize import root
#
#
#
#     # x0 = [x0[0], sin(pi*x0[1]/180.)]
#     # m = array(m)
#     # n = array(n)
#     # Tmn = array(Tmn)
#     # dmn = array(dmn)
#     #
#     # # f = lambda x: (cw**2)*(Tmn - dmn)**2 - (cos(x[0]*pi/180.)*(n - m)*p)**2 - (2*x[1] + p*(m+n)*sin(x[0]*pi/180.))**2
#     #
#     # f = lambda x: (cw*(Tmn - dmn))**2 - 4*x[0]**2 - 2*x[0]*p*(m+n)*x[1] - (n**2 + m**2)*p**2 - 4*m*n*(p*x[1])**2 + 2*m*n*p**2
#     #
#     # R = root(f,x0)
#
#     print(R['x'])
#
#
#
#     xr = (R['x'][0], 180.*arcsin(R['x'][1])/pi)
#
#     return R['success'], xr


def EstimateWedgeParameters(x,fs,n,d,p,cw=2.33):

    from numpy import cos,sin,pi,array,arcsin,abs, argmax
    from matplotlib.pylab import plot, close, ginput

    h = []
    phi = []

    for nn in n:

        xm = abs(x[nn[0],nn[0],:])
        xn = abs(x[nn[1],nn[1],:])

        plot(xm)
        plot(xn)

        inds = ginput(4,timeout=0)

        close()

        Tm = (argmax(xm[int(inds[0][0]):int(inds[1][0])]) + int(inds[0][0]))/fs
        Tn = (argmax(xn[int(inds[2][0]):int(inds[3][0])]) + int(inds[2][0]))/fs

        sphi = (Tn - Tm + d[nn[0]] - d[nn[1]])*cw/(2*p*(nn[1]-nn[0]))

        h.append((Tm - d[nn[0]])*cw/2. - p*nn[0]*sphi)
        h.append((Tn - d[nn[1]])*cw/2. - p*nn[1]*sphi)

        phi.append(180.*arcsin(sphi)/pi)


    return array(h),array(phi)

# 
# def InverseLogNormalCDF(p,mu,sigma):
#
#
#     0.5 + 0.5*erf((log(x)-mu)/(sqrt(2)*sigma)

# def CycloidIntersections(X,R):
#
#     from scipy.optimize import brentq
#     from numpy import amin, amax, sqrt, array, nan, arctan2, arccos, pi, zeros, cos, sin, tan
#
#     # r = R/2
#
#     # x0 = r*pi
#
#     def x(t):
#
#         # return r*(t-sin(t))
#
#         return R*t + r*sin(t)
#
#     def y(t):
#
#         # return r*(1-cos(t))
#
#         return r*cos(t)
#
#     def f(t,X):
#
#         # return y(t)*sin(t) - (X - x(t))*(cos(t)-1)
#
#
#
#
#     # a = 1e-3
#     # b = 2*R
#
#     # def g(x,X):
#     #
#     #     return (f(x)-X)*dfdx(x)
#
#     # def f(y):
#     #
#     #     return x0-r*arccos(1-y/r) + sqrt(y*(2*r-y))
#     #
#     # def g(y,X):
#     #
#     #     return -(x0 - r*arccos(1-y/r) + sqrt(y*(2*r-y)) - X)*(1/sqrt(1-(1-y/r)**2) + (y-r)/sqrt(y*(2*r-y))) - y
#     #
#     #
#     # # yi = [brentq(g,a,b,args=xx,full_output=True) for xx in X]
#     # #
#     # # yi = array([yy[0] if yy[1].converged else nan for yy in yi])
#     #
#     ti = zeros(len(X))
#     #
#     for i in range(len(X)):
#
#         try:
#
#             t = brentq(f,1e-6,pi,args=X[i],full_output=True)
#
#             if t[1]:
#
#                 ti[i] = t[0]
#
#             else:
#
#                 ti[i] = nan
#
#         except:
#
#             ti[i] = nan
#     #
#     # xi =
#
#     return x(ti),y(ti)
