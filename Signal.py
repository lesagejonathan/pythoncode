from numpy import *
from numpy.fft import *
from scipy.signal import *
from functools import reduce
from numpy.linalg import *
from skimage.feature import match_template
from skimage.filters import threshold_li


def EstimateReference(x,N,alpha=1.,eps=1e-6):


    # X = fft(concatenate((zeros(len(x)),x)))

    X = fft(x)
    #
    R = X*conj(X)

    R = R+max(abs(R))*eps


    # R = fft(correlate(x,x))

    # plot(ifft(R))
    # show()

    # R[abs(R)<1e-6]

    U = log(R)

    # U = concatenate((array([0.+0j]),log(R[1::])))


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

    a = argmax(abs(w))-int(N/2)

    w = w[a:a+N]*tukey(N,alpha)

    return w

def PeakBinarize(x,thresh='auto',M=3):

    from skimage.filters import threshold_li

    xh = hilbert(x)

    indpks = argrelmax(abs(xh),order=M)

    if thresh is 'auto':

        thresh = threshold_li(abs(xh[indpks]))

    # indpks = indpks[argsort(abs(xh[indpks]))[0]]
    #
    # indpks = indpks[-N::]
    #
    xb = zeros(len(x))
    #
    for i in indpks:

        if (i>M)&(i<len(xb)-M)&(abs(xh[i])>thresh):

            xb[i-int(round(M/2)):i+int(round(M/2))] = 1.

    return xb

def ToApproxZeroPhase(x,N):

    xx = zeros(N)

    imax = argmax(abs(x))

    x0 = x[imax::]
    xend = x[0:imax]

    xx[0:len(x0)] = x0
    xx[N-len(xend)::] = xend

    return xx


def FitReflectors(x,y,delays,fs):

    N = len(x)+len(y)-1

    X = rfft(ToApproxZeroPhase(x,N))

    X = X.reshape((len(X),1))

    Y = rfft(y,N)

    Y = Y.reshape((len(Y),1))

    f = linspace(0.,fs/2,len(X))

    f = f.reshape((len(f),1))

    H = reduce(lambda x,y: hstack((x,y)), [X*exp(-2j*pi*f*T) for T in delays])

    A = lstsq(H,Y)

    Ym = dot(H,array(A[0]))

    R = Y-Ym

    r = irfft(R.ravel())
    ym = irfft(Ym.ravel())

    return A[0],A[1],r,ym

def NormalizedCrossCorrelate(x,y):

    xy = match_template(y.reshape((len(y),1)),x.reshape((len(x),1)),pad_input=True)

    return xy.ravel()

def MUSIC(x,y,T,M,fs):

    # from misc import MatMultiply
    from functools import reduce

    N = len(x)+len(y)-1
    Y = rfft(y,N)
    X = rfft(ToApproxZeroPhase(x,N))

    X = matrix(diag(X))

    f = matrix(linspace(0,fs/2,len(Y))).transpose()


    # print(f.shape)
    # f = f.reshape((len(f),1))

    # H = Y*conj(X)/(X*conj(X))

    R = outer(Y,conj(Y))

    w,v = eigh(R)

    indsort = argsort(abs(w))

    v = v[:,0:-M]

    v = matrix(v)
    vH = v.getH()

    XH = X.getH()

    # print(exp(2j*pi*f.transpose()).shape)
    # print(XH.shape)
    # print(f.getH().shape)
    # print(v.shape)
    # print(vH.shape)
    # print(X.shape)
    # print(f.shape)
    #
    # print(dot(exp(2j*pi*f.transpose()*0.),XH))


    def MatMultiply(a):

        # from numpy.linalg import dot
        # from functools import reduce

        return reduce(lambda x,y: dot(x,y), [aa for aa in a])

    P = array([1/(MatMultiply([exp(2j*pi*f.transpose()*TT),XH,v,vH,X,exp(-2j*pi*f*TT)])) for TT in T]).ravel()



    return P


def ShiftSignal(x,T,fs):

    Npad = int(ceil(abs(T)*fs))


    X = rfft(x,Npad+len(x),axis=-1)

    f = linspace(0,fs/2,len(X))

    xs = irfft(exp(-1j*2*pi*f*T)*X,axis=-1)

    return xs[0:len(x)]


def FitRayleigh(x,dvtol=1e-6):

    # xthresh = threshold_li(x)

    xthresh = mean(x)

    N = len(x)

    P = lambda x,sigma: x*exp(-0.5*(x/sigma)**2)/(sigma**2)

    def Weight(x,y,phi,sigma):

        # W = P(x,sigma[y])*((phi**y)*(1-phi)**(1-y))/(P(x,sigma[0])*phi + P(x,sigma[1])*(1-phi))

        num = P(x,sigma[y])*((phi**y)*(1-phi)**(1-y))

        print(num)

        den = (P(x,sigma[0])*phi + P(x,sigma[1])*(1-phi))

        print(den)

        # print((where(num>den)-where(num/den))
        # print(all(where(isnan(num/den))==where(num>den)))

        # print(where(isnan(num/den)))

        W = num/den

        W[isnan(W)] = 0.


        return W



    # W = lambda x,y,phi,sigma: (P(x,sigma[y])*((phi**y)*(1-phi)**(1-y)))/(P(x,sigma[0])*phi + P(x,sigma[1])*(1-phi))

    reltol = lambda x,y: abs(x-y)/(min(abs(array([x,y]))))

    # reltol = lambda x,y: abs(x-y)


    x0 = x[x<xthresh]
    x1 = x[x>=xthresh]


    p = sum(x0)/N

    s = (sqrt(sum(x0**2))/(2*len(x0)),sqrt(sum(x1**2))/(2*len(x1)))

    print(p)

    print(s)

    dv = dvtol+1.


    while dv > dvtol:

        # pp = sum(W(x,0,p,s))/N
        #
        # ss = (sqrt(sum(W(x,0,p,s)*x**2)/sum(W(x,0,p,s)))/2, sqrt(sum(W(x,1,p,s)*x**2)/sum(W(x,1,p,s)))/2)

        pp = sum(Weight(x,0,p,s))/N

        ss = (sqrt(sum(Weight(x,0,p,s)*x**2)/sum(Weight(x,0,p,s)))/2, sqrt(sum(Weight(x,1,p,s)*x**2)/sum(Weight(x,1,p,s)))/2)

        dv = max(abs(array([reltol(p,pp), reltol(s[0],ss[0]), reltol(s[1],ss[1])])))

        print(dv)

        print(pp)
        print(ss)

        s = ss
        p = pp

    return p,s

def NLargestPeaks(x,N,M):

    ind = argrelmax(x,order=M)[0]

    xpks = x[ind]

    ind = ind[argsort(xpks)[-N::]]

    ind = ind[::-1]

    return ind

def EstimateProbeDelays(Scans,fsamp,p,h,c=5.92):

    M = Scans.shape[0]
    N = Scans.shape[1]

    x = abs(hilbert(Scans,axis=2))

    W = int(round(fsamp*0.25*h/c))


    Delays = zeros((M,N))

    for m in range(M):

        for n in range(N):

            T = int(round(fsamp*(2*sqrt((0.5*(n-m)*p)**2 + h**2)/c)))

            Delays[m,n] = (argmax(x[m,n,T-W:T+W])+T-W - T)/fsamp


    return Delays


def EstimateAttenuation(x,y,fs,d,fband,fdepend=(0,1,4)):

    from scipy.signal import detrend
    from numpy.linalg import lstsq
    from numpy.fft import rfft
    from numpy import linspace, hstack, log, abs, ones

    NFFT = len(x)+len(y)-1

    X = rfft(detrend(x), NFFT)
    Y = rfft(detrend(y), NFFT)

    f = linspace(0.,fs/2,len(X))

    indf = (f>=fband[0]) & (f<=fband[1])

    X = X[indf].reshape(-1,1)
    Y = Y[indf].reshape(-1,1)

    f = f[indf].reshape(-1,1)

    F = ones(f.shape)


    for n in range(len(fdepend)):

        F = hstack((F,d*f**fdepend[n]))

    G = log(abs(X)/abs(Y))

    alpha = lstsq(F,G)[0]

    return alpha

# def EstimateUsableBandwidth(X,fs):
#
#     from misc import moments
#     from numpy import linspace
#
#     f = linspace(0.,fs/2, len(X))
