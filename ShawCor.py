from numpy.fft import *
from numpy.linalg import *
from numpy import *
from scipy.signal import *
from scipy.optimize import minimize
from matplotlib.pylab import plot,show

def FFTLengthPower2(N):

    return int(2**(ceil(log(N)/log(2))))


def localmax(x):

    indmax=(diff(sign(diff(x))) < 0).nonzero()[0] + 1

    return indmax


def localmin(x):

    indmin=(diff(sign(diff(x))) > 0).nonzero()[0] + 1 # local min

    return indmin

def rootind(x):

    sgn = sign(x)

    indroot = array(range(len(x)-1))

    indroot = indroot[sgn[0:-1]*sgn[1::] <= 0]

    return indroot

def moments(y,x=None,centre='mode'):

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


    m.append(trapz(y*(x-m[1])**2,x))
    m.append(trapz(y*(x-m[1])**3,x)/(m[2]**1.5))
    m.append(trapz(y*(x-m[1])**4,x)/(m[2]**2))

    return m


def SimulateEchoes(d,K,dt=0.02,c=[2.05,1.98,2.91,5.9],rho=[0.94,0.94,1.5,7.8],alpha=[0.089,0.092,0.001],f0=6.5,BW=4.,T0=0.5,T=5):

    Zb = rho[0]*c[0]
    Za = rho[1]*c[1]
    Zp = rho[2]*c[2]
    Zs = rho[3]*c[3]

    Ta = 2*d[0]/c[1]
    Tp = 2*d[1]/c[2]
    Ts = 2*d[2]/c[3]

    ba = c[1]*alpha[0]
    bp = c[2]*alpha[1]
    bs = c[3]*alpha[2]

    N = FFTLengthPower2(round(T/dt))

    f = linspace(0,1/(2*dt),N/2+1)

    Rba = (Za-Zb)/(Za+Zb)
    Tba = 4*Za*Zb/(Za+Zb)**2

    dR = 2j*pi*(Za*Zp/K)*f



    Rap = (Zp-Za+dR)/(Zp+Za+dR)
    Rpa = (Za-Zp+dR)/(Zp+Za+dR)

    Tap = 4*Za*Zp/(Za+Zp+dR)**2

    Rps = (Zs-Zp)/(Zs+Zp)
    Tps = 4*Zs*Zp/(Zs+Zp)**2

    A = (Tap/Rap)*Rps

    f1 = f0-BW/2
    f2 = f0+BW/2

    frng=(f>=f1)&(f<=f2)

    a = -4*log(0.5)/(BW**2)

    X = -exp(-a*(f-f0)**2)*exp(-2j*pi*T0*f)

    X = Rba*X/max(abs(X))

    Hp = Rap

    Hp = Rap+Tap*Rps*exp(-2j*pi*Tp*f)*exp(-bp*Tp*f)+Tap*Rps*Rpa*exp(-4j*pi*Tp*f)*exp(-2*bp*Tp*f)

    Hs = -Tap*Tps*exp(-2j*pi*Ts*f)*exp(-bs*Ts*f)*exp(-2j*pi*Tp*f)*exp(-2*bp*Tp*f)*(1 + Rps*Rpa*exp(-2j*pi*Tp*f)*exp(-2*bp*Tp*f))

    H = Hp + Hs

    Y = X*exp(-2j*pi*Ta*f)*exp(-ba*Ta*f)*H*Tba/Rba

    x = ifft(2*X,N)

    y = ifft(2*Y,N)

    t = linspace(0,dt*(len(y)-1),len(y))

    return t,x+y

def AdhesivePrimerFit(T1,T2,b1,b2,p):

    from scipy.linalg import lstsq


    f = p[0]
    X = p[1]
    Y = p[2]
    # T1 = p[3]
    # T2 = p[4]

    # A = exp(-b1*T1*f)*exp(-2j*pi*T1*f)*hstack((ones((len(f),1)),exp(-b2*T2*f)*exp(-2j*pi*T2*f),exp(-4j*pi*T2*f)*exp(-2*b2*T2*f)))

    A = hstack((exp(-b1*T1*f)*exp(-2j*pi*T1*f)*X,exp(-b1*T1*f)*exp(-2j*pi*T1*f)*X*exp(-b2*T2*f)*exp(-2j*pi*T2*f),exp(-b1*T1*f)*exp(-2j*pi*T1*f)*X*exp(-4j*pi*T2*f)*exp(-2*b2*T2*f)))


    v,r,rnk,sval = lstsq(A,Y)
    r = float(r[0])/len(f)


    return v,r


# def AdhesivePrimerFit(a1,a2,p):

#     f = p[0]
#     H = p[1]
#     T1 = p[2]
#     T2 = p[3]

#     relrtol = 1e-2
#     vtol = 1e-2
#     maxiter = 10

#     A = hstack((exp(-a1*T1*f)*exp(-2j*pi*T1*f),exp(-a1*T1*f)*exp(-a2*T2*f)*exp(-2j*pi*T1*f)*exp(-2j*pi*T2*f),H*exp(-2j*pi*T2*f)*exp(-a2*T2*f)))

#     v,r,rnk,sval = lstsq(A,H)

#     r = float(r)/len(f)

#     dv = 1.
#     dr = 1.

#     Niter = 0

#     while (dr > relrtol) & (dv > vtol) & (Niter < maxiter):

#             A = hstack((exp(-a1*T1*f)*exp(-2j*pi*T1*f),exp(-a1*T1*f)*exp(-a2*T2*f)*exp(-2j*pi*T1*f)*exp(-2j*pi*T2*f),H*exp(-2j*pi*T2*f)*exp(-a2*T2*f)))

#             W = diag(abs(1/(1-v[2,0]*exp(-2j*pi*T2*f)*exp(-a2*T2*f))**2).flatten())

#             vv,rr,rnk,sval = lstsq(dot(W,A),dot(W,H))

#             rr = float(rr)/len(f)

#             dr = abs((rr-r)/r)

#             dv = norm(vv-v)/norm(v)

#             v = vv

#             r = rr

#             Niter = Niter+1


#     return v,r

def AdhesivePrimerFitResidual(x,*params):

    v,r = AdhesivePrimerFit(x[0],x[1],x[2],x[3],params)

    return r

def ModelReconstruction(v,b1,b2,T1,T2,X,dt,N):

    s = linspace(0,1/(2*dt),int(floor(N/2)+1))

    # H = exp(-b1*T1*s)*exp(-2j*pi*s*T1)*(v[0,0]+v[1,0]*exp(-2j*pi*T2*s)*exp(-b2*T2*s)+v[2,0]*exp(-4j*pi*T2*s)*exp(-2*b2*T2*s))

    # # H = exp(-2j*pi*s*T1)*exp(-b1*T1*s)*(v[0,0]+v[1,0]*exp(-2j*pi*T2*s)*exp(-b2*T2*s))/(1-v[2,0]*exp(-b2*T2*s)*exp(-2j*pi*T2*s))


    # Y = H*X

    Y = exp(-b1*T1*s)*exp(-2j*pi*s*T1)*X*v[0,0]+v[1,0]*exp(-b1*T1*s)*exp(-2j*pi*s*T1)*X*exp(-2j*pi*T2*s)*exp(-b2*T2*s)+v[2,0]*exp(-b1*T1*s)*exp(-2j*pi*s*T1)*X*exp(-4j*pi*T2*s)*exp(-2*b2*T2*s)

    y = ifft(2*Y,N)

    return y,Y


def AdhesivePrimerFeatures(x,dt,frng,df=0.01,fprng=[2,10],d=[[1,2.5],[0.1,0.5]],c=[1.98,2.91],beta=[[0.14,0.22],[0.09,0.45]],exptype='immersion'):

    from scipy.optimize import brute

    from numpy.fft import fft

    if exptype=='immersion':

        gates=[(0.5,1.5,0.4,0.01),(1.75,4.25,3,0.01)]

    elif exptype=='contact':

        gates=[(0.,1.,0.4,0.01),(1.7,4.25,3,0.01)]


    F = []
    Hx = []



    NFFT = FFTLengthPower2(round(1/(df*dt)))

    for xx in x:

        try:

            if exptype=='immersion':

                xc = xx[abs(xx).argmax()::]

            elif exptype=='contact':

                xc = xx


            ig1= (int(gates[0][0]/dt),int(gates[0][1]/dt))
            ig2 = (int(gates[1][0]/dt),int(gates[1][1]/dt))

            iw = int(gates[0][2]/dt)

            im1 = abs(xc[ig1[0]:ig1[1]]).argmax()+ig1[0]
            im2 = abs(xc[ig2[0]:ig2[1]]).argmax()+ig2[0]

            il1,ir1 = (im1-iw,im1+iw)
            il2,ir2 = (im2-iw*gates[1][2],im2+iw*gates[1][2])

            b1 = mean(array(beta[0]))
            b2 = mean(array(beta[1]))


            T = dt*(im2-im1)


            x1 = xc[il1:ir1]
            x1 = detrend(x1)
            x1 = x1*tukey(len(x1),gates[0][3])

            x2 = xc[il2:ir2]
            x2 =detrend(x2)
            x2 = x2*tukey(len(x2),gates[1][3])


            im1 = abs(x1).argmax()
            im2 = abs(x2).argmax()


            X1 = rfft(x1,n=NFFT)
            X2 = rfft(hstack((zeros(il2-il1),x2)),n=NFFT)


            f = linspace(0.,1/(2*dt),NFFT/2+1)


            ff = (f>=frng[0])&(f<=frng[1])

            fphase = (f>=fprng[0])&(f<=fprng[1])

            fp = f[fphase]

            f = f[ff]

            df = f[1]-f[0]

            X1 = -X1

            H = X2/X1

            Hp = H[fphase]*exp(2j*pi*T*fp)

            phi = unwrap(angle(Hp))

            H = H[ff]

            G = log(abs(Hp))

            G1,G0 = polyfit(fp,G,1)

            A = exp(G0)

            dphi = detrend(savgol_filter(phi,3,2,deriv=1)*(1/df))


            indmin = localmin(detrend(imag(Hp)))

            indmax = localmax(detrend(imag(Hp)))

            ind = hstack((indmin,indmax))

            ind.sort()

            T2a = 1/(2*(mean(diff(ind))*df))

            indmin = localmin(dphi)

            indmin = indmin[argmin(dphi[indmin])]


            T2b = 1/(2*fp[indmin])

            T2 = array([T2b,3*T2b,5*T2b])

            T2 = T2[argmin(abs(T2-T2a))]

            print(T2)


            xx2 = ifft(2*X2,NFFT)


            T1 = [T-T2,T]

            # T1 = T-T2

            print(T1)

            X1 = X1[ff]
            X2 = X2[ff]

            param = (f.reshape((len(f),1)),X1.reshape((len(X1),1)),X2.reshape((len(X2),1)))



            RT = []

            for i in range(len(T1)):

                # param = (f.reshape((len(f),1)),H.reshape((len(H),1)),T1[i],T2)

                v,r = AdhesivePrimerFit(T1[i],T2,b1,b2,param)

                RT.append(r)

            iTmin = argmin(array(RT))

            T1 = T1[iTmin]

            # ranges = ((beta[0][0],beta[0][1]),(beta[1][0],beta[1][1]))

            # param = (f.reshape((len(f),1)),H.reshape((len(H),1)),T1,T2)

            # X1 = X1[ff]
            # X2 = X2[ff]

            # param = (f.reshape((len(f),1)),X1.reshape((len(X1),1)),X2.reshape((len(X2),1)))


            # R = brute(AdhesivePrimerFitPhaseResidual,ranges,param,Ns=10,finish=False)

            # R = minimize(AdhesivePrimerFitResidual,[b1,b2],args=param,bounds=[(beta[0][0],beta[0][1]),(beta[1][0],beta[1][1])],method='SLSQP')

            R = minimize(AdhesivePrimerFitResidual,[T1,T2,b1,b2],args=param,method='SLSQP')


            T1 = float(R.x[0])
            T2 = float(R.x[1])
            b1 = float(R.x[2])
            b2 = float(R.x[3])

            print(T1)
            print(T2)


            v,r = AdhesivePrimerFit(T1,T2,b1,b2,param)

            B0 = v[0,0]/v[1,0]
            B1 = v[2,0]/v[1,0]

            # x2m,X2m,Hm = ModelReconstruction(v,b1,b2,T1,T2,X1,dt,NFFT)

            x2m,X2m = ModelReconstruction(v,b1,b2,T1,T2,X1,dt,NFFT)


            # Hx.append([H,Hm[ff],xx2,x2m])

            Hx.append([X2,X2m,xx2,x2m])


            F.append([T1,T2,b1,b2,abs(B0),angle(B0),abs(B1),angle(B1)])

        except:

            pass

        Nfail = len(x) - len(F)

    return F,Nfail,Hx




class Pipe:

    def __init__(self,PipeId=None,BondStrength=[]):

        self.PipeId = PipeId
        self.BondStrength = BondStrength
        self.Signals = []
        self.Locations = []
        self.SteelThickness = []


    def ZeroMean(self):


        s=self.Signals.copy()

        for i in range(len(s)):

            self.Signals[i]=s[i]-mean(s[i])



    def LocationIndices(self,Locations):

        from numpy import array,tile
        from numpy.linalg import norm
        from copy import deepcopy

        L=array(deepcopy(self.Locations))
        NL=L.shape[0]

        ind=[]

        for l in Locations:
            ind.append(norm(L-tile(array(l),(NL, 1)),1).argmin())

        return ind

    def ReturnSignals(self,Locations):

        from copy import deepcopy

        x=array(deepcopy(self.Signals))

        ind=self.LocationIndices(Locations)

        return list(x[ind,:])


    def DeleteSignals(self,Locations):

        from numpy import array,delete

        ind=self.LocationIndices(Locations)

        x=list(delete(array(self.Signals),ind,0))
        l=list(delete(array(self.Locations),ind,0))
        self.Locations=l
        self.Signals=x


    def AddSignal(self,Signal,Location,WriteMode='Append',SamplingPeriod=None):

        if WriteMode is 'Append':

            (self.Signals).append(Signal)
            (self.Locations).append(Location)

        elif WriteMode is 'Overwrite':

            self.Signals=Signal
            self.Locations=Location

        if SamplingPeriod is not None:

            self.SamplingPeriod = SamplingPeriod


    def Export(self,Filename,Path):


        if Filename.split('.')[-1] == 'txt':

            from numpy import hstack,array,savetxt

            # Export Raw Data to a structured text file (comma delimited)
            data=hstack((array(self.Locations),array(self.Signals)))

            savetxt(Path+Filename,data,delimiter=',',header=str(self.PipeId)+','+str(self.BondStrength[0])+','+str(self.BondStrength[1])+','+str(self.SamplingPeriod))

        elif Filename.split('.')[-1] == 'p':

            from pickle import dump

            data={'PipeId':self.PipeId,'BondStrength':self.BondStrength,'Locations':self.Locations,'Signals':self.Signals,'SamplingPeriod':self.SamplingPeriod,'SteelThickness':self.SteelThickness}

            dump(data,open(Path+Filename,'wb'))

    def Load(self, File, Path):

        if File.split('.')[1] == 'txt':
            from numpy import loadtxt
            File = Path + File
            data = loadtxt(File,delimiter=',')

            self.Signals=list(data[:,2::])
            self.Locations=list(data[:,0:2])

            with open(File,'r') as f:

                header=f.readline()

            header=header[2::].split(',')

            self.PipeId = int(header[0])
            self.BondStrength = [float(header[1]),float(header[2])]
            self.SamplingPeriod = float(header[3].rstrip())

        elif File.split('.')[1] == 'p':

            from pickle import load

            pipe = load(open(Path + File,'rb'))

            if type(pipe) is dict:

                self.PipeId = pipe['PipeId']
                self.BondStrength = pipe['BondStrength']
                self.Signals = pipe['Signals']
                self.Locations = pipe['Locations']
                self.SamplingPeriod = pipe['SamplingPeriod']
                self.SteelThickness = pipe['SteelThickness']


class PipeSet:

    def __init__(self,Path):

        self.Path=Path
        self.Pipes = []


    def AddPipesByStrength(self,StrengthRanges):

        from os import listdir

        files=[f for f in listdir(self.Path) if f.endswith('.p')]

        for f in files:

            p=Pipe()
            p.Load(f,Path=self.Path)

            if any(abs(sum(array(StrengthRanges)-array(p.BondStrength),1))<1e-16):

                p.ZeroMean()
                self.Pipes.append(p)

    def AddPipesById(self,IdList=None):

        from os import listdir

        files=[f for f in listdir(self.Path) if f.endswith('.p')]

        if IdList==None:

            for f in files:

                p=Pipe()
                p.Load(f,Path=self.Path)

                self.Pipes.append(p)

        else:

            for f in files:

                p=Pipe()
                p.Load(f,Path=self.Path)

                if p.PipeId in IdList:

                    self.Pipes.append(p)


    def ExtractFeatures(self,ContactType,FrequencyRange,ScansPerPipe=None,rand=False):

        from random import sample

        for p in self.Pipes:

            if ScansPerPipe==None:

                NScans=len(p.Signals)

            else:

                NScans=ScansPerPipe

            if rand:

                ind = sample(range(len(p.Signals)),ScansPerPipe)

            else:

                ind=range(NScans)

            signals = [p.Signals[i] for i in ind]

            F,Nfail,R = AdhesivePrimerFeatures(signals,p.SamplingPeriod,frng=FrequencyRange,exptype=ContactType)

            p.Features=array(F)
            p.NFailed=Nfail
            p.Reconstructions=R



    def MakeTrainingSet(self,StrengthRanges,Scale='standard'):

        ''' StrengthRanges list of lists defining the Bond Strength Ranges defining each class '''

        from sklearn import preprocessing
        from numpy import zeros

        X=zeros((1,4))
        y=zeros(1)

        for p in self.Pipes:

            bs=mean(p.BondStrength)

            for i in range(len(StrengthRanges)):

                if StrengthRanges[i][0]<=bs<=StrengthRanges[i][1]:


                    X=vstack((X,p.Features))
                    y=hstack((y,i*ones(shape(p.Features)[0])))

        y=y[1::]

        self.y=y.astype(int)

        X=X[1::,:]

        if Scale=='standard':

            ss = preprocessing.StandardScaler()

            self.FeatureScaling = ss.fit(X)

            X = ss.transform(X)

        elif Scale=='robust':

            ss = preprocessing.RobustScaler()

            self.FeatureScaling = ss.fit(X)

            X = ss.transform(X)

        self.X=X

    def FitRBFClassifier(self,C_range=logspace(-3,3,4),gamma_range=logspace(-3,3,4),niter=10):

        from sklearn.cross_validation import StratifiedShuffleSplit
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm

        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(self.y, n_iter=niter, test_size=1/niter, random_state=42)
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.y)

        self.RBFClassifier = svm.SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'])
        self.RBFClassifier.fit(self.X,self.y)
        self.RBFScore = self.RBFClassifier.score(self.X,self.y)
        self.MaxDistance = max(self.RBFClassifier.decision_function(self.X))
        self.MinDistance = min(self.RBFClassifier.decision_function(self.X))

        self.RBFClassifierCVScore = grid.best_score_

    def FitLinearClassifier(self,C_range=logspace(-3,3,4),niter=10):

        from sklearn.cross_validation import StratifiedShuffleSplit
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm


        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(self.y, n_iter=niter, test_size=1/niter, random_state=42)
        grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.y)

        self.LinearClassifier = svm.SVC(kernel='linear',C=grid.best_params_['C'])
        self.LinearClassifierCVScore = grid.best_score_
