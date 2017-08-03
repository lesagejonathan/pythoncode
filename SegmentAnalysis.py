from numpy import *
from numpy.fft import *
# from matplotlib.pylab import *
from misc import *
from scipy.ndimage import *
# from skimage.filters import *
import os
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import pickle
from Signal import *

StackingFunctions = {}

StackingFunctions['Sum'] = lambda x,y: x+y
StackingFunctions['MaxSum'] = lambda x,y: [x,y][argmax(array([sum(x.ravel()),sum(y.ravel())]))]
StackingFunctions['MaxValue'] = lambda x,y: [x,y][argmax(array([max(x.ravel()),max(y.ravel())]))]


# def ShiftSignal(x,T,fs):
#
#     Npad = int(ceil(abs(T)*fs))
#
#
#     X = rfft(x,Npad+len(x),axis=-1)
#
#     f = linspace(0,fs/2,len(X))
#
#     xs = irfft(exp(-1j*2*pi*f*T)*X,axis=-1)
#
#     return xs[0:len(x)]
#
# def EstimateProbeDelays(Scans,fsamp,p,h,c=5.92):
#
#     M = Scans.shape[0]
#     N = Scans.shape[1]
#
#     x = abs(hilbert(Scans,axis=2))
#
#     W = int(round(fsamp*0.25*h/c))
#
#
#     Delays = zeros((M,N))
#
#     for m in range(M):
#
#         for n in range(N):
#
#             T = int(round(fsamp*(2*sqrt((0.5*(n-m)*p)**2 + h**2)/c)))
#
#             Delays[m,n] = (argmax(x[m,n,T-W:T+W])+T-W - T)/fsamp
#
#
#     return Delays


def CalibrateWedge(Scans,Delays,fsamp,p,c,angle,h):

    from scipy.signal import hilbert
    from numpy.linalg import lstsq

    M,N,L = Scans.shape

    sphi = sin(angle*pi/180.)

    x = abs(hilbert(Scans,axis=2))

    W = int(round(fsamp*0.25*h/c))

    T = zeros((M,1))


    for m in range(M):

        TT = int(round(fsamp*(2*(m*p*sphi + h)/c)))

        # print(TT)

        T[m] = (argmax(x[m,m,TT-W:TT+W])+TT-W)/fsamp - Delays[m,m]


    A = hstack((-2.*ones((M,1)),T))

    b = 2*linspace(0,M-1,M).reshape(-1,1)*sphi*p

    v = lstsq(A,b)[0]

    return v[0],v[1]


def ToCentreAperatureDelay(T,x,th,cw,N,p):

    return T + (2/cw)*(x[0]-N*p/2)*sin((pi/180.)*th[0]) + (2/cw)*(x[1]-N*p/2)*sin((pi/180.)*th[1])


def GetAngle(x,m,h,p,phi):

    sphi = sin(phi)
    cphi = cos(phi)
    tphi = sphi/cphi


    ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
    vtr = array([x-m*p*cphi,h+m*p*sphi])


    return (180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr)))


def CheckBounds(x,bndlist):

    return all([(x[i]>=bndlist[i][0])&(x[i]<=bndlist[i][1]) for i in range(len(x))])

def Standardize(x,ax=-1):

    return (x-mean(x,axis=ax))/std(x,ax)


def CapRayTrace(Modes,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalFL']
    l3 = WeldParameters['HorizontalFL']

    cs = {'L':5.9,'T':3.24}

    c = [cs[m] for m in Modes]

    phic = arctan2(l3,l2)


    a2 = arcsin(cos(phic)*c[1]/c[2]) - phic

    x3 = l1 - l2/2

    y3 = -(l3/l2)*(x3 - l1 + l2)

    x2 = x3 - (l0-y3)*tan(a2)

    a1 = arcsin(c[0]*sin(a2)/c[1])

    x1 = x2 - l0*tan(a1)

    a0 = arcsin((cw/c[0])*sin(a1))

    x0 = (x1-tan(a0)*h)/(cos(phi) + tan(a0)*sin(phi))

    print(x3)
    # print(x1)
    # print(x2)
    # print(x3)

    if (x0>=-l3/2)&(x0<=N*p+l3/2):

        T = 2*(sqrt((x0*cos(phi)-x1)**2 + (h+x0*sin(phi))**2)/cw + sqrt((x2-x1)**2 +l0**2)/c[0] + sqrt((x3-x2)**2 +(y3-l0)**2)/c[1] + (l1-x3)/c[2])

        n = array([(h+x0*sin(phi))*tan(phi),h+x0*sin(phi)])
        v = array([x1-x0*cos(phi),h+x0*sin(phi)])

        th = (180/pi)*sign(v[0]-n[0])*arccos(vdot(v,n)/(norm(v)*norm(n)))

        d = {'Delay': T, 'Angle': th, 'Intercept': x0}

    else:

        d = {}

    return d



def PlaneWaveDelays(Path,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    from scipy.optimize import minimize
    from numpy.random import sample

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalFL']
    l3 = WeldParameters['HorizontalFL']

    cs = {'L':5.9,'T':3.24}

    cphi = cos(phi)
    sphi = sin(phi)

    tphi = sphi/cphi

    def GetAngles(x,m):


        ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
        vtr = array([x[0]-m*p*cphi,h+m*p*sphi])

        nrc = array([(h+x[-1]*sphi)*tphi,h+x[-1]*sphi])
        vrc = array([x[-2]-x[-1]*cphi,h+x[-1]*sphi])

        return (180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr))),(180/pi)*sign(vrc[0]-nrc[0])*arccos(vdot(vrc,nrc)/(norm(vrc)*norm(nrc)))


    def ToCentreAperatureDelay(T,x,th):

        return T + (2/cw)*(x[0]-N*p/2)*sin((pi/180.)*th[0]) + (2/cw)*(x[1]-N*p/2)*sin((pi/180.)*th[1])


    def BCDCB(m,c):

        f = lambda x: sqrt((h + sphi*x[7])**2 + (cphi*x[7] - x[6])**2)/cw + sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt(l0**2 + (x[5] - x[6])**2)/c[5] + sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)/c[4] + sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)/c[3] + sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)/c[2] + sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (-l1 + x[2] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)/(c[2]*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) + (-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)), (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])/(c[3]*l2*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)) + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)), (x[4] - x[5] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[4])/l2**2)/(c[4]*sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)) + (-l1 + x[4] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])/l2**2)/(c[3]*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)), (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) - (x[4] - x[5])/(c[4]*sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)), -(cphi*x[7] - x[6])/(cw*sqrt((h + sphi*x[7])**2 + (cphi*x[7] - x[6])**2)) - (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)), (cphi*(cphi*x[7] - x[6]) + sphi*(h + sphi*x[7]))/(cw*sqrt((h + sphi*x[7])**2 + (cphi*x[7] - x[6])**2))])

        bnds = ((m*p*cphi,l1),(m*p*cphi, l1),(l1-l2,l1),(-l3,0.),(l1-l2,l1),(0.,l1),(0., l1),(0.,N*p))


        xi = [0.25*(bnds[0][1]-bnds[0][0]),0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]), 0.75*(bnds[4][1]-bnds[4][0]), 0.75*(bnds[5][1]-bnds[5][0]), 0.35*(bnds[6][1]-bnds[6][0]), 0.5*(bnds[7][1]-bnds[7][0])]

        # xi = [(b[0] + sample(1)*(b[1]-b[0]))[0] for b in bnds]
        #
        # print(bnds)
        # print(xi)

        res = minimize(f,xi,method='BFGS',jac=J)

        # res = minimize(f,xi,method='Newton-CG',jac=J,options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        # res = minimize(f,xi,method='BFGS')


        print(res.x)

        print(bnds)







        print(res.success)
        print(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]]))

        # if (res.success)&(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]])):

        if (res.success):


            # T = sqrt(res.fun)

            T = res.fun

            th = GetAngles(res.x,m)

            print(ToCentreAperatureDelay(T,res.x,th))
            print(th)

            d = [T,res.x[-1],th]

        else:

            d = []

        return d

    def BCDCBFB(m,c):

        f = lambda x: sqrt((h + sphi*x[9])**2 + (cphi*x[9] - x[8])**2)/cw + sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt(l0**2 + (x[7] - x[8])**2)/c[7] + sqrt(l0**2 + (x[6] - x[7])**2)/c[6] + sqrt(l0**2 + (x[5] - x[6])**2)/c[5] + sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)/c[4] + sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)/c[3] + sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)/c[2] + sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (-l1 + x[2] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)/(c[2]*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) + (-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)), (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])/(c[3]*l2*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)) + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)), (x[4] - x[5] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[4])/l2**2)/(c[4]*sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)) + (-l1 + x[4] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])/l2**2)/(c[3]*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[4])**2/l2**2)), (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) - (x[4] - x[5])/(c[4]*sqrt((x[4] - x[5])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)), (x[6] - x[7])/(c[6]*sqrt(l0**2 + (x[6] - x[7])**2)) - (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)), (x[7] - x[8])/(c[7]*sqrt(l0**2 + (x[7] - x[8])**2)) - (x[6] - x[7])/(c[6]*sqrt(l0**2 + (x[6] - x[7])**2)), -(cphi*x[9] - x[8])/(cw*sqrt((h + sphi*x[9])**2 + (cphi*x[9] - x[8])**2)) - (x[7] - x[8])/(c[7]*sqrt(l0**2 + (x[7] - x[8])**2)), (cphi*(cphi*x[9] - x[8]) + sphi*(h + sphi*x[9]))/(cw*sqrt((h + sphi*x[9])**2 + (cphi*x[9] - x[8])**2))])



        bnds = ((m*p*cphi,l1),(m*p*cphi, l1),(l1-l2,l1),(-l3,0.),(l1-l2,l1),(0,l1),(0., l1),(0., l1),(0., l1),(0.,N*p))

        xi = [0.25*(bnds[0][1]-bnds[0][0]),0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]), 0.75*(bnds[4][1]-bnds[4][0]), 0.75*(bnds[5][1]-bnds[5][0]), 0.5*(bnds[6][1]-bnds[6][0]), 0.25*(bnds[7][1]-bnds[7][0]), 0.1*(bnds[8][1]-bnds[8][0]), 0.5*(bnds[9][1]-bnds[9][0])]

        # xi = [(b[0] + sample(1)*(b[1]-b[0]))[0] for b in bnds]
        #
        # print(bnds)
        # print(xi)

        res = minimize(f,xi,method='BFGS',jac=J)

        # res = minimize(f,xi,method='Newton-CG',jac=J,options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        # res = minimize(f,xi,method='BFGS')


        print(res.x)

        print(res.fun)

        print(res.message)




        print(res.success)
        print(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]]))

        # if (res.success)&(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]])):

        if (res.success):


            # T = sqrt(res.fun)

            T = res.fun

            th = GetAngles(res.x,m)

            # print(th)

            print(ToCentreAperatureDelay(T,res.x,th))
            print(th)

            d = [T,res.x[-1],th]

        else:

            d = []

        return d

    def BFBCDCBFB(m,c):


        f = lambda x: sqrt((x[10] - x[11]*cphi)**2 + (x[11]*sphi + h)**2)/cw + sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt(l0**2 + (x[10] - x[9])**2)/c[9] + sqrt(l0**2 + (x[8] - x[9])**2)/c[8] + sqrt(l0**2 + (x[7] - x[8])**2)/c[7] + sqrt((x[6] - x[7])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[6])**2/l2**2)/c[6] + sqrt((l1 - x[6])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[6])**2/l2**2)/c[5] + sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[4])**2/l2**2)/c[4] + sqrt((x[3] - x[4])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)/c[3] + sqrt(l0**2 + (x[2] - x[3])**2)/c[2] + sqrt(l0**2 + (x[1] - x[2])**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]


        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[2] - x[3])/(c[2]*sqrt(l0**2 + (x[2] - x[3])**2)) - (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2)), (x[3] - x[4])/(c[3]*sqrt((x[3] - x[4])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)) - (x[2] - x[3])/(c[2]*sqrt(l0**2 + (x[2] - x[3])**2)), (-l1 + x[4] + l3*(-l1*l3 + l2*l3 + l2*x[5] + l3*x[4])/l2**2)/(c[4]*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[4])**2/l2**2)) + (-x[3] + x[4] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[4])/l2**2)/(c[3]*sqrt((x[3] - x[4])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[4])**2/l2**2)), (-l1*l3 + l2*l3 + l2*x[5] + l3*x[6])/(c[5]*l2*sqrt((l1 - x[6])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[6])**2/l2**2)) + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[4])/(c[4]*l2*sqrt((l1 - x[4])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[4])**2/l2**2)), (x[6] - x[7] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[6])/l2**2)/(c[6]*sqrt((x[6] - x[7])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[6])**2/l2**2)) + (-l1 + x[6] + l3*(-l1*l3 + l2*l3 + l2*x[5] + l3*x[6])/l2**2)/(c[5]*sqrt((l1 - x[6])**2 + (-l1*l3 + l2*l3 + l2*x[5] + l3*x[6])**2/l2**2)), (x[7] - x[8])/(c[7]*sqrt(l0**2 + (x[7] - x[8])**2)) - (x[6] - x[7])/(c[6]*sqrt((x[6] - x[7])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[6])**2/l2**2)), (x[8] - x[9])/(c[8]*sqrt(l0**2 + (x[8] - x[9])**2)) - (x[7] - x[8])/(c[7]*sqrt(l0**2 + (x[7] - x[8])**2)), -(x[10] - x[9])/(c[9]*sqrt(l0**2 + (x[10] - x[9])**2)) - (x[8] - x[9])/(c[8]*sqrt(l0**2 + (x[8] - x[9])**2)), (x[10] - x[11]*cphi)/(cw*sqrt((x[10] - x[11]*cphi)**2 + (x[11]*sphi + h)**2)) + (x[10] - x[9])/(c[9]*sqrt(l0**2 + (x[10] - x[9])**2)), (-cphi*(x[10] - x[11]*cphi) + sphi*(x[11]*sphi + h))/(cw*sqrt((x[10] - x[11]*cphi)**2 + (x[11]*sphi + h)**2))])

        bnds = ((m*p*cphi,l1),(m*p*cphi, l1),(m*p*cphi,l1),(m*p*cphi,l1),(l1-l2,l1),(-l3,0.),(l1-l2,l1),(0.,l1),(0., l1),(0., l1),(0., l1),(0.,N*p))

        xi = [0.1*(bnds[0][1]-bnds[0][0]),0.2*(bnds[1][1]-bnds[1][0]), 0.3*(bnds[2][1]-bnds[2][0]), 0.4*(bnds[3][1]-bnds[3][0]),0.5*(bnds[4][1]-bnds[4][0]), 0.5*(bnds[5][1]-bnds[5][0]), 0.75*(bnds[6][1]-bnds[6][0]), 0.7*(bnds[7][1]-bnds[7][0]), 0.5*(bnds[8][1]-bnds[8][0]), 0.3*(bnds[9][1]-bnds[9][0]), 0.2*(bnds[10][1]-bnds[10][0]), 0.5*(bnds[11][1]-bnds[11][0])]

        # xi = [(b[0] + sample(1)*(b[1]-b[0]))[0] for b in bnds]
        #
        # print(bnds)
        # print(xi)

        res = minimize(f,xi,method='BFGS',jac=J)

        # res = minimize(f,xi,method='Newton-CG',jac=J,options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        # res = minimize(f,xi,method='BFGS')


        print(res.x)

        print(res.fun)

        print(res.message)




        print(res.success)
        # print(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]]))

        print(CheckBounds(res.x,bnds))

        # if (res.success)&(CheckBounds([res.x[2],res.x[3],res.x[4],res.x[-1]],[bnds[2],bnds[3],bnds[4],bnds[-1]])):

        if (res.success):


            # T = sqrt(res.fun)

            T = res.fun

            th = GetAngles(res.x,m)

            # print(th)

            print(ToCentreAperatureDelay(T,res.x,th))
            print(th)

            d = [T,res.x[-1],th]

        else:

            d = []

        return d




    C = [cs[cc] for cc in Path[1]]

    delayfncts = {}
    delayfncts['BCDCB'] = lambda m: BCDCB(m,C)
    delayfncts['BCDCBFB'] = lambda m: BCDCBFB(m,C)
    delayfncts['BFBCDCBFB'] = lambda m: BFBCDCBFB(m,C)



    m = 0

    D = []

    while (len(D)==0)&(m<=N):

        D = delayfncts[Path[0]](m)

        print(m)
        m += 1

    if len(D)>0:

        return ToCentreAperatureDelay(D[0],(p*(m-1),D[1]),D[2])

    else:

        return []




def DirectDelays(Path,Elements,X,Y,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    from scipy.optimize import minimize

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalFL']
    l3 = WeldParameters['HorizontalFL']

    cs = {'L':5.9,'T':3.24}

    cphi = cos(phi)
    sphi = sin(phi)

    tphi = sphi/cphi


    def BackWall(m,c):

        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((X - x[1])**2 + (Y - l0)**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]
        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), -(X - x[1])/(c[1]*sqrt((X - x[1])**2 + (Y - l0)**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2))])

        bnds = ((m*p*cphi,l1),(m*p*cphi, l1))

        xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]))

        res = minimize(f,xi,method='BFGS',jac=J)


        if (res.success)&(CheckBounds([res.x[0],res.x[1]],[bnds[0],bnds[1]])):

            T = res.fun

            # x = [res.x[0],res.x[1]]

            d = T

        else:

            d = nan

        return d


    def FrontWall(m,c):

        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt(Y**2 + (X - x[2])**2)/c[2] + sqrt(l0**2 + (x[1] - x[2])**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]
        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), -(X - x[2])/(c[2]*sqrt(Y**2 + (X - x[2])**2)) - (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2))])

        bnds = ((m*p*cphi,l1-l2),(m*p*cphi, l1-l2),(m*p*cphi,l1-l2))

        xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]),0.75*(bnds[2][1]-bnds[2][0]))

        res = minimize(f,xi,method='BFGS',jac=J)


        if (res.success)&(CheckBounds([res.x[0],res.x[1],res.x[2]],[bnds[0],bnds[1],bnds[2]])):

            T = res.fun

            # x = [res.x[0],res.x[1]]

            d = {'Delay':T,'Angle':GetAngle(res.x[0],m,h,p,phi)}


        else:

            d = {'Delay':nan,'Angle':nan}

        return d


    def Cap(m,c):

        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((X - x[2])**2 + (Y*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[2] + sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (-X + x[2] + l3*(Y*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[2]*sqrt((X - x[2])**2 + (Y*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) + (-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2))])

        bnds = ((m*p*cphi,l1),(m*p*cphi, l1),(l1-l2,l1))

        xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]),0.5*(bnds[2][1]-bnds[2][0]))

        res = minimize(f,xi,method='BFGS',jac=J)


        if (res.success)&(CheckBounds([res.x[0],res.x[1],res.x[2]],[bnds[0],bnds[1],bnds[2]])):

            # T = res.fun

            # x = [res.x[0],res.x[1]]

            # d = {'Delay':T,'Angle':GetAngle(res.x[0],m,h,p,phi)}

            d = res.fun


        else:

            # d = {'Delay':T,'Angle':nan}

            d = nan

        return d

    def Direct(m,c):

        c = c[0]

        f = lambda x: 2*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x)**2)/cw + 2*sqrt(Y**2 + (X - x)**2)/c
        J = lambda x: 2*(-cphi*m*p + x)/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x)**2)) + 2*(-X + x)/(c*sqrt(Y**2 + (X - x)**2))

        bnds = ((m*p*cphi,l1),(m*p*cphi, l1))

        xi = (0.5*(bnds[0][1]-bnds[0][0]))

        res = minimize(f,xi,method='BFGS',jac=J)


        if (res.success)&(CheckBounds(res.x,bnds)):

            T = res.fun





            # x = [res.x[0],res.x[1]]

            d = {'Delay':T,'Angle':GetAngle(res.x[0],m,h,p,phi)}

        else:

            d = {'Delay':nan, 'Angle':nan}

        return d



    DelayFunctions = {}

    cf = [cs[cc] for cc in Path[0][1]]
    cb = [cs[cc] for cc in Path[1][1]]
    #
    DelayFunctions['BackWall'] = lambda m,c: BackWall(m,c)
    DelayFunctions['Cap'] = lambda m,c: Cap(m,c)
    # DelayFunctions['FrontWall'] = lambda m,c: FrontWall(m,c)
    # DelayFunctions['Direct'] = lambda m,c: Direct(m,c)


    # if Path[0][0] is Path[1][0]:
    #
    #     d = [2*DelayFunctions[Path[0][0]](m,cf) for m in Elements[0]]
    #
    # else:

    df = [ DelayFunctions[Path[0][0]](m,cf) for m in Elements[0]]
    db = [ DelayFunctions[Path[1][0]](m,cb) for m in Elements[1]]

    # print(df[0])
    # print(db[0])

    # d = [[(ddf['Delay'] + ddb['Delay'],(ddf['Angle'],ddb['Angle'])) for ddb in db] for ddf in df]

    # print(df[0])

    d = [[ddf + ddb for ddb in db] for ddf in df]



    return d


def SymmetricDelays(Path,Elements,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    from scipy.optimize import minimize

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalFL']
    l3 = WeldParameters['HorizontalFL']

    cs = {'L':5.9,'T':3.24}

    cphi = cos(phi)
    sphi = sin(phi)

    tphi = sphi/cphi

    def SideWallCap(m,c):

        f = lambda x: 2*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + 2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)/c[2] + 2*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + 2*sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2))])

        # H = lambda x: array([[2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))*(cphi*m*p - x[0])/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)**(3/2)) + 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2) + c[0]*(x[0] - x[1])*(-cphi*m*p + x[0])/sqrt(l0**2 + (x[0] - x[1])**2) + cw*(x[0] - x[1])*(-cphi*m*p + x[0])/sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2) + cw*sqrt((h + m*p*sphi)**2 +(cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 2*(-x[0] + x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 2*(c[0]*(-x[0] + x[1])*(-cphi*m*p + x[0])/sqrt(l0**2 + (x[0] - x[1])**2) - cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 2*(x[0] - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 0, 0], [2*(c[0]*(-l1 + x[1])*(x[0] - x[1])/sqrt(l0**2 + (x[0] - x[1])**2) - c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(l1 - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2) + c[0]*(-l1 + x[1])*(-x[0] + x[1])/sqrt(l0**2 + (x[0] - x[1])**2) + c[1]*(-l1 + x[1])*(-x[0] + x[1])/sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(x[0] - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(-l0 + x[2])*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l0 - x[2])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), 0], [0, 2*(-l1 + x[1])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l1 - x[1])*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), -2*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[1]*c[2]*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l0 - x[2])*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2*(c[1]*l2*sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[1]*(-l0 + x[2])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), -(l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(c[1]*l3*sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2])*(l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(2*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))], [0, 0, 2*l3/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)) - 2*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)), 2*(l2**2 + l3**2)/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)) - (l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2))]])

        bnds = ((m*p*cphi,l1-l2),(m*p*cphi, l1),(-l3,l0),(l1-l2,l1))

        # xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]),0.75*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]),(0.5*(bnds[6][1]-bnds[6][0])))

        xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]))

        # res = minimize(f,xi,method='Newton-CG',jac=J,hess=H,options={'disp': False, 'xtol': 1e-03, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        res = minimize(f,xi,method='BFGS',jac=J)


        if (res.success)&(CheckBounds([res.x[2],res.x[3]],[bnds[2],bnds[3]])):

            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])

            # nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            # vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])

            th = (180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr)))

            x = [res.x[0],res.x[1],res.x[3]]

            d = {'Delay':T,'Angle':th,'x':[res.x[0],res.x[1],res.x[3]],'y':res.x[2]}

        else:

            d = {}

            # T = nan
            # th = nan
            # x = nan
            # y = nan
            #
            # d = {'Delay':T}

        return d

    def CapSideWall(m,c):


        f = lambda x: 2*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + 2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)/c[2] + 2*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[1] + 2*sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 2*(x[1] - x[2])/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) - 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), 2*(c[1]*sqrt((l2**2*(x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2)/l2**2)*(l2**2*(-l1 + x[2]) + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])) + c[2]*sqrt((l2**2*(l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2)/l2**2)*(l2**2*(-x[1] + x[2]) + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])))/(c[1]*c[2]*l2**2*sqrt((l2**2*(l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2)/l2**2)*sqrt((l2**2*(x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2)/l2**2)), 2*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*sqrt((l2**2*(l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2)/l2**2))])

        bnds = ((m*p*cphi,l1-l2),(m*p*cphi, l1),(l1-l2,l1),(-l3,l0))

                # xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]),0.75*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]),(0.5*(bnds[6][1]-bnds[6][0])))

        xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]))

        res = minimize(f,xi,method='BFGS',jac=J)

        if (res.success)&(CheckBounds([res.x[2],res.x[3]],[bnds[2],bnds[3]])):

            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])

            # nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            # vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])

            th = (180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr)))

            x = [res.x[0],res.x[1],res.x[3]]

            d = {'Delay':T,'Angle':th,'x':[res.x[0],res.x[1],res.x[2]],'y':res.x[3]}

        else:

            d = {}

            # T = nan
            # th = nan
            # x = nan
            # y = nan
            #
            # d = {'Delay':T}

        return d



    DelayFunctions = {}

    c = [cs[cc] for cc in Path[1]]

    DelayFunctions['SideWallCap'] = lambda m,c: SideWallCap(m,c)
    DelayFunctions['CapSideWall'] = lambda m,c: CapSideWall(m,c)

    d = [DelayFunctions[Path[0]](m,c) for m in Elements]

    return d


def Delays(Path,Elements,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    from scipy.optimize import minimize

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalFL']
    l3 = WeldParameters['HorizontalFL']

    cs = {'L':5.9,'T':3.24}

    cphi = cos(phi)
    sphi = sin(phi)

    tphi = sphi/cphi

    def Corner(m,n,c):

        # T Minimization

        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]
        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), -(l1 - x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) - l0/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + x[2]/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), -(cphi*n*p - x[3])/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[3])**2)) - (l1 - x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))])
        H = lambda x: array([[-(cphi*m*p - x[0])**2/(cw*((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)**(3/2)) + 1/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), 0, 0], [-1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -(l1 - x[1])**2/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 1/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -(l0 - x[2])*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), 0], [0, -l0*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + x[2]*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), -x[2]**2/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)) + 1/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) - l0*(l0 - x[2])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + x[2]*(l0 - x[2])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 1/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), x[2]*(l1 - x[3])/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2))], [0, 0, x[2]*(l1 - x[3])/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)), -(cphi*n*p - x[3])**2/(cw*((h + n*p*sphi)**2 + (cphi*n*p - x[3])**2)**(3/2)) + 1/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[3])**2)) - (l1 - x[3])**2/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)) + 1/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))]])



        # T**2 Minization

        # f = lambda x: (c[0]*c[1]*c[2]*(sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2) + sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[3])**2)) + c[0]*c[1]*cw*sqrt(x[2]**2 + (l1 - x[3])**2) + c[0]*c[2]*cw*sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[1]*c[2]*cw*sqrt(l0**2 + (x[0] - x[1])**2))**2/(c[0]**2*c[1]**2*c[2]**2*cw**2)
        #
        # J = lambda x: array([(2*(-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (2*(-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (2*x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (2*(-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])])
        #
        # H = lambda x: array([[((-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*(-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))) + (2*(-cphi*m*p + x[0])*(cphi*m*p - x[0])/(cw*((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)**(3/2)) + 2/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)) + 2*(-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (-x[0] + x[1])**2)**(3/2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (-2/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)) + 2*(x[0] - x[1])**2/(c[0]*(l0**2 + (-x[0] + x[1])**2)**(3/2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]) + ((-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + (-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*(-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))), (x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + (-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*(2*(-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))), (2*(-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + 2*(x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*((-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + (-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)))], [(-2/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)) + 2*(-x[0] + x[1])**2/(c[0]*(l0**2 + (-x[0] + x[1])**2)**(3/2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]) + (2*(-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*((-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))), ((-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + (-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*(-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))) + (2*(-l1 + x[1])*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)) + 2*(-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (-x[0] + x[1])**2)**(3/2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (2*(-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + (-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))) + 2*(l0 - x[2])*(-l1 + x[1])*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), (2*(-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*((-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + (-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)))], [(2*x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*((-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2))), ((-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + (-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))) + 2*(-l0 + x[2])*(l1 - x[1])*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), (x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + (-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*(2*x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))) + (-2*x[2]**2/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)) + 2/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])*(l0 - x[2])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0]), (2*x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + 2*(-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*((-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + (-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))) + 2*x[2]*(l1 - x[3])*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2))], [((-cphi*m*p + x[0])/(cw*sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*(-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))), ((-l1 + x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + (-x[0] + x[1])/(c[0]*sqrt(l0**2 + (-x[0] + x[1])**2)))*(2*(-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))), (x[2]/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)) + (-l0 + x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)))*(2*(-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))) - 2*x[2]*(-l1 + x[3])*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)), ((-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + (-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)))*(2*(-cphi*n*p + x[3])/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2))) + (2*(-cphi*n*p + x[3])*(cphi*n*p - x[3])/(cw*((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)**(3/2)) + 2/(cw*sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)) + 2*(-l1 + x[3])*(l1 - x[3])/(c[2]*(x[2]**2 + (l1 - x[3])**2)**(3/2)) + 2/(c[2]*sqrt(x[2]**2 + (l1 - x[3])**2)))*(sqrt((h + m*p*sphi)**2 + (-cphi*m*p + x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (-cphi*n*p + x[3])**2)/cw + sqrt(x[2]**2 + (l1 - x[3])**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (-x[0] + x[1])**2)/c[0])]])

        bnds = ((m*p*cphi,m*p*cphi + tan(arcsin(cw/c[0]))*(h + m*p*sphi)),(m*p*cphi,l1),(0.,l0),(n*p*cphi, l1-l2))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]))


        res = minimize(f,xi,method='Newton-CG',jac=J,hess=H,options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        # res = minimize(f,xi,method='BFGS',jac=J)




        if (res.success)&(res.fun>0.)&(CheckBounds(res.x,bnds)):

            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])


            nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])


            th = ((180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr))), (180/pi)*sign(vrc[0]-nrc[0])*arccos(vdot(vrc,nrc)/(norm(vrc)*norm(nrc))))


            # th = ( arccos(dot(vtr.transpose(),ntr)[0]/(sqrt(dot(ntr,ntr.transpose())[0])*sqrt(dot(vtr,vtr.transpose())[0]))), arccos(dot(vrc,nrc.transpose())[0]/(sqrt(dot(nrc,nrc.transpose())[0])*sqrt(dot(vrc,vrc.transpose())[0]))))

        else:

            T = nan
            th = nan

        return T,th

    def SideWallCap(m,n,c):

        # f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)/cw + sqrt(l0**2 + (x[4] - x[5])**2)/c[4] + sqrt((l0 - x[3])**2 + (l1 - x[4])**2)/c[3] + sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)/c[2] + sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]
        # J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (-l1 + x[2] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)/(c[2]*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) + (-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)), -(l0 - x[3])/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)) + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)), (x[4] - x[5])/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) - (l1 - x[4])/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)), -(cphi*n*p - x[5])/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)) - (x[4] - x[5])/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2))])
        # H = lambda x: array([[-(cphi*m*p - x[0])**2/(cw*((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)**(3/2)) + 1/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), 0, 0, 0, 0], [-1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), (-x[1] + x[2])*(x[1] - x[2])/(c[1]*((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)**(3/2)) + 1/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), (x[1] - x[2])*(x[1] - x[2] - l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)**(3/2)) - 1/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)), 0, 0, 0], [0, (-x[1] + x[2])*(-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)**(3/2)) - 1/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)), (1 + l3**2/l2**2)/(c[2]*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) + (-l1 + x[2] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)*(l1 - x[2] - l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)/(c[2]*((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)**(3/2)) + (1 + l3**2/l2**2)/(c[1]*sqrt((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)) + (-x[1] + x[2] + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)*(x[1] - x[2] - l3*(l0*l2 - l1*l3 + l2*l3 + l3*x[2])/l2**2)/(c[1]*((x[1] - x[2])**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x[2])**2/l2**2)**(3/2)), l3/(c[2]*l2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) - (-l1 + x[2] + l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)**(3/2)), 0, 0], [0, 0, l3/(c[2]*l2*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) + (l1 - x[2] - l3*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/l2**2)*(-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])/(c[2]*l2*((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)**(3/2)), -(l0 - x[3])**2/(c[3]*((l0 - x[3])**2 + (l1 - x[4])**2)**(3/2)) + 1/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)) + 1/(c[2]*sqrt((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)) - (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/(c[2]*l2**2*((l1 - x[2])**2 + (-l1*l3 + l2*l3 + l2*x[3] + l3*x[2])**2/l2**2)**(3/2)), -(l0 - x[3])*(l1 - x[4])/(c[3]*((l0 - x[3])**2 + (l1 - x[4])**2)**(3/2)), 0], [0, 0, 0, -(l0 - x[3])*(l1 - x[4])/(c[3]*((l0 - x[3])**2 + (l1 - x[4])**2)**(3/2)), 1/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) + (-x[4] + x[5])*(x[4] - x[5])/(c[4]*(l0**2 + (x[4] - x[5])**2)**(3/2)) - (l1 - x[4])**2/(c[3]*((l0 - x[3])**2 + (l1 - x[4])**2)**(3/2)) + 1/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)), -1/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) + (x[4] - x[5])**2/(c[4]*(l0**2 + (x[4] - x[5])**2)**(3/2))], [0, 0, 0, 0, -1/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) - (-x[4] + x[5])*(x[4] - x[5])/(c[4]*(l0**2 + (x[4] - x[5])**2)**(3/2)), -(cphi*n*p - x[5])**2/(cw*((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)**(3/2)) + 1/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)) + 1/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) - (x[4] - x[5])**2/(c[4]*(l0**2 + (x[4] - x[5])**2)**(3/2))]])

        # bnds = ((m*p*cphi,m*p*cphi + tan(arcsin(cw/c[0]))*(h + m*p*sphi)),(l1-l2 - l0*l2/l3, l1),(l1-l2,l1),(0.,l0),(n*p*cphi,l1),(n*p*cphi,n*p*cphi + tan(arcsin(cw/c[4]))*(h + n*p*sphi)))

        # bnds = ((m*p*cphi,m*p*cphi + tan(arcsin(cw/c[0]))*(h + m*p*sphi)),(m*p*cphi, l1),(l1-l2,l1),(0.,l0),(n*p*cphi,l1),(n*p*cphi,n*p*cphi + tan(arcsin(c[4]/cw))*(h + n*p*sphi)))


        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[6])**2)/cw + sqrt(l0**2 + (x[5] - x[6])**2)/c[5] + sqrt((l0 - x[4])**2 + (l1 - x[5])**2)/c[4] + sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)/c[3] + sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)/c[2] + sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), -(l1 - x[1])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)) - (l0 - x[2])/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/l2**2)/(c[3]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)) + (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/l2**2)/(c[2]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)), -(l0 - x[4])/(c[4]*sqrt((l0 - x[4])**2 + (l1 - x[5])**2)) + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/(c[3]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)), (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) - (l1 - x[5])/(c[4]*sqrt((l0 - x[4])**2 + (l1 - x[5])**2)), -(cphi*n*p - x[6])/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[6])**2)) - (x[5] - x[6])/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2))])

        H = lambda x: array([[-(cphi*m*p - x[0])**2/(cw*((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)**(3/2)) + 1/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) + (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), 0, 0, 0, 0, 0], [-1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (-x[0] + x[1])*(x[0] - x[1])/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -(l1 - x[1])**2/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 1/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 1/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)) - (x[0] - x[1])**2/(c[0]*(l0**2 + (x[0] - x[1])**2)**(3/2)), -(l0 - x[2])*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), 0, 0, 0, 0], [0, -(l0 - x[2])*(l1 - x[1])/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), 1/(c[2]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)) - (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/(c[2]*l2**2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)**(3/2)) - (l0 - x[2])**2/(c[1]*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 1/(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), l3/(c[2]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)) + (l1 - x[3] - l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/l2**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)**(3/2)), 0, 0, 0], [0, 0, l3/(c[2]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)) - (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/l2**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)**(3/2)), (1 + l3**2/l2**2)/(c[3]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)) + (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/l2**2)*(l1 - x[3] - l3*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/l2**2)/(c[3]*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)**(3/2)) + (1 + l3**2/l2**2)/(c[2]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)) + (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/l2**2)*(l1 - x[3] - l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/l2**2)/(c[2]*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)**(3/2)), l3/(c[3]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)) - (-l1 + x[3] + l3*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/l2**2)*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/(c[3]*l2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)**(3/2)), 0, 0], [0, 0, 0, l3/(c[3]*l2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)) + (l1 - x[3] - l3*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/l2**2)*(-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])/(c[3]*l2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)**(3/2)), -(l0 - x[4])**2/(c[4]*((l0 - x[4])**2 + (l1 - x[5])**2)**(3/2)) + 1/(c[4]*sqrt((l0 - x[4])**2 + (l1 - x[5])**2)) + 1/(c[3]*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)) - (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/(c[3]*l2**2*((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[4] + l3*x[3])**2/l2**2)**(3/2)), -(l0 - x[4])*(l1 - x[5])/(c[4]*((l0 - x[4])**2 + (l1 - x[5])**2)**(3/2)), 0], [0, 0, 0, 0, -(l0 - x[4])*(l1 - x[5])/(c[4]*((l0 - x[4])**2 + (l1 - x[5])**2)**(3/2)), 1/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) + (-x[5] + x[6])*(x[5] - x[6])/(c[5]*(l0**2 + (x[5] - x[6])**2)**(3/2)) - (l1 - x[5])**2/(c[4]*((l0 - x[4])**2 + (l1 - x[5])**2)**(3/2)) + 1/(c[4]*sqrt((l0 - x[4])**2 + (l1 - x[5])**2)), -1/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) + (x[5] - x[6])**2/(c[5]*(l0**2 + (x[5] - x[6])**2)**(3/2))], [0, 0, 0, 0, 0, -1/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) - (-x[5] + x[6])*(x[5] - x[6])/(c[5]*(l0**2 + (x[5] - x[6])**2)**(3/2)), -(cphi*n*p - x[6])**2/(cw*((h + n*p*sphi)**2 + (cphi*n*p - x[6])**2)**(3/2)) + 1/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[6])**2)) + 1/(c[5]*sqrt(l0**2 + (x[5] - x[6])**2)) - (x[5] - x[6])**2/(c[5]*(l0**2 + (x[5] - x[6])**2)**(3/2))]])

        # bnds = ((m*p*cphi,m*p*cphi + tan(arcsin(cw/c[0]))*(h + m*p*sphi)),(m*p*cphi, l1),(l1-l2,l1),(0.,l0),(n*p*cphi,l1),(n*p*cphi,l1))

        bnds = ((m*p*cphi,l1-l2),(m*p*cphi, l1),(0.,l0),(l1-l2,l1),(0.,l0),(n*p*cphi,l1),(n*p*cphi,l1-l2))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]),0.75*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]),(0.5*(bnds[6][1]-bnds[6][0])))

        res = minimize(f,xi,method='Newton-CG',jac=J,hess=H,options={'disp': False, 'xtol': 1e-03, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        if (res.success)&(CheckBounds([res.x[2]],[bnds[2]])):


            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])

            nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])

            th = ((180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr))), (180/pi)*sign(vrc[0]-nrc[0])*arccos(vdot(vrc,nrc)/(norm(vrc)*norm(nrc))))


        else:

            T = nan
            th = nan

        return T,th

    def MultiSkipCorner(m,n,c):

        f = lambda x: sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)/cw + sqrt(l0**2 + (x[4] - x[5])**2)/c[4] + sqrt((l0 - x[3])**2 + (l1 - x[4])**2)/c[3] + sqrt(x[3]**2 + (l1 - x[2])**2)/c[2] + sqrt(l0**2 + (x[1] - x[2])**2)/c[1] + sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([-(cphi*m*p - x[0])/(cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2)) - (x[0] - x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)), -(l1 - x[2])/(c[2]*sqrt(x[3]**2 + (l1 - x[2])**2)) - (x[1] - x[2])/(c[1]*sqrt(l0**2 + (x[1] - x[2])**2)), -l0/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)) + x[3]/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)) + x[3]/(c[2]*sqrt(x[3]**2 + (l1 - x[2])**2)), (x[4] - x[5])/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2)) - (l1 - x[4])/(c[3]*sqrt((l0 - x[3])**2 + (l1 - x[4])**2)), -(cphi*n*p - x[5])/(cw*sqrt((h + n*p*sphi)**2 + (cphi*n*p - x[5])**2)) - (x[4] - x[5])/(c[4]*sqrt(l0**2 + (x[4] - x[5])**2))])



        bnds = ((m*p*cphi,l1-l2),(m*p*cphi,l1-l2),(m*p*cphi,l1-l2),(0.,l0),(n*p*cphi,l1),(n*p*cphi,l1-l2))

        xi = (0.25*(bnds[0][1]-bnds[0][0]),0.5*(bnds[1][1]-bnds[1][0]),0.75*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]),0.5*(bnds[4][1]-bnds[4][0]),0.25*(bnds[5][1]-bnds[5][0]))

        # bnds = ((m*p*cphi,m*p*cphi + tan(arcsin(cw/c[0]))*(h + m*p*sphi)),(m*p*cphi,l1),(0.,l0),(n*p*cphi, l1-l2))
        #
        # xi = (0.25*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]))


        # res = minimize(f,xi,method='Newton-CG',jac=J,hess=H,options={'disp': False, 'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        # print(bnds)


        res = minimize(f,xi,method='BFGS',jac=J)

        # res = minimize(f,xi,method='BFGS')

        #
        # print(res.x)
        print(res.message)


        if (res.success)&(CheckBounds(res.x,bnds)):

            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])


            nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])


            th = ((180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr))), (180/pi)*sign(vrc[0]-nrc[0])*arccos(vdot(vrc,nrc)/(norm(vrc)*norm(nrc))))


            # th = ( arccos(dot(vtr.transpose(),ntr)[0]/(sqrt(dot(ntr,ntr.transpose())[0])*sqrt(dot(vtr,vtr.transpose())[0]))), arccos(dot(vrc,nrc.transpose())[0]/(sqrt(dot(nrc,nrc.transpose())[0])*sqrt(dot(vrc,vrc.transpose())[0]))))

        else:

            T = nan
            th = nan

        return T,th





    DelayFunctions = {}

    DelayFunctions['Corner'] = lambda m,n,c: Corner(m,n,c)
    DelayFunctions['SideWallCap'] = lambda m,n,c: SideWallCap(m,n,c)
    DelayFunctions['MultiSkipCorner'] = lambda m,n,c: MultiSkipCorner(m,n,c)


    c = [cs[cc] for cc in Path[1]]

    d = [[ DelayFunctions[Path[0]](m,n,c) for n in Elements[1] ] for m in Elements[0] ]

    return d



class FMC:

    def __init__(self,ascans,fsamp=25.,defaultwp={'Thickness':30., 'VerticalFL': 8.0, 'HorizontalFL': 8.0, 'SideWallPosition': 34.16}, weldtype='L',probeid='5L32',wedgeid='L10'):


        self.PathList = []

        self.PathList.append([('Cap',('T','T','T')),('Cap',('L','T','T'))])
        # self.PathList.append([('Cap',('T','T','T')),('Cap',('L','L','L'))])
        # self.PathList.append([('Cap',('L','L','T')),('Cap',('L','L','L'))])



        self.PathList.append([('Cap',('T','L','T')),('Cap',('T','L','L'))])
        # self.PathList.append([('Cap',('L','L','T')),('Cap',('L','L','L'))])



        # self.PathList.append()

        Wedge = {}
        Wedge['L10'] = {'Height':15.12,'Angle':10.,'Velocity':2.32,'Impedance':{'Longitudinal':2.32*1.04,'Transverse':2.32*1.04/2}}
        Wedge['Black'] = {'Height':6.4,'Angle':31.6,'Velocity':2.32,'Impedance':{'Longitudinal':2.32*1.04,'Transverse':2.32*1.04/2}}

        Probe = {}
        Probe['5L32'] = {'Pitch':0.6,'NumberOfElements':32,'CentreFrequency':5.0}

        self.PieceParameters = {'Velocity':{'Longitudinal':5.92,'Transverse':3.24}, 'Impedance':{'Longitudinal':5.92*7.8,'Transverse':3.24*7.8}}

        self.WeldType = weldtype

        self.WedgeParameters = Wedge[wedgeid]
        self.ProbeParameters = Probe[probeid]

        self.SamplingFrequency = fsamp

        self.LoadAScans(ascans,defaultwp)



        #
        # DelayFunction = {}
        #
        # DelayFunction['L'] = lambda pth: LWeldDelays(pth,self.WeldParameters,self.ProbeParameters,self.WedgeParameters)
        #
        # self.DelayFunction = DelayFunction[weldtype]
        #
        # self.FusionLine = []

    def LoadAScans(self,inpt,defaultwp):

        # from Civa import LoadAScansFromTxt
        #
        # loader = {}

        # loader['civa'] = lambda x: LoadAScansFromTxt(x).astype(float)
        #
        # loader['array'] = lambda x: (x.copy()).astype(float)

        # loader['list'] = lambda x: [loader['array'](xx) for xx in x]

        # loader['esndt'] = lambda x:


        self.AScans = [(i.copy()).astype(float) for i in inpt]



        self.WeldParameters = [defaultwp.copy() for i in range(len(self.AScans))]



        # if ApplyHilbert:
        #
        #     self.AScans = hilbert(A,axis=2)
        #
        # else:
        #
        #     self.AScans = A

        # self.GetSpectrum()
        #
        self.Time = linspace(0,self.AScans[0].shape[2]/self.SamplingFrequency,self.AScans[0].shape[2])

        self.Tend = self.Time[-1]


    def Calibrate(self,RefParams=(10,0.1),BPParams=(1.0,8.5,2.),Offset=0.75,BWRange = (22.,37.),ProbeDelays=None):

        from scipy.signal import detrend,tukey,firwin,fftconvolve,hilbert,correlate

        # AScans are windowed (to remove main bang + inter-element talk), band-passed, back-wall estimated, and reference taken from back-wall.
        # RefParams - tuple (Number of Cycles, Tukey Window Alpha)
        # BPParms - tuple (Low Frequency, High Frequency, )
        # Offset - float amount to remove from beginning of scans (fraction of minimum piece Thickness)

        dt = 1/self.SamplingFrequency
        phi = (pi/180)*self.WedgeParameters['Angle']
        L = self.AScans[0].shape[-1]
        N = self.ProbeParameters['NumberOfElements']

        Nwin = int(RefParams[0]/(dt*self.ProbeParameters['CentreFrequency']))



        for iScan in range(len(self.AScans)):


            for m in range(N):

                T = 2*(self.WedgeParameters['Height'] + m*self.ProbeParameters['Pitch']*sin(phi))/self.WedgeParameters['Velocity'] + 2*(BWRange[0]*Offset/self.PieceParameters['Velocity']['Longitudinal'])

                self.AScans[iScan][m,:,0:int(T/dt)] = 0.
                self.AScans[iScan][m,:,int(T/dt)::] = detrend(self.AScans[iScan][m,:,int(T/dt)::],bp=[0,int((L-(T/dt))/3),L-(int(T/dt))])


                if ProbeDelays is not None:

                    for n in range(N):

                        self.AScans[iScan][m,n,:] = ShiftSignal(self.AScans[iScan][m,n,:],-ProbeDelays[m,n],self.SamplingFrequency)


        self.GetSpectrum()


        Tw = 2*(self.WedgeParameters['Height']+N*sin(phi)*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']
        Tbw = ((Tw + 2*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'],Tw + 2*BWRange[1]/self.PieceParameters['Velocity']['Longitudinal']),(Tw + 4*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'],Tw + 4*BWRange[1]/self.PieceParameters['Velocity']['Longitudinal']))

        # Tbw = ((Tw + 6*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'],Tw + 6*BWRange[1]/self.PieceParameters['Velocity']['Longitudinal']),(Tw + 8*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'],Tw + 8*BWRange[1]/self.PieceParameters['Velocity']['Longitudinal']))




        indbw = ((int(round(Tbw[0][0]/dt)),int(round(Tbw[0][1]/dt))),(int(round(Tbw[1][0]/dt)),int(round(Tbw[1][1]/dt))))
        # indbwmax = (argmax(P[indbw[0][0]:indbw[0][1]+1])+indbw[0][0], argmax(P[indbw[1][0]:indbw[1][1]+1])+indbw[1][0])

        # print(indbw)

        self.Reference = []

        for iScan in range(len(self.AScans)):


            p = self.PlaneWaveFocus(iScan,(-self.WedgeParameters['Angle'],-self.WedgeParameters['Angle']))
            # P = abs(p)



            # indbwmax = (argmax(P[indbw[0][0]:indbw[0][1]+1])+indbw[0][0], argmax(P[indbw[1][0]:indbw[1][1]+1])+indbw[1][0])


            # h = EstimateReference(real(p[int(round((Tw+5*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'])*self.SamplingFrequency))::]),Nwin,RefParams[1])


            h = EstimateReference(real(p[int(round((Tw+1*BWRange[0]/self.PieceParameters['Velocity']['Longitudinal'])*self.SamplingFrequency))::]),Nwin,RefParams[1])


            h = Standardize(h*tukey(len(h),RefParams[1]))


            for m in range(N):
                for n in range(N):

                    self.AScans[iScan][m,n,:] = correlate(Standardize(self.AScans[iScan][m,n,:]),h,mode='same')



            P = abs(self.PlaneWaveFocus(iScan,(-self.WedgeParameters['Angle'],-self.WedgeParameters['Angle'])))



            indbwmax = (argmax(P[indbw[0][0]:indbw[0][1]+1])+indbw[0][0], argmax(P[indbw[1][0]:indbw[1][1]+1])+indbw[1][0])




            self.WeldParameters[iScan]['Thickness'] = self.PieceParameters['Velocity']['Longitudinal']*dt*(indbwmax[1]-indbwmax[0])/2

            self.EstimateSideWall(iScan)


            self.AScans[iScan] = hilbert(self.AScans[iScan],axis=2)

            self.Reference.append(h)

    def ExportAScans(self,fldr):


        if not(os.path.isdir(fldr[0:-1])):
            os.mkdir(fldr)

        for i in range(len(self.AScans)):

            (real(self.AScans[i]).astype(int16)).tofile(open(fldr+str(i)+'.txt','wb'))


    def GetSpectrum(self):

        self.TemporalSpectrum = [rfft(real(a),axis=2) for a in self.AScans]

        self.Frequency = linspace(0.,self.SamplingFrequency/2,self.TemporalSpectrum[0].shape[2])


        # self.SpatialSpectrum = fftshift(fftn(self.TemporalSpectrum,axes=(0,1)))
        # self.SpatialSpectrum = self.SpatialSpectrum/(prod(array(self.SpatialSpectrum.shape)))
        #
        # self.SpatialFrequency = linspace(-1/(2*self.ProbeParameters['Pitch']),1/(2*self.ProbeParameters['Pitch']),self.SpatialSpectrum.shape[0])



    def PlaneWaveFocus(self,ScanIndex,angles):

        from numpy import sum as asum

        d = self.ProbeParameters['Pitch']*self.ProbeParameters['NumberOfElements']

        d = linspace(-d/2,d/2,self.ProbeParameters['NumberOfElements'])


        T = meshgrid(self.Frequency,d*sin(pi*angles[1]/180)/self.WedgeParameters['Velocity'])

        X = asum(self.TemporalSpectrum[ScanIndex]*exp(-2j*pi*T[0]*T[1]),axis=1,keepdims=False)

        T = meshgrid(self.Frequency,d*sin(pi*angles[0]/180)/self.WedgeParameters['Velocity'])

        X = asum(X*exp(-2j*pi*T[0]*T[1]),axis=0,keepdims=False)

        x = ifft(X,n=2*(len(X)-1))

        return x

    def PlaneWaveSweep(self,ScanIndex,TRangles, RCangles,Gate=None):

        x = array([[ self.PlaneWaveFocus(ScanIndex,(tr,rc)) for rc in RCangles] for tr in TRangles])


        if Gate is None:

            return x

        elif (type(Gate[0]) is float)&(type(Gate[1]) is float):

            return x[:,:,int(floor(Gate[0]*self.SamplingFrequency)):int(ceil(Gate[1]*self.SamplingFrequency))]

        else:

            return x[:,:,Gate[0]:Gate[1]]


    def Sweep(self,TRangles, RCangles,BWLims=(2,5),N=10,M=5,PW=8):

        from Signal import NLargestPeaks

        import itertools
        # from scipy.signal import argrelmax

        # print(self.WeldParameters['Thickness'])
        # print(self.PieceParameters['Velocity']['Longitudinal'])
        # print(self.WedgeParameters['Height'])
        # print(self.WedgeParameters['Angle'])
        # print(self.ProbeParameters['NumberOfElements'])
        # print(self.WedgeParameters['Velocity'])

        # thsw = array([[wp['Thickness'], wp['SideWallPosition']] for wp in self.WeldParameters])
        #
        # th = mean(thsw[:,0])
        #
        # sw = mean(thsw[:,1])
        #
        # wp = self.WeldParameters[0].copy()
        #
        #
        #
        # wp['Thickness'] = th
        # wp['SideWallPosition'] = sw
        #
        #
        # Tg0 = DirectDelays((('FrontWall',('L','L','L')), ('FrontWall',('L','L','L'))), (range(1),range(1)), sw,th,wp)
        #
        # Tg0 = ToCentreAperatureDelay(Tg0[0][0][0], (0.,0.), Tg0[0][0][1], self.WedgeParameters['Velocity'], self.ProbeParameters['NumberOfElements'], self.ProbeParameters['Pitch'])
        #
        # Tg1 = DirectDelays((('FrontWall',('L','L','L')), ('Direct', ('L'))), (range(1),range(1)), sw, th, wp)
        #
        # Tg1 = ToCentreAperatureDelay(Tg1[0][0][0], (0.,0.), Tg1[0][0][1], self.WedgeParameters['Velocity'], self.ProbeParameters['NumberOfElements'], self.ProbeParameters['Pitch'])
        #
        # Tg2 = DirectDelays((('FrontWall',('L','T','L')), ('FrontWall',('L','T','L'))), (range(1),range(1)), sw, th, wp)
        #
        # Tg2 = ToCentreAperatureDelay(Tg2[0][0][0], (0.,0.), Tg2[0][0][1], self.WedgeParameters['Velocity'], self.ProbeParameters['NumberOfElements'], self.ProbeParameters['Pitch'])
        #
        #
        # g1 = (int(round(self.SamplingFrequency*Tg0))+PW, int(round(self.SamplingFrequency*Tg1)) - PW)
        #
        # g2 = (int(round(self.SamplingFrequency*Tg1))+PW, int(round(self.SamplingFrequency*Tg2)) - PW)
        #
        #
        # print(g1)
        # print(g2)

        #


        # print(self.WeldParameters[0]['Thickness']*6./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity'])
        #
        # Gate = (self.WeldParameters['Thickness']*6./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']/2)/self.WedgeParameters['Velocity']-L, self.WeldParameters['Thickness']*8./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']/2)/self.WedgeParameters['Velocity']-L)

        # Gate = [[int(round(self.SamplingFrequency*(self.WeldParameters[i]['Thickness']*6./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']-L))), int(round(self.SamplingFrequency*(self.WeldParameters[i]['Thickness']*8./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']-2*L)))] for i in range(len(self.AScans))]




        Gate = [[int(round(self.SamplingFrequency*(self.WeldParameters[i]['Thickness']*2*BWLims[0]/self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']/self.WedgeParameters['Velocity'])))), int(round(self.SamplingFrequency*(self.WeldParameters[i]['Thickness']*2*BWLims[1]/self.PieceParameters['Velocity']['Longitudinal'] + 2*self.WedgeParameters['Height']/self.WedgeParameters['Velocity'])))] for i in range(len(self.AScans))]



        #
        #
        Gate = array(Gate)
        #
        # Gate = (min(Gate[:,0]),1624)
        Gate = (min(Gate[:,0]),max(Gate[:,1]))

        if Gate[1]>self.AScans[0].shape[2]:

            Gate[1] = self.AScans[0].shape[2]

        # Gate = (min(Gate[:,0]))

        # Gate[0,:] = min(Gate[0,:])
        #
        #
        #
        # dGate = Gate[:,1]-Gate[:,0]
        #
        # Gate[:,1] = Gate[:,0]+max(dGate)
        #
        # Gate = Gate.astype(int)

        # print(Gate)


        # print(type(Gate))


        # p = [self.PlaneWaveSweep(i,TRangles,RCangles, (self.WeldParameters[i]['Thickness']*6./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']-L, self.WeldParameters[i]['Thickness']*8./self.PieceParameters['Velocity']['Longitudinal'] + 2*(self.WedgeParameters['Height']+sin(self.WedgeParameters['Angle']*(pi/180.))*self.ProbeParameters['NumberOfElements']*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']-2*L)) for i in range(len(self.AScans))]

        # p = [[self.PlaneWaveSweep(i,TRangles,RCangles,Gate1), self.PlaneWaveSweep(i,TRangles,RCangles,Gate2)] for i in range(len(self.AScans))]

        # p = array([self.PlaneWaveSweep(i,TRangles,RCangles,Gate) for i in range(len(self.AScans))])

        amp = [norm(self.PlaneWaveFocus(i,(self.WedgeParameters['Angle'],self.WedgeParameters['Angle']))) for i in range(len(self.AScans))]



        p = [self.PlaneWaveSweep(i,TRangles,RCangles,Gate)/amp[i] for i in range(len(self.AScans))]



        #
        # p0 = sum(sum(abs(p[:,:,:,g1[0]:g1[1]]),axis=1),axis=1).transpose()
        #
        # p1 = sum(sum(abs(p[:,:,:,g2[0]:g2[1]]),axis=1),axis=1).transpose()



        # for pp in p:
        #
        #     print(pp.shape)
        # print(array(p).shape)

        pmaxstack = array([amax(amax(abs(pp),axis=0),axis=0) for pp in p]).transpose()

        psumstack = array([sum(sum(abs(pp),axis=0),axis=0) for pp in p]).transpose()


        # print(array(pstack).shape)

        # pstack = [sum(sum(abs(pp),axis=0),axis=0) for pp in p]

        # pstack = [[sum(sum(abs(p[i][0]),axis=0),axis=0), sum(sum(abs(p[i][1]),axis=0),axis=0)] for i in range(len(p))]


        # self.SweepStack = (p0,p1)

        # print(pstack[0].shape)

        # indpks = [ NLargestPeaks(p,N,M)[2::] for p in pstack ]

        # indpks = [[[ NLargestPeaks(abs(pp[i,j,:]),N,M)[2::] for j in range(len(RCangles))] for i in range(len(TRangles))] for pp in p]

        indpks = [[[ NLargestPeaks(abs(pp[i,j,:]),N,M)[2::] for j in range(len(RCangles))] for i in range(len(TRangles))] for pp in p]




        # s = array([sum(pstack[i][indpks[i]]) for i in range(len(pstack))])

        # s = array([sum(pstack[i][indpks[i]]) for i in range(len(pstack))])

        # s = array([[[sum(abs(p[k][i,j,indpks[k][i][j]])) for j in range (len(RCangles))] for i in range(len(TRangles))] for k in range(len(p))]])

        s = array([reduce(lambda x,y: x+y, [sum(abs(p[k][i,j,indpks[k][i][j]])) for j in range(len(RCangles)) for i in range(len(TRangles))]) for k in range(len(p))])
        #
        # print(indpks[0].shape)
        #
        # s = array([sum(pstack[i][indpks[i][2::]]) for i in range(len(pstack))])


        # print(array(pstack).shape)
        # ss = sum(sum(sum(abs(array(pstack)),axis=1),axis=1),axis=0)
        # imshow(pmaxstack,aspect=0.01)
        # show()

        def ToSTDAboveBackground(x):

            th = threshold_li(x)

            mu = mean(x[x<th])
            sig = std(x[x<th])

            return (x-mu)/sig

        sshape = pmaxstack.shape

        Smaxn = ToSTDAboveBackground(pmaxstack.flatten())

        Smaxn = Smaxn.reshape(sshape)

        Ssumn = ToSTDAboveBackground(psumstack.flatten())

        Ssumn = Ssumn.reshape(sshape)


        self.SweepMaxStack = Smaxn*(Smaxn>0.)

        # imshow(self.SweepMaxStack,aspect=0.01)
        # show()

        self.SweepSumStack = Ssumn*(Ssumn>0.)

        self.SweepPeakSum = s

        self.SweepMaxStackSum = ToSTDAboveBackground(sum(pmaxstack,axis=0))

        self.SweepSumStackSum = ToSTDAboveBackground(sum(psumstack,axis=0))

        #




        # indpks =

        # pref = mean(pstack,axis=1)
        #
        # p = moveaxis(dstack(p),2,3)
        #
        # P = zeros(p.shape,dtype=complex)
        #
        # for i in range(len(self.AScans)):
        #
        #     ind = argmax(abs(correlate(pref,pstack[i],'same')))
        #
        #     P[:,:,i,:] = ShiftSignal(p[:,:,i,:],int(round(ind/self.SamplingFrequency)),self.SamplingFrequency)




    def EstimateSideWall(self,ScanIndex,SWRange = (34.,45.),dl=0.3):

        # Lref = len(self.Reference)

        wp = self.WeldParameters[ScanIndex].copy()

        elements = ((range(15,16),range(15,16)))



        def SideWallError(x):

            wp['SideWallPosition'] = x

            # T = [ Delays(('Corner',p),elements,wp) for p in [('L','L','L'),('L','L','T')]]

            T = [ Delays(('Corner',p),elements,wp) for p in [('L','L','L')]]


            R = reduce(lambda x,y: x+y,  [abs(self.PlaneWaveFocus(ScanIndex,TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))]) if isfinite(TT[m][n][0]) else 0. for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T])


            return -R

        swgrid = linspace(SWRange[0],SWRange[1],int((SWRange[1]-SWRange[0])/dl))

        SWErrGrid = array([SideWallError(sw) for sw in swgrid])
        #
        # plot(swgrid,SWErrGrid)
        #
        # show()


        self.WeldParameters[ScanIndex]['SideWallPosition'] = swgrid[argmin(SWErrGrid)]


    # def EstimateCap(self,CapRanges=((4.7,8.3),(2.7,8.3)),dl=0.3):
    #
    #     import itertools
    #     from scipy.optimize import basinhopping
    #
    #     wp = self.WeldParameters.copy()
    #
    #     Lref = len(self.Reference)
    #
    #
    #     def CapError(x):
    #
    #         wp['VerticalFL'] = x[0]
    #         wp['HorizontalFL'] = x[1]
    #
    #         # elements = ((range(15,16)),(range(15,16)))
    #
    #         elements = range(0,31)
    #
    #
    #         # modes = itertools.product(['L','T'],repeat = 3)
    #
    #
    #
    #         # modes = [('L', 'T', 'L', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L'),('L', 'L', 'T', 'L', 'T', 'T'),('T', 'L', 'L', 'L', 'L', 'T'),('L', 'T', 'T', 'L', 'T', 'L')]
    #
    #         # modes = [('L', 'T', 'L', 'T', 'T', 'L'),('T', 'T', 'L', 'L', 'L', 'L'),('L', 'T', 'L', 'T', 'L', 'L'),('L', 'T', 'T', 'T', 'T', 'T'),('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'L', 'L', 'T', 'L')]
    #
    #         # modes = [('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L')]
    #
    #         # modes = [('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L')]
    #
    #
    #         # modes = [('L', 'T', 'T', 'L', 'T', 'L')]
    #         #
    #         # modes = [('L','L','L'),('L','L','T'),('T')]
    #
    #         # modes = [('T', 'T', 'T', 'T', 'T', 'L')]
    #
    #         # modes = [('L','L','L'),('L','L','T')]
    #
    #         modes = [('L','L','L')]
    #
    #         # modes = [('L', 'T', 'L', 'T', 'T', 'L'),('T', 'T', 'L', 'L', 'L', 'L'),('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L')]
    #
    #         # T = [ Delays(('SideWallCap',p),elements,wp) for p in modes ]
    #
    #         # T = [ Delays(('SymmetricSideWallCap',p),elements,wp) for p in modes ]
    #
    #         # T = [ Delays(('SymmetricSideWallCap',p),elements,wp) for p in modes ]
    #
    #         T = [SymmetricDelays(('SideWallCap',p),elements,wp) for p in modes ]
    #
    #         if any(isfinite(array(T[0][0:-1][0]))):
    #
    #             print(x[0])
    #             print(x[1])
    #
    #         # R = [ InnerProduct(self.Reference, self.PlaneWaveFocus(TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2)):int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2))+Lref],ipopt='Normalized') for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T if isfinite(TT[m][n][0])]
    #
    #         # R = [ self.PlaneWaveFocus(TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))] for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T if isfinite(TT[m][n][0])]
    #
    #
    #         R = [InnerProduct(self.Reference,self.AScans[elements[m],elements[m],int(round(TT[m][0]*self.SamplingFrequency))-int(round(Lref/2)):int(round(TT[m][0]*self.SamplingFrequency))-int(round(Lref/2))+Lref]) for m in range(len(elements)) for TT in T if isfinite(TT[m][0])]
    #
    #         # R = [ self.AScans[elements[m],elements[m],int(round(TT[m][0]*self.SamplingFrequency))] for m in range(len(elements)) for TT in T if isfinite(TT[m][0])]
    #
    #         # R = abs(array(R))
    #
    #         R = array(R)
    #
    #         # print(R.shape)
    #         # print(conj(R).shape)
    #
    #
    #         if len(R)>0:
    #
    #             R = -real(dot(R,conj(R)))/len(R)
    #
    #             # R = -dot(R,conj(R))
    #
    #         else:
    #
    #             R = 0.
    #
    #
    #
    #         return R
    #
    #
    #     vgrid = linspace(CapRanges[0][0],CapRanges[0][1],int((CapRanges[0][1]-CapRanges[0][0])/dl))
    #     #
    #     # print(vgrid)
    #     #
    #     hgrid = linspace(CapRanges[1][0],CapRanges[1][1],int((CapRanges[1][1]-CapRanges[1][0])/dl))
    #
    #     # res = basinhopping(CapError,(7.,4.))
    #
    #     vhgrid = [(v,h) for v in vgrid for h in hgrid if h<=v]
    #
    #     CapErrGrid = array([CapError(vh) for vh in vhgrid])
    #
    #     plot(CapErrGrid)
    #
    #     show()
    #
    #     # print(res.x)
    #
    #     print(vhgrid[argmin(CapErrGrid)])
    #
    #
    #     # self.WeldParameters['VerticalFL'] = vhgrid[argmin(CapErrGrid)][0]
    #     #
    #     # self.WeldParameters['HorizontalFL'] = vhgrid[argmin(CapErrGrid)][1]
    #
    #
    # def GetDelays(self,path,wp=None):
    #
    #     if wp is not None:
    #
    #         for w in wp:
    #
    #             self.WeldParameters[w[0]] = w[1]
    #
    #
    #     return self.DelayFunction(path)



    def DirectDelayImage(self,pth,Y,X='sidewall',elements=(range(10),range(10)),vflrng=(6,8,5),hflrng=(4,6,5)):


        Img = []

        vrng = linspace(vflrng[0],vflrng[1],vflrng[2])

        hrng = linspace(hflrng[0],hflrng[1],hflrng[2])


        def GetImage(ScanIndex,vfl,hfl):

            wp['VerticalFL'] = vfl
            wp['HorizontalFL'] = hfl

            T = [[ DirectDelays(pth,elements,x,y,wp) for x in X] for y in Y]

            I = zeros((len(Y),len(X)),dtype='complex')

            for iy in range(len(Y)):

                for ix in range(len(X)):

                    I[iy,ix] = reduce(lambda x,y:x+y, [self.AScans[ScanIndex][m,n,int(round(T[iy][ix][m][n]*self.SamplingFrequency))] if isfinite(T[iy][ix][m][n]) else 0+0j for m in elements[0] for n in elements[1]])

            return I

        for iScan in range(len(self.AScans)):

            wp = self.WeldParameters[iScan].copy()

            if X is 'sidewall':

                X = array([wp['SideWallPosition']])

            # I = zeros((len(Y),len(X)))

            I = array([GetImage(iScan,v,h) for v in vrng for h in hrng if h<=v])

            # for v in vrng:
            #
            #     h = [h for h in hrng if h<=v]
            #
            #     for hh in h:
            #
            #         I += abs(GetImage(iScan,v,hh))

            Img.append(I)


            # Img.append(reduce(StackingFunctions[stackfnct], [abs(GetImage(iScan,v,h)) for v in vrng for h in hrng if h<v]))

            # I = zeros((len(Y),len(X)))
            #
            # for v in vflrng:
            #
            #     h = [h for h in hflrng if h<v]
            #
            #         for hh in h:
            #
            #         I += abs(GetImage(v,hh))
            #
            # Img.append(I)


        # Img = array([I.ravel() for I in Img])
        #
        # Img = Standardize(Img,ax=0).transpose()
        #
        # Img = Img*(Img>0.)

        return Img

    def FusionLineFocus(self,y=linspace(0,-6.,10)):

        self.FusionLineImages = [self.DirectDelayImage(pths,Y=y) for pths in self.PathList]









    # def SymmetricModeImage(self,pathkey,CapRanges=((3.0,7.0),(3.0,7.0)),dl=0.5):
    #
    #     import itertools
    #
    #     wp = self.WeldParameters.copy()
    #
    #     elements = range(0,31)
    #
    #
    #     def Image(vfl,hfl,modes):
    #
    #         wp['VerticalFL'] = vfl
    #         wp['HorizontalFL'] = hfl
    #         # wp['SideWallPosition'] = x[2]
    #
    #
    #         T = SymmetricDelays((pathkey,modes),elements,wp)
    #
    #         I = [[T[m]['y'], self.AScans[elements[m],elements[m],int(round(T[m]['Delay']*self.SamplingFrequency))]] for m in range(len(elements)) if len(T[m])>0]
    #
    #
    #         return I
    #
    #
    #
    #     vgrid = linspace(CapRanges[0][0],CapRanges[0][1],int((CapRanges[0][1]-CapRanges[0][0])/dl))
    #
    #     hgrid = linspace(CapRanges[1][0],CapRanges[1][1],int((CapRanges[1][1]-CapRanges[1][0])/dl))
    #
    #     # vhgrid = [(v,h) for v in vgrid for h in hgrid if h<=v]
    #
    #     l = []
    #
    #
    #     for m in itertools.product(['L','T'],repeat = 3):
    #
    #         for v in vgrid:
    #
    #             H = [h for h in hgrid if h<v]
    #
    #             for h in H:
    #
    #                 I = Image(v,h,m)
    #
    #                 if len(I)>0:
    #
    #                     I = array(I)
    #
    #                     l.append({'Modes':m,'y':I[:,0],'Intensity':I[:,1],'VFL':v,'HFL':h})
    #
    #     return l






    # self.WeldParameters['VerticalFL'] = vhgrid[argmin(CapErrGrid)][0]
    #
    # self.WeldParameters['HorizontalFL'] = vhgrid[argmin(CapErrGrid)][1]




    # def FusionLineFocus(self,Delays):
    #
    #     # IM = Delays
    #
    #     # n,m,y1,y2,T,C
    #
    #     L = len(self.Window)
    #
    #     GetAmplitude = lambda p: self.AScans[p[0],p[1],int(round(self.SamplingFrequency*p[2]))]
    #     # GetInnerProduct = lambda m,n,T: InnerProduct(real(self.Reference),real(self.AScans[m,n,int(round(self.SamplingFrequency*T))-int(round(L/2)):int(round(self.SamplingFrequency*T))-int(round(L/2))+L])*real(self.Window))
    #
    #     # I = [[GetAmplitude((IM[0],IM[1],IM[3]))*IM[4], sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]), IM[2]] for IM in Delays if len(IM)>0]
    #
    #
    #     I = []
    #
    #     IP = []
    #     Y = []
    #
    #     # if Delays:
    #     #
    #     #     for IM in Delays:
    #     #
    #     #         I.append(GetAmplitude((IM[0],IM[1],IM[3]))*IM[4])
    #     #
    #     #         IP.append(sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]))
    #     #
    #     #
    #     #         Y.append(IM[2])
    #     #
    #     #     # self.FusionLine.append({'Path':path,'Intensity':I,'YCoordinate':Y})
    #     #     return (array(Y),array(I),array(IP))
    #     #
    #     # else:
    #     #
    #     #     return ([],[],[])
    #
    #     try:
    #
    #         for IM in Delays:
    #
    #             I.append(GetAmplitude((IM[0],IM[1],IM[3]))*IM[4])
    #
    #             IP.append(sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]))
    #
    #
    #             Y.append(IM[2])
    #
    #         # self.FusionLine.append({'Path':path,'Intensity':I,'YCoordinate':Y})
    #         return (array(Y),array(I),array(IP))
    #
    #     except:
    #
    #         return ([],[],[])
