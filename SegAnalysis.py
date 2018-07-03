
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
from functools import reduce

from Signal import *



def CheckBounds(x,bndlist):

    return all([(x[i]>=bndlist[i][0])&(x[i]<=bndlist[i][1]) for i in range(len(x))])

TRDict={}
TRDict['WedgeToSteel'] = lambda ti,ttl,tts,cw,css,csL,rs,rw: (array([[cos(ti)/cw, cos(ttl)/csL, -sin(tts)/css], [rw*(0.5*sin(ti)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(ttl)**2/csL**2, -1.0*rs*sin(2*tts)], [0, css**2*rs*sin(2*ttl)/csL**2, rs*cos(2*tts)]]),array([[cos(ti)/cw], [-rw*(0.5*sin(ti)**2 - 1.0)], [0]]))
TRDict['SteelToWedgeCompressionIncidence'] = lambda ti,ttl,trs,trl,cw,css,csL,rs,rw: (array([[cos(ttl)/cw, cos(trl)/csL, sin(trs)/css], [rw*(0.5*sin(ttl)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(trl)**2/csL**2, 1.0*rs*sin(2*trs)], [0, -css**2*rs*sin(2*trl)/csL**2, rs*cos(2*trs)]]),array([[cos(ti)/csL], [-1.0*rs + 2.0*css**2*rs*sin(ti)**2/csL**2], [-css**2*rs*sin(2*ti)/csL**2]]))
TRDict['SteelToWedgeShearIncidence'] = lambda ti,ttl,trs,trl,cw,css,csL,rs,rw: (array([[cos(ttl)/cw, cos(trl)/csL, sin(trs)/css], [rw*(0.5*sin(ttl)**2 - 1.0), 1.0*rs - 2.0*css**2*rs*sin(trl)**2/csL**2, 1.0*rs*sin(2*trs)], [0, -css**2*rs*sin(2*trl)/csL**2, rs*cos(2*trs)]]),array([[-sin(ti)/css], [1.0*rs*sin(2*ti)], [-rs*cos(2*ti)]]))
TRDict['SteelToVacuumShearIncidence'] = lambda ti,trs,trl,css,csL,rs: (array([[-rs + 2*css**2*rs*sin(trl)**2/csL**2, -rs*sin(2*trs)], [-css**2*rs*sin(2*trl)/csL**2,  rs*cos(2*trs)]]),array([[-rs*sin(2*ti)],[-rs*cos(2*ti)]]))
TRDict['SteelToVacuumCompressionIncidence'] = lambda ti,trs,trl,css,csL,rs: (array([[-rs + 2*css**2*rs*sin(trl)**2/csL**2, -rs*sin(2*trs)],[-css**2*rs*sin(2*trl)/csL**2,  rs*cos(2*trs)]]),array([[rs - 2*css**2*rs*sin(ti)**2/csL**2],[-css**2*rs*sin(2*ti)/csL**2]]))

def InnerProduct(x,y,eps=1e-5,ipopt='Normalized'):

    # Cross Correlation Function with Option of Normalizing Inner Product at Each Shift: x - is template, y - signal to match template against, t = 0 is assumed to correspond to the max of x
    # eps is a stabilizing constant to avoid low variances scaling the output in an unbounded way

    M = len(x)
    N = len(y)


    ipfnct = {}

    ipfnct['Normalized'] = lambda x,y: dot((x-mean(x))/std(x),conj((y-mean(y))/(std(y)+eps)))/M
    ipfnct['Centred'] = lambda x,y: dot(x-mean(x),conj(y-mean(y)))/M

    return real(array(ipfnct[ipopt](x,y)))


def LWeldDelays(Path,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Pitch':0.6,'Angle':10.0,'NumberOfElements':32},dy=0.15,eps=0.01):

    from scipy.optimize import minimize,root,fsolve
    from itertools import product

    jactol = 1e-5

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalLOF']
    l3 = WeldParameters['HorizontalLOF']

    Ny = int(round(l3/dy))
    Y = dy*linspace(-Ny,Ny,2*Ny)

    cs = {'L':5.9,'T':3.23}

    cphi = cos(phi)
    sphi = sin(phi)
    tphi = sphi/cphi

    def Path1Delay(x,n,m,c1,c2,c3,c4,c5):

        x0,x1,x2,x3,x4,y = x[0],x[1],x[2],x[3],x[4],x[5]

        f = sqrt((h + m*p*sphi)**2 + (-m*p*cphi + x4)**2)/cw + sqrt((h + n*p*sphi)**2 + (-n*p*cphi + x0)**2)/cw + sqrt(l0**2 + (x3 - x4)**2)/c5 + sqrt((l0 - y)**2 + (l1 - x3)**2)/c4 + sqrt((l1 - x2)**2 + (y - (l1*l3 - l2*l3 - l3*x2)/l2)**2)/c3 + sqrt((l0 - (l1*l3 - l2*l3 - l3*x2)/l2)**2 + (-x1 + x2)**2)/c2 + sqrt(l0**2 + (-x0 + x1)**2)/c1

        J = [-(n*p*cphi - x0)/(cw*sqrt((h + n*p*sphi)**2 + (n*p*cphi - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (x1 - x2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (-l1 + x2 + l3*(y + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2)) + (-x1 + x2 + l3*(l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)), (x3 - x4)/(c5*sqrt(l0**2 + (x3 - x4)**2)) - (l1 - x3)/(c4*sqrt((l0 - y)**2 + (l1 - x3)**2)), -(m*p*cphi - x4)/(cw*sqrt((h + m*p*sphi)**2 + (m*p*cphi - x4)**2)) - (x3 - x4)/(c5*sqrt(l0**2 + (x3 - x4)**2)), -(l0 - y)/(c4*sqrt((l0 - y)**2 + (l1 - x3)**2)) + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2))]

        return f,array(J)

    def Path1(n,m,c1,c2,c3,c4,c5):

        bnds = ((n*p*cphi,l1-l2),(n*p*cphi, l1),(l1-l2,l1),(m*p*cphi,l1),(m*p*cphi,l1-l2),(-l3,0))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]),0.5*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]),0.5*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][0]-bnds[5][1]))

        res = minimize(Path1Delay,xi,args=(n,m,c1,c2,c3,c4,c5),method='L-BFGS-B',jac=True,bounds=bnds)

        if (res.success)&(amax(abs(array(res.jac)))<=jactol):

            T = res.fun
            X = res.x
            y = X[5]

        else:

            T = nan
            X = nan
            y = nan

        return (T,X,y)

    def Path1CorrectionForward(n,m,X,c1,c2,c3,c4,c5):

        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        x4 = X[4]
        y = X[5]

        rs = 7.8
        rw = 1.05

        css = cs['T']
        csL = cs['L']

        ti = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        ttl = arcsin(complex((csL/cw)*sin(ti)))
        tts = arcsin(complex((css/cw)*sin(ti)))

        M = TRDict['WedgeToSteel'](ti,ttl,tts,cw,css,csL,rs,rw)
        RL,TL,TS = solve(M[0],M[1])

        if c1 == csL:

            A1 = TL

            ti = arctan2((x1-x0),l0)
            trl = ti
            trs = arcsin(complex((css/csL)*sin(ti)))

            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((l1 - x3),(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((l1 - x3),(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((l1 - x3),(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((l1 - x3),(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((l1 - x3),(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((l1 - x3),(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

        else:

            A1 = TS

            ti = arctan((x1-x0)/l0)
            trs = ti
            trl = arcsin(complex((csL/css)*sin(ti)))

            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan((l0 - yp)/(x2 - x1)) + arctan(l3/l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan((abs(y - yp))/(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan((l1 - x3)/(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    ti = arctan((abs(y - yp))/(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan((l1 - x3)/(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan((l0 - yp)/(x2 - x1)) + arctan(l3/l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan((abs(y - yp))/(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan((l1 - x3)/(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    ti = arctan((abs(y - yp))/(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan((l1 - x3)/(l0 - y))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan((l1 - x3)/(l0 - y))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan((x3 - x4)/l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan((x3 - x4)/l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

        Path1TRCorrectionForward = (A1*A2*A3*A4*A5*A6)[0]

        Path1GeometricalCorrectionForward = sqrt(sqrt((x0 - n*p*cphi)**2 + (h + n*p*sphi)**2) + sqrt((x1 - x0)**2 + l0**2) + sqrt((x2 - x1)**2 + (l0 - yp)**2) + sqrt((l1 - x2)**2 + (y - yp)**2))*sqrt(sqrt((l1 - x3)**2 + (l0-y)**2) + sqrt((x3 - x4)**2 + (l0)**2) + sqrt((x4 - m*p*cphi)**2 + (h + m*p*sphi)**2))

        tin = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        tdirn = abs(arcsin(sphi) - tin)

        tim = arctan2((x4 - m*p*cphi),(h + m*p*sphi))
        tdirm = abs(arcsin(sphi) - tim)

        landa = cw/5
        Path1Directivity = sinc(p*sin(tdirn)/landa)*sinc(p*sin(tdirm)/landa)
        if Path1Directivity < 0:
            Path1Directivity = 0

        return sign(Path1TRCorrectionForward)*Path1GeometricalCorrectionForward/(Path1Directivity*abs(Path1TRCorrectionForward) + eps)

    def Path1CorrectionBackward(n,m,X,c1,c2,c3,c4,c5):

        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        x4 = X[4]
        y = X[5]

        rs = 7.8
        rw = 1.05

        css = cs['T']
        csL = cs['L']

        ti = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        ttl = arcsin(complex((csL/cw)*sin(ti)))
        tts = arcsin(complex((css/cw)*sin(ti)))

        M = TRDict['WedgeToSteel'](ti,ttl,tts,cw,css,csL,rs,rw)
        RL,TL,TS = solve(M[0],M[1])

        if c1 == csL:

            A1 = TL

            ti = arctan2((x1-x0),l0)
            trl = ti
            trs = arcsin(complex((css/csL)*sin(ti)))

            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                ti = arctan2(l0 - y,l1 - x1)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

            else:

                A2 = RS

                ti = arctan2(l0 - y,l1 - x1)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

        else:

            A1 = TS

            ti = arctan((x1-x0)/l0)
            trs = ti
            trl = arcsin(complex((csL/css)*sin(ti)))

            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                ti = arctan2(l0 - y,l1 - x1)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

            else:

                A2 = RS

                ti = arctan2(l0 - y,l1 - x1)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                else:

                    A3 = RS

                    yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x2) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x2),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x2))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x2)) + arctan2((yp - y),(l1 - x2))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                    else:

                        A4 = RS

                        ti = arctan2((x2 - x3),(l0 - yp))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),l0)
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))
                            ttl = arcsin(complex((cw/csL)*sin(ti)))

                            M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),l0)
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))
                            ttl = arcsin(complex((cw/css)*sin(ti)))

                            M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                            TL,RL,RS = solve(M[0],M[1])

                            A6 = TL

        Path1TRCorrectionBackward = (A1*A2*A3*A4*A5*A6)[0]

        Path1GeometricalCorrectionBackward = sqrt(sqrt((x0 - n*p*cphi)**2 + (h + n*p*sphi)**2) + sqrt((x1 - x0)**2 + l0**2) + sqrt((x2 - x1)**2 + (l0 - yp)**2) + sqrt((l1 - x2)**2 + (y - yp)**2))*sqrt(sqrt((l1 - x3)**2 + (l0-y)**2) + sqrt((x3 - x4)**2 + (l0)**2) + sqrt((x4 - m*p*cphi)**2 + (h + m*p*sphi)**2))

        tin = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        tdirn = abs(arcsin(sphi) - tin)

        tim = arctan2((x4 - m*p*cphi),(h + m*p*sphi))
        tdirm = abs(arcsin(sphi) - tim)

        landa = cw/5
        Path1Directivity = sinc(p*sin(tdirn)/landa)*sinc(p*sin(tdirm)/landa)
        if Path1Directivity < 0:
            Path1Directivity = 0

        return sign(Path1TRCorrectionBackward)*Path1GeometricalCorrectionBackward/(Path1Directivity*abs(Path1TRCorrectionBackward) + eps)

    def Path2Delay(x,n,m,c1,c2,c3,c4,c5,c6):

        x0,x1,x2,x3,x4,x5,y = x[0],x[1],x[2],x[3],x[4],x[5],x[6]

        f = sqrt((h + m*p*sphi)**2 + (-m*p*cphi + x5)**2)/cw + sqrt((h + n*p*sphi)**2 + (-n*p*cphi + x0)**2)/cw + sqrt(l0**2 + (x4 - x5)**2)/c6 + sqrt((l0 - (l1*l3 - l2*l3 - l3*x3)/l2)**2 + (x3 - x4)**2)/c5 + sqrt((l1 - x3)**2 + (y - (l1*l3 - l2*l3 - l3*x3)/l2)**2)/c4 + sqrt((l1 - x2)**2 + (y - (l1*l3 - l2*l3 - l3*x2)/l2)**2)/c3 + sqrt((l0 - (l1*l3 - l2*l3 - l3*x2)/l2)**2 + (-x1 + x2)**2)/c2 + sqrt(l0**2 + (-x0 + x1)**2)/c1

        J = [-(n*p*cphi - x0)/(cw*sqrt((h + n*p*sphi)**2 + (n*p*cphi - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (x1 - x2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (-l1 + x2 + l3*(y + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2)) + (-x1 + x2 + l3*(l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)), (x3 - x4 + l3*(l0 + (-l1*l3 + l2*l3 + l3*x3)/l2)/l2)/(c5*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x3)/l2)**2 + (x3 - x4)**2)) + (-l1 + x3 + l3*(y + (-l1*l3 + l2*l3 + l3*x3)/l2)/l2)/(c4*sqrt((l1 - x3)**2 + (y + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)), (x4 - x5)/(c6*sqrt(l0**2 + (x4 - x5)**2)) - (x3 - x4)/(c5*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x3)/l2)**2 + (x3 - x4)**2)), -(m*p*cphi - x5)/(cw*sqrt((h + m*p*sphi)**2 + (m*p*cphi - x5)**2)) - (x4 - x5)/(c6*sqrt(l0**2 + (x4 - x5)**2)), (y + (-l1*l3 + l2*l3 + l3*x3)/l2)/(c4*sqrt((l1 - x3)**2 + (y + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)) + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2))]

        return f,array(J)

    def Path2(n,m,c1,c2,c3,c4,c5,c6):

        bnds = ((n*p*cphi,l1-l2),(n*p*cphi, l1),(l1-l2,l1), (l1-l2,l1), (m*p*cphi,l1) , (m*p*cphi,l1-l2),(-l3,0))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]),0.5*(bnds[2][1]-bnds[2][0]),0.75*(bnds[3][1]-bnds[3][0]),0.5*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]), 0.5*(bnds[6][0]-bnds[6][1]))

        res = minimize(Path2Delay,xi, args=(n,m,c1,c2,c3,c4,c5,c6), jac=True, method='L-BFGS-B',bounds=bnds)

        if (res.success)&(amax(abs(array(res.jac)))<=jactol):

            T = res.fun
            X = res.x
            y = X[6]

        else:

            T = nan
            X = nan
            y = nan

        return (T,X,y)


    def Path2Correction(n,m,X,c1,c2,c3,c4,c5,c6):

        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        x4 = X[4]
        x5 = X[5]
        y = X[6]

        rs = 7.8
        rw = 1.05
        css = cs['T']
        csL = cs['L']

        ti = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        ttl = arcsin(complex((csL/cw)*sin(ti)))
        tts = arcsin(complex((css/cw)*sin(ti)))

        M = TRDict['WedgeToSteel'](ti,ttl,tts,cw,css,csL,rs,rw)
        RL,TL,TS = solve(M[0],M[1])

        if c1 == csL:

            A1 = TL

            ti = arctan2((x1-x0),l0)
            trl = ti
            trs = arcsin(complex((css/csL)*sin(ti)))

            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),(l0))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),(l0))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

        else:

            A1 = TS

            ti = arctan2((x1-x0),l0)
            trs = ti
            trl = arcsin(complex((csL/css)*sin(ti)))

            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                        else:

                            A5 = RS

                            ti = arctan2((x3 - x4),(l0 - yp))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x5 - x4),l0)
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))
                                ttl = arcsin(complex((cw/csL)*sin(ti)))

                                M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x5 - x4),l0)
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))
                                ttl = arcsin(complex((cw/css)*sin(ti)))

                                M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                TL,RL,RS = solve(M[0],M[1])

                                A7 = TL

        Path2TRCorrection = (A1*A2*A3*A4*A5*A6*A7)[0]

        yp1 = (-l3*x2 + l3*l1 - l3*l2)/(l2)
        yp2 = (-l3*x3 + l3*l1 - l3*l2)/(l2)

        Path2GeometricalCorrection = sqrt(sqrt((x0 - n*p*cphi)**2 + (h + n*p*sphi)**2) + sqrt((x1 - x0)**2 + l0**2) + sqrt((x2 - x1)**2 + (l0 - yp1)**2) + sqrt((l1 - x2)**2 + (y - yp1)**2))*sqrt(sqrt((l1 - x3)**2 + (y - yp2)**2) + sqrt((x3 - x4)**2 + (l0 - yp2)**2) + sqrt((x4 - x5)**2 + (l0)**2) + sqrt((x5 - m*p*cphi)**2 + (h + m*p*sphi)**2))

        tin = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        tdirn = abs(arcsin(sphi) - tin)

        tim = arctan2((x5 - m*p*cphi),(h + m*p*sphi))
        tdirm = abs(arcsin(sphi) - tim)

        landa = cw/5
        Path2Directivity = sinc(p*sin(tdirn)/landa)*sinc(p*sin(tdirm)/landa)
        if Path2Directivity < 0:
            Path2Directivity = 0

        Path2TRCorrection = real(Path2TRCorrection)

        return sign(Path2TRCorrection)*Path2GeometricalCorrection/(Path2Directivity*abs(Path2TRCorrection) + eps)


    def Path3Delay(x,n,m,c1,c2,c3,c4,c5,c6,c7):

        x0,x1,x2,x3,x4,x5,x6,y = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]

        f = sqrt((h + m*p*sphi)**2 + (-m*p*cphi + x6)**2)/cw + sqrt((h + n*p*sphi)**2 + (-n*p*cphi + x0)**2)/cw + sqrt(l0**2 + (-x5 + x6)**2)/c7 + sqrt((l0 - x4)**2 + (l1 - x5)**2)/c6 + sqrt((l1 - x3)**2 + (x4 - (l1*l3 - l2*l3 - l3*x3)/l2)**2)/c5 + sqrt((l1 - x3)**2 + (y - (l1*l3 - l2*l3 - l3*x3)/l2)**2)/c4 + sqrt((l1 - x2)**2 + (y - (l1*l3 - l2*l3 - l3*x2)/l2)**2)/c3 + sqrt((l0 - (l1*l3 - l2*l3 - l3*x2)/l2)**2 + (-x1 + x2)**2)/c2 + sqrt(l0**2 + (-x0 + x1)**2)/c1

        J = [-(n*p*cphi - x0)/(cw*sqrt((h + n*p*sphi)**2 + (n*p*cphi - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (x1 - x2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (-l1 + x2 + l3*(y + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2)) + (-x1 + x2 + l3*(l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)/l2)/(c2*sqrt((l0 + (-l1*l3 + l2*l3 + l3*x2)/l2)**2 + (x1 - x2)**2)), (-l1 + x3 + l3*(x4 + (-l1*l3 + l2*l3 + l3*x3)/l2)/l2)/(c5*sqrt((l1 - x3)**2 + (x4 + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)) + (-l1 + x3 + l3*(y + (-l1*l3 + l2*l3 + l3*x3)/l2)/l2)/(c4*sqrt((l1 - x3)**2 + (y + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)), -(l0 - x4)/(c6*sqrt((l0 - x4)**2 + (l1 - x5)**2)) + (x4 + (-l1*l3 + l2*l3 + l3*x3)/l2)/(c5*sqrt((l1 - x3)**2 + (x4 + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)), (x5 - x6)/(c7*sqrt(l0**2 + (x5 - x6)**2)) - (l1 - x5)/(c6*sqrt((l0 - x4)**2 + (l1 - x5)**2)), -(m*p*cphi - x6)/(cw*sqrt((h + m*p*sphi)**2 + (m*p*cphi - x6)**2)) - (x5 - x6)/(c7*sqrt(l0**2 + (x5 - x6)**2)), (y + (-l1*l3 + l2*l3 + l3*x3)/l2)/(c4*sqrt((l1 - x3)**2 + (y + (-l1*l3 + l2*l3 + l3*x3)/l2)**2)) + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)/(c3*sqrt((l1 - x2)**2 + (y + (-l1*l3 + l2*l3 + l3*x2)/l2)**2))]

        return f,array(J)

    def Path3(n,m,c1,c2,c3,c4,c5,c6,c7):

        bnds = ((n*p*cphi,l1-l2),(n*p*cphi, l1),(l1-l2,l1),(l1-l2,l1),(-l3,l0),(m*p*cphi, l1),(m*p*cphi,l1-l2),(-l3,0))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]),0.5*(bnds[2][1]-bnds[2][0]),0.75*(bnds[3][1]-bnds[3][0]),0.5*(bnds[4][0]+bnds[4][1]),0.5*(bnds[5][1]-bnds[5][0]), 0.5*(bnds[6][1]-bnds[6][0]), 0.5*(bnds[7][0]-bnds[7][1]))

        res = minimize(Path3Delay,xi,args=(n,m,c1,c2,c3,c4,c5,c6,c7),jac=True, method='L-BFGS-B',bounds=bnds)

        if (res.success)&(amax(abs(array(res.jac)))<=jactol):

            T = res.fun
            X = res.x
            y = X[7]

        else:

            T = nan
            X = nan
            y = nan



        return (T,X,y)

    def Path3CorrectionForward(n,m,X,c1,c2,c3,c4,c5,c6,c7):

        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        x4 = X[4]
        x5 = X[5]
        x6 = X[6]
        y = X[7]

        rs = 7.8
        rw = 1.05
        css = cs['T']
        csL = cs['L']

        ti = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        ttl = arcsin(complex((csL/cw)*sin(ti)))
        tts = arcsin(complex((css/cw)*sin(ti)))

        M = TRDict['WedgeToSteel'](ti,ttl,tts,cw,css,csL,rs,rw)
        RL,TL,TS = solve(M[0],M[1])

        if c1 == csL:

            A1 = TL

            ti = arctan2((x1-x0),l0)
            trl = ti
            trs = arcsin(complex((css/csL)*sin(ti)))

            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

        else:

            A1 = TS

            ti = arctan2((x1-x0),l0)
            trs = ti
            trl = arcsin(complex((csL/css)*sin(ti)))

            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

            else:

                A2 = RS

                yp = (-l3*x2 + l3*l1 - l3*l2)/(l2)
                ti = pi/2 - arctan2((l0 - yp),(x2 - x1)) + arctan2(l3,l2)
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    ti = arctan2((abs(y - yp)),(l1 - x2))
                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                        yL = (l2/l3)*(l1 - x3) + yp
                        beta = arctan2(l2,l3)

                        if y >= yL:

                            alpha = arctan2((l1 - x3),(y - yp))
                            ti = pi/2 - (alpha + beta)

                        else:

                            if y > yp:

                                alpha = arctan2((y -yp),(l1 - x3))
                                ti = beta - alpha

                            else:

                                ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)

                            ti = arctan2((x4 - yp),(l1 - x3))
                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((l1 - x5),(l0 - x4))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

        Path3TRCorrectionForward = (A1*A2*A3*A4*A5*A6*A7*A8)[0]

        yp1 = (-l3*x2 + l3*l1 - l3*l2)/(l2)
        yp2 = (-l3*x3 + l3*l1 - l3*l2)/(l2)

        Path3GeometricalCorrectionForward = sqrt(sqrt((x0 - n*p*cphi)**2 + (h + n*p*sphi)**2) + sqrt((x1 - x0)**2 + l0**2) + sqrt((x2 - x1)**2 + (l0 - yp1)**2) + sqrt((l1 - x2)**2 + (y - yp1)**2))*sqrt(sqrt((l1 - x3)**2 + (y - yp2)**2) + sqrt((l1 - x3)**2 + (x4 - yp2)**2) + sqrt((l1 - x5)**2 + (l0 - x4)**2) + sqrt((x5 - x6)**2 + (l0)**2) + sqrt((x6 - m*p*cphi)**2 + (h + m*p*sphi)**2))

        tin = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        tdirn = abs(arcsin(sphi) - tin)

        tim = arctan2((x6 - m*p*cphi),(h + m*p*sphi))
        tdirm = abs(arcsin(sphi) - tim)

        landa = cw/5
        Path3Directivity = sinc(p*sin(tdirn)/landa)*sinc(p*sin(tdirm)/landa)
        if Path3Directivity < 0:
            Path3Directivity = 0

        return sign(real(Path3TRCorrectionForward))*Path3GeometricalCorrectionForward/(Path3Directivity*abs(Path3TRCorrectionForward)+eps)

    def Path3CorrectionBackward(n,m,X,c1,c2,c3,c4,c5,c6,c7):

        x0 = X[0]
        x1 = X[1]
        x2 = X[2]
        x3 = X[3]
        x4 = X[4]
        x5 = X[5]
        x6 = X[6]
        y = X[7]

        rs = 7.8
        rw = 1.05
        css = cs['T']
        csL = cs['L']

        ti = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        ttl = arcsin(complex((csL/cw)*sin(ti)))
        tts = arcsin(complex((css/cw)*sin(ti)))

        M = TRDict['WedgeToSteel'](ti,ttl,tts,cw,css,csL,rs,rw)
        RL,TL,TS = solve(M[0],M[1])

        if c1 == csL:

            A1 = TL

            ti = arctan2((x1-x0),l0)
            trl = ti
            trs = arcsin(complex((css/csL)*sin(ti)))

            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                ti = arctan2((l0 - x2),(l1 - x1))
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

            else:

                A2 = RS

                ti = arctan2((l0 - x2),(l1 - x1))
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

        else:

            A1 = TS

            ti = arctan2((x1-x0),l0)
            trs = ti
            trl = arcsin(complex((csL/css)*sin(ti)))

            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
            RL,RS = solve(M[0],M[1])

            if c2 == csL:

                A2 = RL

                ti = arctan2((l0 - x2),(l1 - x1))
                trl = ti
                trs = arcsin(complex((css/csL)*sin(ti)))

                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

            else:

                A2 = RS

                ti = arctan2((l0 - x2),(l1 - x1))
                trs = ti
                trl = arcsin(complex((csL/css)*sin(ti)))

                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                RL,RS = solve(M[0],M[1])

                if c3 == csL:

                    A3 = RL

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trl = ti
                    trs = arcsin(complex((css/csL)*sin(ti)))

                    M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                else:

                    A3 = RS

                    yp = (-l3*x3 + l3*l1 - l3*l2)/(l2)
                    yL = (l2/l3)*(l1 - x3) + yp
                    beta = arctan2(l2,l3)

                    if y >= yL:

                        alpha = arctan2((l1 - x3),(y - yp))
                        ti = pi/2 - (alpha + beta)

                    else:

                        if y > yp:

                            alpha = arctan2((y -yp),(l1 - x3))
                            ti = beta - alpha

                        else:

                            ti = arctan2((yL - yp),(l1 - x3)) + arctan2((yp - y),(l1 - x3))

                    trs = ti
                    trl = arcsin(complex((csL/css)*sin(ti)))

                    M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                    RL,RS = solve(M[0],M[1])

                    if c4 == csL:

                        A4 = RL

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trl = ti
                        trs = arcsin(complex((css/csL)*sin(ti)))

                        M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                    else:

                        A4 = RS

                        ti = arctan2(abs(y - yp),(l1 - x3))
                        trs = ti
                        trl = arcsin(complex((csL/css)*sin(ti)))

                        M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                        RL,RS = solve(M[0],M[1])

                        if c5 == csL:

                            A5 = RL

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)

                            trl = ti
                            trs = arcsin(complex((css/csL)*sin(ti)))

                            M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                        else:

                            A5 = RS

                            yp = (-l3*x4 + l3*l1 - l3*l2)/(l2)
                            beta = arctan(l3/l2)

                            if yp > y:

                                alpha = arctan((yp - y)/(l1 - x4))
                                ti = pi/2 - beta + alpha

                            else:

                                alpha = arctan((y - yp)/(l1 - x4))
                                ti = pi/2 - (alpha + beta)


                            trs = ti
                            trl = arcsin(complex((csL/css)*sin(ti)))

                            M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                            RL,RS = solve(M[0],M[1])

                            if c6 == csL:

                                A6 = RL

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trl = ti
                                trs = arcsin(complex((css/csL)*sin(ti)))

                                M = TRDict['SteelToVacuumCompressionIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                            else:

                                A6 = RS

                                ti = arctan2((x4 - x5),(l0 - yp))
                                trs = ti
                                trl = arcsin(complex((csL/css)*sin(ti)))

                                M = TRDict['SteelToVacuumShearIncidence'](ti,trs,trl,css,csL,rs)
                                RL,RS = solve(M[0],M[1])

                                if c7 == csL:

                                    A7 = RL

                                    ti = arctan2((x5 - x6),l0)
                                    trl = ti
                                    trs = arcsin(complex((css/csL)*sin(ti)))
                                    ttl = arcsin(complex((cw/csL)*sin(ti)))

                                    M = TRDict['SteelToWedgeCompressionIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

                                else:

                                    A7 = RS

                                    ti = arctan2((x5 - x6),l0)
                                    trs = ti
                                    trl = arcsin(complex((csL/css)*sin(ti)))
                                    ttl = arcsin(complex((cw/css)*sin(ti)))

                                    M = TRDict['SteelToWedgeShearIncidence'](ti,ttl,trs,trl,cw,css,csL,rs,rw)
                                    TL,RL,RS = solve(M[0],M[1])

                                    A8 = TL

        Path5TRCorrectionBackward = (A1*A2*A3*A4*A5*A6*A7*A8)[0]

        yp1 = (-l3*x3 + l3*l1 - l3*l2)/(l2)
        yp2 = (-l3*x4 + l3*l1 - l3*l2)/(l2)

        Path5GeometricalCorrectionBackward = sqrt(sqrt((x0 - n*p*cphi)**2 + (h + n*p*sphi)**2) + sqrt((x1 - x0)**2 + l0**2) + sqrt((l1 - x1)**2 + (l0 - x2)**2) + sqrt((l1 - x3)**2 + (x2 - yp1)**2) + sqrt((l1 - x3)**2 + (y - yp1)**2))*sqrt(sqrt((l1 - x4)**2 + (y - yp2)**2) + sqrt((x4 - x5)**2 + (l0 - yp2)**2) + sqrt((x6 - x5)**2 + (l0)**2) + sqrt((x6 - m*p*cphi)**2 + (h + m*p*sphi)**2))

        tin = arctan2((x0 - n*p*cphi),(h + n*p*sphi))
        tdirn = abs(arcsin(sphi) - tin)

        tim = arctan2((x6 - m*p*cphi),(h + m*p*sphi))
        tdirm = abs(arcsin(sphi) - tim)

        landa = cw/5
        Path5Directivity = sinc(p*sin(tdirn)/landa)*sinc(p*sin(tdirm)/landa)
        if Path5Directivity < 0:
            Path5Directivity = 0

        return sign(real(Path5TRCorrectionBackward))*Path5GeometricalCorrectionBackward/(Path5Directivity*abs(Path5TRCorrectionBackward)+eps)

# new double side wall paths
    def Path4Delay(x,n,m,c1,c2,c3,c4,c5,c6):

        x0,x1,y1,x2,y2,x3,x4 = x[0],x[1],x[2],x[3],x[4],x[5],x[6]

        f = sqrt((h + m*p*sin(phi))**2 + (-m*p*cos(phi) + x4)**2)/cw + sqrt((h + n*p*sin(phi))**2 + (-n*p*cos(phi) + x0)**2)/cw + sqrt(l0**2 + (x3 - x4)**2)/c6 + sqrt((l0 - y2)**2 + (l1 - x3)**2)/c5 + sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y2 + l3*x2)**2/l2**2)/c4 + sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y1 + l3*x2)**2/l2**2)/c3 + sqrt((l0 - y1)**2 + (l1 - x1)**2)/c2 + sqrt(l0**2 + (x0 - x1)**2)/c1

        J = [-(n*p*cos(phi) - x0)/(cw*sqrt((h + n*p*sin(phi))**2 + (n*p*cos(phi) - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), -(l1 - x1)/(c2*sqrt((l0 - y1)**2 + (l1 - x1)**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (-l1*l3 + l2*l3 + l2*y1 + l3*x2)/(c3*l2*sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y1 + l3*x2)**2/l2**2)) - (l0 - y1)/(c2*sqrt((l0 - y1)**2 + (l1 - x1)**2)), (-l1 + x2 + l3*(-l1*l3 + l2*l3 + l2*y2 + l3*x2)/l2**2)/(c4*sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y2 + l3*x2)**2/l2**2)) + (-l1 + x2 + l3*(-l1*l3 + l2*l3 + l2*y1 + l3*x2)/l2**2)/(c3*sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y1 + l3*x2)**2/l2**2)), -(l0 - y2)/(c5*sqrt((l0 - y2)**2 + (l1 - x3)**2)) + (-l1*l3 + l2*l3 + l2*y2 + l3*x2)/(c4*l2*sqrt((l1 - x2)**2 + (-l1*l3 + l2*l3 + l2*y2 + l3*x2)**2/l2**2)), (x3 - x4)/(c6*sqrt(l0**2 + (x3 - x4)**2)) - (l1 - x3)/(c5*sqrt((l0 - y2)**2 + (l1 - x3)**2)), -(m*p*cos(phi) - x4)/(cw*sqrt((h + m*p*sin(phi))**2 + (m*p*cos(phi) - x4)**2)) - (x3 - x4)/(c6*sqrt(l0**2 + (x3 - x4)**2))]

        return f,array(J)

    def Path4(n,m,c1,c2,c3,c4,c5,c6):

        bnds = ((n*p*cphi,l1-l2),(n*p*cphi, l1),(-l3,l0),(l1-l2,l1),(-l3,l0),(m*p*cphi,l1),(m*p*cphi,l1-l2))

        xi = (0.5*(bnds[0][1]-bnds[0][0]),0.5*(bnds[1][1]-bnds[1][0]),0.5*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]),0.75*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]),0.5*(bnds[6][1]-bnds[6][0]))

        res = minimize(Path4Delay,xi, args=(n,m,c1,c2,c3,c4,c5,c6), jac=True, method='L-BFGS-B',bounds=bnds)

        if (res.success)&(amax(abs(array(res.jac)))<=jactol):

            T = res.fun
            X = res.x
            y1 = X[2]
            y2 = X[4]

        else:

            T = nan
            X = nan
            y1 = nan
            y2 = nan

        return (T,X,y1,y2)


# Symmetric to cap
    def Path5Delay(x,n,m,c1,c2,c3):

        x0,x1,x2 = x[0],x[1],x[2]

        f = sqrt((h + n*p*sin(phi))**2 + (-n*p*cos(phi) + x0)**2)/cw + l1/c3 - x2/c3 + sqrt((x1 - x2)**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x2)**2/l2**2)/c2 + sqrt(l0**2 + (x0 - x1)**2)/c1

        J = [-(n*p*cos(phi) - x0)/(cw*sqrt((h + n*p*sin(phi))**2 + (n*p*cos(phi) - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), (x1 - x2)/(c2*sqrt((x1 - x2)**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x2)**2/l2**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), -1/c3 + (-x1 + x2 + l3*(l0*l2 - l1*l3 + l2*l3 + l3*x2)/l2**2)/(c2*sqrt((x1 - x2)**2 + (l0*l2 - l1*l3 + l2*l3 + l3*x2)**2/l2**2))]

        return f,array(J)

#  Symmetric to SideWall

    def Path6Delay(x,m,n,c1,c2,c3):

        x0,x1,y = x[0],x[1],x[2]

        f = sqrt((h + n*p*sin(phi))**2 + (-n*p*cos(phi) + x0)**2)/cw + sqrt((l2*l3**2 + l2*l3*y)**2/(l2**2 + l3**2)**2 + (l2**3*l3 + l2**3*y)**2/(l2**2*(l2**2 + l3**2)**2))/c3 + sqrt((l0 - y)**2 + (l1 - x1)**2)/c2 + sqrt(l0**2 + (x0 - x1)**2)/c1

        J = [-(n*p*cos(phi) - x0)/(cw*sqrt((h + n*p*sin(phi))**2 + (n*p*cos(phi) - x0)**2)) + (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), -(l1 - x1)/(c2*sqrt((l0 - y)**2 + (l1 - x1)**2)) - (x0 - x1)/(c1*sqrt(l0**2 + (x0 - x1)**2)), ((2 - 2*l3/(l2*(l2/l3 + l3/l2)))*(y + (-l1*l3 + l2*l3 + l3*(l1*l2/l3 + l1*l3/l2 - l3 - y)/(l2/l3 + l3/l2))/l2)/2 + (l1 - (l1*l2/l3 + l1*l3/l2 - l3 - y)/(l2/l3 + l3/l2))/(l2/l3 + l3/l2))/(c3*sqrt((l1 - (l1*l2/l3 + l1*l3/l2 - l3 - y)/(l2/l3 + l3/l2))**2 + (y + (-l1*l3 + l2*l3 + l3*(l1*l2/l3 + l1*l3/l2 - l3 - y)/(l2/l3 + l3/l2))/l2)**2)) - (l0 - y)/(c2*sqrt((l0 - y)**2 + (l1 - x1)**2))]

        return f,array(J)



    ForTFM = {}

    ForTFM['path1'] = {}
    ForTFM['path1']['ConvergenceCheck'] = lambda c: [[Path1(n,m,c[0],c[1],c[2],c[3],c[4]) for m in range(N)] for n in range(N)]
    ForTFM['path1']['TFMInput'] = lambda M,c: [((n,m,M[n][m][2],M[n][m][0],Path1CorrectionForward(n,m,M[n][m][1],c[0],c[1],c[2],c[3],c[4])),(m,n,M[n][m][2],M[n][m][0],Path1CorrectionBackward(n,m,M[n][m][1],c[0],c[1],c[2],c[3],c[4]))) for m in range(N) for n in range(N) if isfinite(M[n][m][0])]

    ForTFM['path2'] = {}
    ForTFM['path2']['ConvergenceCheck'] = lambda c: [[Path2(n,m,c[0],c[1],c[2],c[3],c[4],c[5]) for m in range(N)] for n in range(N)]
    ForTFM['path2']['TFMInput'] = lambda M,c: [(n,m,M[n][m][2],M[n][m][0],Path2Correction(n,m,M[n][m][1],c[0],c[1],c[2],c[3],c[4],c[5])) for m in range(N) for n in range(N) if isfinite(M[n][m][0])]

    ForTFM['path3'] = {}
    ForTFM['path3']['ConvergenceCheck'] = lambda c: [[Path3(n,m,c[0],c[1],c[2],c[3],c[4],c[5],c[6]) for m in range(N)] for n in range(N)]
    ForTFM['path3']['TFMInput'] = lambda M,c: [((n,m,M[n][m][2],M[n][m][0],Path3CorrectionForward(n,m,M[n][m][1],c[0],c[1],c[2],c[3],c[4],c[5],c[6])),(m,n,M[n][m][2],M[n][m][0],Path3CorrectionBackward(n,m,M[n][m][1],c[0],c[1],c[2],c[3],c[4],c[5],c[6]))) for m in range(N) for n in range(N) if isfinite(M[n][m][0])]

    ForTFM['path4'] = {}
    ForTFM['path4']['ConvergenceCheck'] = lambda c: [[Path4(n,m,c[0],c[1],c[2],c[3],c[4],c[5]) for m in range(N)] for n in range(N)]
    ForTFM['path4']['TFMInput'] = lambda M,c: [((n,m,M[n][m][2],M[n][m][3],M[n][m][0],1),(m,n,M[n][m][2],M[n][m][3],M[n][m][0],1)) for m in range(N) for n in range(N) if isfinite(M[n][m][0])]

# n,m,y,T,C
# n,m,y1,y2,T,C
    pthkey = Path[0]
    C = [cs[pol] for pol in Path[1]]
    I = ForTFM[pthkey]['TFMInput'](ForTFM[pthkey]['ConvergenceCheck'](C),C)

    ForImaging = []

    if I:

        if len(I[0]) > 2:

            ForImaging = I

        else:

            for i in range(0,len(I)):

                ForImaging.append(I[i][0])
                ForImaging.append(I[i][1])

    # else:
    #
    #     ForImaging = nan


    return ForImaging

def SymmetricDelays(Path,Elements,WeldParameters,ProbeParameters={'Pitch':0.6,'NumberOfElements':32},WedgeParameters={'Velocity':2.33,'Height':15.1,'Angle':10.0}):

    from scipy.optimize import minimize

    cw = WedgeParameters['Velocity']
    h = WedgeParameters['Height']
    phi = WedgeParameters['Angle']*(pi/180)
    p = ProbeParameters['Pitch']
    N = ProbeParameters['NumberOfElements']

    l0 = WeldParameters['Thickness']
    l1 = WeldParameters['SideWallPosition']
    l2 = WeldParameters['VerticalLOF']
    l3 = WeldParameters['HorizontalLOF']

    cs = {'L':5.9,'T':3.24}

    cphi = cos(phi)
    sphi = sin(phi)

    tphi = sphi/cphi

    def SideWallCap(m,c):

        f = lambda x: 2*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)/cw + 2*sqrt((l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2/l2**2)/c[2] + 2*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)/c[1] + 2*sqrt(l0**2 + (x[0] - x[1])**2)/c[0]

        J = lambda x: array([2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2))])

        H = lambda x: array([[2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))*(cphi*m*p - x[0])/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)**(3/2)) + 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2) + c[0]*(x[0] - x[1])*(-cphi*m*p + x[0])/sqrt(l0**2 + (x[0] - x[1])**2) + cw*(x[0] - x[1])*(-cphi*m*p + x[0])/sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2) + cw*sqrt((h + m*p*sphi)**2 +(cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 2*(-x[0] + x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 2*(c[0]*(-x[0] + x[1])*(-cphi*m*p + x[0])/sqrt(l0**2 + (x[0] - x[1])**2) - cw*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)) + 2*(x[0] - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-cphi*m*p + x[0]) + cw*(x[0] - x[1])*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2))/(c[0]*cw*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((h + m*p*sphi)**2 + (cphi*m*p - x[0])**2)), 0, 0], [2*(c[0]*(-l1 + x[1])*(x[0] - x[1])/sqrt(l0**2 + (x[0] - x[1])**2) - c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(-x[0] + x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(l1 - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2) + c[0]*(-l1 + x[1])*(-x[0] + x[1])/sqrt(l0**2 + (x[0] - x[1])**2) + c[1]*(-l1 + x[1])*(-x[0] + x[1])/sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(x[0] - x[1])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*(l0**2 + (x[0] - x[1])**2)**(3/2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), 2*(-l0 + x[2])*(-x[0] + x[1])/(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l0 - x[2])*(c[0]*sqrt(l0**2 + (x[0] - x[1])**2)*(-l1 + x[1]) + c[1]*(-x[0] + x[1])*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))/(c[0]*c[1]*sqrt(l0**2 + (x[0] - x[1])**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), 0], [0, 2*(-l1 + x[1])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l1 - x[1])*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)), -2*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[1]*c[2]*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(l0 - x[2])*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*((l0 - x[2])**2 + (l1 - x[1])**2)**(3/2)) + 2*(c[1]*l2*sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[1]*(-l0 + x[2])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2])*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)), -(l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(c[1]*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2]))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2)) + 2*(c[1]*l3*sqrt((l0 - x[2])**2 + (l1 - x[1])**2) + c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(-l0 + x[2])*(l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(2*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)))/(c[1]*c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*sqrt((l0 - x[2])**2 + (l1 - x[1])**2))], [0, 0, 2*l3/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)) - 2*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])/(c[2]*l2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)), 2*(l2**2 + l3**2)/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)) - (l2**2*(-2*l1 + 2*x[3]) + 2*l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))*(l2**2*(-l1 + x[3]) + l3*(-l1*l3 + l2*l3 + l2*x[2] + l3*x[3]))/(c[2]*l2**2*sqrt((l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2)/l2**2)*(l2**2*(l1 - x[3])**2 + (-l1*l3 + l2*l3 + l2*x[2] + l3*x[3])**2))]])

        bnds = ((m*p*cphi,l1-l2),(m*p*cphi, l1),(0.,l0),(l1-l2,l1))

        # xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]), 0.5*(bnds[3][1]-bnds[3][0]),0.75*(bnds[4][1]-bnds[4][0]),0.5*(bnds[5][1]-bnds[5][0]),(0.5*(bnds[6][1]-bnds[6][0])))

        xi = (0.5*(bnds[0][1]-bnds[0][0]), 0.5*(bnds[1][1]-bnds[1][0]), 0.5*(bnds[2][1]-bnds[2][0]),0.5*(bnds[3][1]-bnds[3][0]))

        res = minimize(f,xi,method='Newton-CG',jac=J,hess=H,options={'disp': False, 'xtol': 1e-03, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None})

        if (res.success)&(CheckBounds([res.x[2],res.x[3]],[bnds[2],bnds[3]])):

            # T = sqrt(res.fun)

            T = res.fun

            ntr = array([(h+m*p*sphi)*tphi,h+m*p*sphi])
            vtr = array([res.x[0]-m*p*cphi,h+m*p*sphi])

            # nrc = array([(h+n*p*sphi)*tphi,h+n*p*sphi])
            # vrc = array([res.x[-1]-n*p*cphi,h+n*p*sphi])

            th = (180/pi)*sign(vtr[0]-ntr[0])*arccos(vdot(vtr,ntr)/(norm(vtr)*norm(ntr)))


        else:

            T = nan
            th = nan

        return T,th



    DelayFunctions = {}

    c = [cs[cc] for cc in Path[1]]

    DelayFunctions['SideWallCap'] = lambda m,c: SideWallCap(m,c)

    d = [DelayFunctions[Path[0]](m,c) for m in Elements ]

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
    l2 = WeldParameters['VerticalLOF']
    l3 = WeldParameters['HorizontalLOF']

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


    DelayFunctions = {}

    DelayFunctions['Corner'] = lambda m,n,c: Corner(m,n,c)
    DelayFunctions['SideWallCap'] = lambda m,n,c: SideWallCap(m,n,c)
    DelayFunctions['SymmetricSideWallCap'] = lambda m,c: SymmetricSideWallCap(m,n,c)


    c = [cs[cc] for cc in Path[1]]

    d = [[ DelayFunctions[Path[0]](m,n,c) for n in Elements[1] ] for m in Elements[0] ]

    return d



class FMC:

    def __init__(self,fsamp,weldparameters={'Thickness':30., 'VerticalLOF': 8.0, 'HorizontalLOF': 8.0, 'SideWallPosition': 34.16, 'Velocity':{'Longitudinal':5.9,'Transverse':3.24}, 'Impedance':{'Longitudinal':5.9*7.8,'Transverse':3.23*7.8}},weldtype='L',probeid='5L32',wedgeid='L10'):

        self.SamplingFrequency = fsamp

        Wedge = {}
        Wedge['L10'] = {'Height':15.12,'Angle':10.,'Velocity':2.33,'Impedance':{'Longitudinal':2.33*1.04,'Transverse':2.33*1.04/2}}
        Wedge['Black'] = {'Height':6.4,'Angle':31.6,'Velocity':2.33,'Impedance':{'Longitudinal':2.33*1.04,'Transverse':2.33*1.04/2}}

        Probe = {}
        Probe['5L32'] = {'Pitch':0.6,'NumberOfElements':32,'CentreFrequency':5.0}

        self.WeldType = weldtype
        self.WeldParameters = weldparameters

        self.WedgeParameters = Wedge[wedgeid]
        self.ProbeParameters = Probe[probeid]



        DelayFunction = {}

        DelayFunction['L'] = lambda pth: LWeldDelays(pth,self.WeldParameters,self.ProbeParameters,self.WedgeParameters)

        self.DelayFunction = DelayFunction[weldtype]

        self.FusionLine = []

    def LoadAScans(self,inpt,Format='civa'):

        from Civa import LoadAScansFromTxt

        loader = {}

        loader['civa'] = lambda x: LoadAScansFromTxt(x)

        loader['array'] = lambda x: x

        # loader['esndt'] = lambda x:


        self.AScans = loader[Format](inpt)

        # if ApplyHilbert:
        #
        #     self.AScans = hilbert(A,axis=2)
        #
        # else:
        #
        #     self.AScans = A

        # self.GetSpectrum()
        #
        self.Time = linspace(0,self.AScans.shape[2]/self.SamplingFrequency,self.AScans.shape[2])

        self.Tend = self.Time[-1]


    def Calibrate(self,RefParams=(10,0.1),BPParams=(1.0,8.5,2.),Offset=0.75,BWRange = (22.,37.)):

        from scipy.signal import detrend,tukey,firwin,fftconvolve,hilbert

        # AScans are windowed (to remove main bang + inter-element talk), band-passed, back-wall estimated, and reference taken from back-wall.
        # RefParams - tuple (Number of Cycles, Tukey Window Alpha)
        # BPParms - tuple (Low Frequency, High Frequency, )
        # Offset - float amount to remove from beginning of scans (fraction of wedge height)

        dt = 1/self.SamplingFrequency

        phi = (pi/180)*self.WedgeParameters['Angle']


        L = self.AScans.shape[-1]

        N = self.ProbeParameters['NumberOfElements']

        T = 2*Offset*self.WedgeParameters['Height']/self.WedgeParameters['Velocity']

        self.AScans[:,:,0:int(T/dt)] = 0.

        self.AScans[:,:,int(T/dt)::] = detrend(self.AScans[:,:,int(T/dt)::],bp=[0,int((L-(T/dt))/3),L-(int(T/dt))])



        # self.AScans = self.AScans*tukey(L,0.1)

        # h = firwin(L-1,[BPParams[0]/(self.SamplingFrequency*0.5),BPParams[1]/(self.SamplingFrequency*0.5)],width=BPParams[2]/(0.5*self.SamplingFrequency),pass_zero=False)

        h = firwin(L-1,[BPParams[0]/(self.SamplingFrequency*0.5),BPParams[1]/(self.SamplingFrequency*0.5)],pass_zero=False)

        #
        # self.Correlation = zeros((N,N,L))
        #
        # self.CorrelationPeaks = zeros((N,N,L))
        #
        # Nwin = int(RefParams[0]/(dt*self.ProbeParameters['CentreFrequency']))
        #
        #
        # self.References = zeros((N,N,Nwin))



        for m in range(N):
            for n in range(N):

                # print(m)
                # print(n)

                self.AScans[m,n,:] = fftconvolve(self.AScans[m,n,:],h,mode='same')

                # self.References[m,n,:] = EstimateReference(self.AScans[m,n,:],Nwin)
                #
                # self.Correlation[m,n,:] = NormalizedCrossCorrelate(self.References[m,n,:],self.AScans[m,n,:])
                #
                # self.CorrelationPeaks[m,n,:] = PeakBinarize(self.CorrelationPeaks[m,n,:],0.1,5)


        # self.GetSpectrum()
        #
        # p = self.PlaneWaveFocus((-self.WedgeParameters['Angle'],-self.WedgeParameters['Angle']))
        #
        # P = abs(p)
        #
        # Nwin = int(RefParams[0]/(dt*self.ProbeParameters['CentreFrequency']))
        #
        # self.Window = tukey(Nwin,RefParams[1])+0j
        #
        # Tw = 2*(self.WedgeParameters['Height']+N*sin(phi)*self.ProbeParameters['Pitch']/2)/self.WedgeParameters['Velocity']
        # Tbw = ((Tw + 2*BWRange[0]/self.WeldParameters['Velocity']['Longitudinal'],Tw + 2*BWRange[1]/self.WeldParameters['Velocity']['Longitudinal']),(Tw + 4*BWRange[0]/self.WeldParameters['Velocity']['Longitudinal'],Tw + 4*BWRange[1]/self.WeldParameters['Velocity']['Longitudinal']))
        #
        #
        # indbw = ((int(round(Tbw[0][0]/dt)),int(round(Tbw[0][1]/dt))),(int(round(Tbw[1][0]/dt)),int(round(Tbw[1][1]/dt))))
        #
        # indbwmax = (argmax(P[indbw[0][0]:indbw[0][1]+1])+indbw[0][0], argmax(P[indbw[1][0]:indbw[1][1]+1])+indbw[1][0])
        #
        # self.Reference = -p[indbwmax[0] - int(round(Nwin/2)):indbwmax[0] - int(round(Nwin/2)) + Nwin]*self.Window
        #
        # self.WeldParameters['Thickness'] = self.WeldParameters['Velocity']['Longitudinal']*dt*(indbwmax[1]-indbwmax[0])/2
        #
        # self.EstimateSideWall()

        # self.AScans = hilbert(self.AScans,axis=2)


    def GetSpectrum(self):


        self.TemporalSpectrum = rfft(real(self.AScans),axis=2)

        # self.SpatialSpectrum = fftshift(fftn(self.TemporalSpectrum,axes=(0,1)))
        # self.SpatialSpectrum = self.SpatialSpectrum/(prod(array(self.SpatialSpectrum.shape)))
        #
        # self.SpatialFrequency = linspace(-1/(2*self.ProbeParameters['Pitch']),1/(2*self.ProbeParameters['Pitch']),self.SpatialSpectrum.shape[0])

        self.Frequency = linspace(0.,self.SamplingFrequency/2.,self.TemporalSpectrum.shape[2])


    def PlaneWaveFocus(self,angles):

        from numpy import sum as asum

        d = self.ProbeParameters['Pitch']*self.ProbeParameters['NumberOfElements']

        d = linspace(-d/2,d/2,self.ProbeParameters['NumberOfElements'])


        T = meshgrid(self.Frequency,d*sin(pi*angles[1]/180)/self.WedgeParameters['Velocity'])

        X = asum(self.TemporalSpectrum*exp(-2j*pi*T[0]*T[1]),axis=1,keepdims=False)

        T = meshgrid(self.Frequency,d*sin(pi*angles[0]/180)/self.WedgeParameters['Velocity'])

        X = asum(X*exp(-2j*pi*T[0]*T[1]),axis=0,keepdims=False)

        x = ifft(X,n=2*(len(X)-1))

        return x

    def PlaneWaveSweep(self,TRangles, RCangles):

        x = array([[ self.PlaneWaveFocus((tr,rc)) for rc in RCangles] for tr in TRangles])

        return x

    def EstimateSideWall(self,SWRange = (32.,45.),dl=0.3):

        Lref = len(self.Reference)

        wp = self.WeldParameters.copy()


        def SideWallError(x):

            wp['SideWallPosition'] = x


            elements = ((range(15,16),range(15,16)))


            T = [ Delays(('Corner',p),elements,wp) for p in [('L','L','L'),('L','L','T')]]

            R = [ InnerProduct(real(self.Reference), real(self.PlaneWaveFocus(TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2)):int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2))+Lref])) for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T if isfinite(TT[m][n][0])]

            R = array(R)

            if len(R)>0:

                R = -dot(R,conj(R))/len(R)

            else:

                R = 0.

            return real(R)

        swgrid = linspace(SWRange[0],SWRange[1],int((SWRange[1]-SWRange[0])/dl))

        SWErrGrid = array([SideWallError(sw) for sw in swgrid])


        self.WeldParameters['SideWallPosition'] = swgrid[argmin(SWErrGrid)]


    def EstimateCap(self,CapRanges=((4.7,8.3),(2.7,8.3)),dl=0.3):

        import itertools
        from scipy.optimize import basinhopping

        wp = self.WeldParameters.copy()

        Lref = len(self.Reference)


        def CapError(x):

            wp['VerticalLOF'] = x[0]
            wp['HorizontalLOF'] = x[1]

            # elements = ((range(15,16)),(range(15,16)))

            elements = range(0,31)


            # modes = itertools.product(['L','T'],repeat = 3)



            # modes = [('L', 'T', 'L', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L'),('L', 'L', 'T', 'L', 'T', 'T'),('T', 'L', 'L', 'L', 'L', 'T'),('L', 'T', 'T', 'L', 'T', 'L')]

            # modes = [('L', 'T', 'L', 'T', 'T', 'L'),('T', 'T', 'L', 'L', 'L', 'L'),('L', 'T', 'L', 'T', 'L', 'L'),('L', 'T', 'T', 'T', 'T', 'T'),('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'L', 'L', 'T', 'L')]

            # modes = [('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L')]

            # modes = [('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L'),('L', 'L', 'L', 'L', 'L', 'L')]


            # modes = [('L', 'T', 'T', 'L', 'T', 'L')]
            #
            # modes = [('L','L','L'),('L','L','T'),('T')]

            # modes = [('T', 'T', 'T', 'T', 'T', 'L')]

            # modes = [('L','L','L'),('L','L','T')]

            modes = [('L','L','L')]

            # modes = [('L', 'T', 'L', 'T', 'T', 'L'),('T', 'T', 'L', 'L', 'L', 'L'),('T', 'T', 'T', 'T', 'T', 'L'),('L', 'T', 'T', 'L', 'T', 'L')]

            # T = [ Delays(('SideWallCap',p),elements,wp) for p in modes ]

            # T = [ Delays(('SymmetricSideWallCap',p),elements,wp) for p in modes ]

            # T = [ Delays(('SymmetricSideWallCap',p),elements,wp) for p in modes ]

            T = [SymmetricDelays(('SideWallCap',p),elements,wp) for p in modes ]

            if any(isfinite(array(T[0][0:-1][0]))):

                print(x[0])
                print(x[1])

            # R = [ InnerProduct(self.Reference, self.PlaneWaveFocus(TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2)):int(round(TT[m][n][0]*self.SamplingFrequency))-int(round(Lref/2))+Lref],ipopt='Normalized') for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T if isfinite(TT[m][n][0])]

            # R = [ self.PlaneWaveFocus(TT[m][n][1])[int(round(TT[m][n][0]*self.SamplingFrequency))] for n in range(len(elements[1])) for m in range(len(elements[0])) for TT in T if isfinite(TT[m][n][0])]


            R = [InnerProduct(self.Reference,self.AScans[elements[m],elements[m],int(round(TT[m][0]*self.SamplingFrequency))-int(round(Lref/2)):int(round(TT[m][0]*self.SamplingFrequency))-int(round(Lref/2))+Lref]) for m in range(len(elements)) for TT in T if isfinite(TT[m][0])]

            # R = [ self.AScans[elements[m],elements[m],int(round(TT[m][0]*self.SamplingFrequency))] for m in range(len(elements)) for TT in T if isfinite(TT[m][0])]

            # R = abs(array(R))

            R = array(R)

            # print(R.shape)
            # print(conj(R).shape)


            if len(R)>0:

                R = -real(dot(R,conj(R)))/len(R)

                # R = -dot(R,conj(R))

            else:

                R = 0.



            return R


        vgrid = linspace(CapRanges[0][0],CapRanges[0][1],int((CapRanges[0][1]-CapRanges[0][0])/dl))
        #
        # print(vgrid)
        #
        hgrid = linspace(CapRanges[1][0],CapRanges[1][1],int((CapRanges[1][1]-CapRanges[1][0])/dl))

        # res = basinhopping(CapError,(7.,4.))

        vhgrid = [(v,h) for v in vgrid for h in hgrid if h<=v]

        CapErrGrid = array([CapError(vh) for vh in vhgrid])

        plot(CapErrGrid)

        show()

        # print(res.x)

        print(vhgrid[argmin(CapErrGrid)])


        # self.WeldParameters['VerticalLOF'] = vhgrid[argmin(CapErrGrid)][0]
        #
        # self.WeldParameters['HorizontalLOF'] = vhgrid[argmin(CapErrGrid)][1]


    def GetDelays(self,path,wp=None):

        if wp is not None:

            for w in wp:

                self.WeldParameters[w[0]] = w[1]


        return self.DelayFunction(path)

    def FusionLineFocus(self,Delays):

        # IM = Delays

        # n,m,y1,y2,T,C

        L = len(self.Window)

        GetAmplitude = lambda p: self.AScans[p[0],p[1],int(round(self.SamplingFrequency*p[2]))]
        GetInnerProduct = lambda m,n,T: InnerProduct(real(self.Reference),real(self.AScans[m,n,int(round(self.SamplingFrequency*T))-int(round(L/2)):int(round(self.SamplingFrequency*T))-int(round(L/2))+L])*real(self.Window))

        # I = [[GetAmplitude((IM[0],IM[1],IM[3]))*IM[4], sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]), IM[2]] for IM in Delays if len(IM)>0]


        I = []

        IP = []
        Y = []

        # if Delays:
        #
        #     for IM in Delays:
        #
        #         I.append(GetAmplitude((IM[0],IM[1],IM[3]))*IM[4])
        #
        #         IP.append(sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]))
        #
        #
        #         Y.append(IM[2])
        #
        #     # self.FusionLine.append({'Path':path,'Intensity':I,'YCoordinate':Y})
        #     return (array(Y),array(I),array(IP))
        #
        # else:
        #
        #     return ([],[],[])

        try:

            for IM in Delays:

                I.append(GetAmplitude((IM[0],IM[1],IM[3]))*IM[4])

                IP.append(sign(IM[4])*GetInnerProduct(IM[0],IM[1],IM[3]))


                Y.append(IM[2])

            # self.FusionLine.append({'Path':path,'Intensity':I,'YCoordinate':Y})
            return (array(Y),array(I),array(IP))

        except:

            return ([],[],[])


class AnalyzeWeld:

    def __init__(self,metrics):


        d = pickle.load(open(metrics,'rb'))

        self.Thickness = []
        self.SideWallPosition = []
        self.Intensities = []
        self.InnerProducts = []
        self.YCoordinates = []
        self.NumberOfFrames = len(d)

        for dd in d:

            self.Thickness.append(dd['Thickness'])
            self.SideWallPosition.append(dd['SideWallPosition'])
            self.YCoordinates.append(dd['YCoordinates'])
            self.Intensities.append(dd['Intensities'])
            self.InnerProducts.append(dd['InnerProducts'])

    def LoadDisbondLengths(self,fl,trimedframes=(5,5)):

        d = loadtxt(fl,delimiter=',')

        # plot(d)
        # show()

        ds = int(round(len(d)/(self.NumberOfFrames+trimedframes[0]+trimedframes[1])))

        d = d[0::ds]

        self.DisbondLengths = d[trimedframes[0]:trimedframes[0]+self.NumberOfFrames]

    def StackMetrics(self):


        Is = []
        ys = []
        IP = []

        for i in range(self.NumberOfFrames):

            Is.append(reduce(lambda x,y: concatenate((x,y)),[abs(array(self.Intensities[i][j][k])) for j in range(len(self.Intensities[i])) for k in range(len(self.Intensities[i][0]))]))
            IP.append(reduce(lambda x,y: concatenate((x,y)),[abs(array(self.InnerProducts[i][j][k])) for j in range(len(self.InnerProducts[i])) for k in range(len(self.InnerProducts[i][0]))]))
            ys.append(reduce(lambda x,y: concatenate((x,y)),[abs(array(self.YCoordinates[i][j][k])) for j in range(len(self.YCoordinates[i])) for k in range(len(self.YCoordinates[i][0]))]))


        self.StackedMetrics = {'Intensities':(Is,ys), 'InnerProducts':(IP,ys)}


    def GetMaxMetrics(self):

        from numpy import mean


        IP = []
        I = []
        YI = []
        YIP = []

        Iind = []
        IPind = []

        for i in range(len(self.InnerProducts)):

            ip = [reduce(lambda x,y: concatenate((x,y)),[abs(array(self.InnerProducts[i][j][k])) for k in range(len(self.InnerProducts[i][j]))]) for j in range(len(self.InnerProducts[i]))]

            y = [reduce(lambda x,y: concatenate((x,y)),[abs(array(self.YCoordinates[i][j][k]))  for k in range(len(self.YCoordinates[i][j]))]) for j in range(len(self.YCoordinates[i]))]
            II = [reduce(lambda x,y: concatenate((x,y)),[abs(array(self.Intensities[i][j][k]))  for k in range(len(self.Intensities[i][j]))]) for j in range(len(self.Intensities[i]))]

            ipmean = [mean(ii) if len(ii)>0 else 0. for ii in ip]

            imean = [mean(ii) if len(ii)>0 else 0. for ii in II]

            ipmaxind = argmax(ipmean)

            imaxind = argmax(imean)

            IP.append(array(ip[ipmaxind]))
            I.append(array(II[imaxind]))

            Iind.append(imaxind)
            IPind.append(ipmaxind)


            YI.append(array(y[imaxind]))
            YIP.append(array(y[ipmaxind]))

        self.MaxMetrics = {'InnerProducts':(IP,YIP,IPind),'Intensities':(I,YI,Iind)}


    def GetMoments(self,metrickey,thresh=0.):

        def ComputeMoments(x,f):

            ff = f[f>=thresh]
            xx = x[f>=thresh]

            m = []

            m.append(sum(ff))
            m.append(sum(xx*(ff/m[0])))
            m.append(sqrt(sum((ff/m[0])*xx**2)))
            m.append(mean(ff))
            m.append(std(ff))
            m.append(max(ff))

            return m

        M = eval('self.'+metrickey[0])

        V = M[metrickey[1]][0]
        Y = M[metrickey[1]][1]

        M = array([ComputeMoments(Y[i],V[i]) for i in range(self.NumberOfFrames)])


        return M
