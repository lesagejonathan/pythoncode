from numpy import *
from matplotlib.pylab import *


# L = 27.
# v = 6.
# h = 3.
N = 32
p = 0.6
H = 3.7

c0 = 3.24
c1 = 5.9

cw = 2.33

def CapApertureOverlap(xa,lc):

    # lc = sqrt(v**2+h**2)/2

    # cv = (xa-lc>0.) and (xa+lc<N*p)

    Xa = [xa-lc,xa+lc]


    if ((Xa[1]<0.) or (Xa[0]>N*p)):

        la = 0.

    else:

        if Xa[0]<0:

            Xa[0] = 0.

        if Xa[1]>N*p:

            Xa[1] = N*p

        la = 100*(Xa[1] - Xa[0])/(2*lc)

    return la



def SideCap(L,v,h,l,tw):

    tw = tw*(pi/180.)

    tn = arcsin((c0/cw)*sin(tw))


    tc = arctan2(h,v)
    ts = arcsin((c0/c1)*cos(tc))

    xc = l - 0.5*v
    yc = 0.5*h + L

    ys = yc - ((l-xc)/tan(tc))
    xe = (l*tan(ts) - ys)/tan(ts)

    te = arcsin((cw/c0)*cos(ts))
    xa = (xe - H*tan(te))/(cos(tw) + tan(te)*sin(tw))


    return CapApertureOverlap(xa,sqrt(v**2 + h**2)/2)


    # td = 100*((pi/2 - ts) - tn)/tn

def CapSide(L,v,h,l,tw):

    tw = tw*(pi/180.)

    tc = arctan2(h,v)

    t1 = arcsin((c0/c1)*cos(tc)) - tc

    xe = (l-0.5*v) - (L+0.5*h)*tan(t1)

    t0 = arcsin((cw/c0)*sin(t1))

    xa = (xe - tan(t0)*H)/(cos(tw) + tan(t0)*sin(tw))


    return CapApertureOverlap(xa,sqrt(v**2 + h**2)/2)


# vhgrid = [(v,h) for v in linspace(5.,8.,6) for h in linspace(3.,8.,10) if h<=v]

lgrid = linspace(0.,100.,1000)


#
# twgrid = linspace(25.,60.,35)
#
# vgrid = linspace(5.,8.,100)
# hgrid = linspace(3.,8.,100)

# f = array([[OffsetFunction(30.,v,h,43.,31.5) for v in vgrid] for h in hgrid])


f = array([CapSide(30.,7.,4.,l,31.5) for l in lgrid])
# f = array([[CapSide(12.,v,h,15.,31.5) for v in vgrid] for h in hgrid])


# f = array([[[OffsetFunction(30.,vh[0],vh[1],l,tw) for vh in vhgrid] for l in lgrid] for tw in twgrid])

# f = array([OffsetFunction(30.,7.,4.,l,31.5) for l in lgrid])

# print(lgrid[argmax(f)])



# l = linspace(0,100,1000)
#
#
# f = [OffsetFunction(ll) for ll in l]
#
# print(l[argmin(array(f))])
#
# plot(l,array(f))
#
# show()
