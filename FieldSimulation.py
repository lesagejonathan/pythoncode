
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import griddata
import time
import pickle
from functools import reduce

from matplotlib import animation

def PointFocalLaw(f,F,p,N,x,y,c):

    xn = np.linspace(-0.5*(N-1)*p,0.5*p*(N-1),N)

    Fn = [F*np.exp(-2j*f*np.pi*np.sqrt((x-xn[n])**2 + y**2)/c) for n in range(N)]

    return Fn

def SweepFocalLaw(f,F,p,N,angle,c):

    xn = np.linspace(-0.5*(N-1)*p,0.5*p*(N-1),N)

    Fn = [F*np.exp(2j*f*np.pi*np.sin(angle*np.pi/180.)*xn[n]/c) for n in range(N)]

    return Fn

def GetCenterline(x,y,I,angle,dr=0.1):

    x,y = np.meshgrid(x,y)

    r = np.arange(0,np.amax(y)/np.cos(angle*np.pi/180.),dr)

    X = r*np.sin(angle*np.pi/180.)
    Y = r*np.cos(angle*np.pi/180.)

    s = griddata((x.flatten(),y.flatten()), I.flatten(), (X.flatten(),Y.flatten()), method='linear')

    return s

def ToPolarField(x,y,I,res=(0.1,5),isongrid=True):

    if isongrid:

        x,y = np.meshgrid(x,y)

        x = x.flatten()
        y = y.flatten()


    r = np.sqrt(x**2+y**2)

    th = np.arctan2(x,y)*180./np.pi

    rr,tth = np.arange(np.amin(r),np.amax(r),res[0]), np.arange(np.amin(th),np.amax(th),res[1])

    R,Th = np.meshgrid(rr,tth)

    s = R.shape

    R = R.flatten()
    Th = Th.flatten()

    II = griddata((r,th), (I.flatten()), (R,Th), fill_value=0.).reshape(s)

    return np.arange(0.,np.amax(r),res[0]), np.arange(np.amin(th),np.amax(th),res[1]), II


def GetScatteringDistribution(x,y,scatterparams,c=5.92):

    if scatterparams['Type']=='SDH':

        xc,yc = scatterparams['Center']
        R = scatterparams['Radius']

        x,y=np.meshgrid(x,y)

        I = ((y-yc)**2)<=(R**2 - (x-xc)**2)

        I = (I.astype(np.float).reshape(x.shape))*(1/(0.33)**2 - 1/c**2)


    return I


def GetNormalline(x,y,I,angle,r,dr=0.1):

    x,y = np.meshgrid(x,y)

    x0 = r*np.sin(angle*np.pi/180.)
    y0 = r*np.cos(angle*np.pi/180.)

    # y0 = np.amax(y)
    #
    # x0 = y0*np.tan(angle*np.pi/180.)
    #

    xx = (np.amax(y) - y0)/np.cos(angle*np.pi/180.)
    yy = (np.amax(x) - x0)/np.sin(angle*np.pi/180.)

    #
    # yy = (xx - x0)/np.tan(np.pi/2 - angle*np.pi/180.)
    #
    # print(yy)

    # r = np.arange(0.,np.sqrt((xx-x0)**2 + yy**2),dr)

    r = np.arange(-np.sqrt((xx-x0)**2 + (np.amax(y)-y0)**2),np.sqrt((np.amax(x)-x0)**2 + (np.amax(yy)-y0)**2),dr)

    X = r*np.sin(np.pi/2 - angle*np.pi/180.) + x0
    Y = r*np.cos(np.pi/2 - angle*np.pi/180.) + y0

    s = griddata((x.flatten(),y.flatten()), I.flatten(), (X.flatten(),Y.flatten()), method='linear')

    return r,s


def CompressionElementField3d(L,W,f,F,resolution,Lx,Lz,rho=7.8,cp=5.92,cs=3.24,Nkz = 10,bc='displacement',eta=1e-4):


    # cp = cp*(1-eta*1j)


    dx = resolution
    Nx = int(Lx/dx)
    Nz = int(Lz/dx)


    kx,ky,z,w = np.meshgrid(2*np.pi*np.linspace(-1/L,1/L,Nkz), 2*np.pi*np.linspace(-1/(2*dx),1/(2*dx),Nx),np.linspace(0,Lz,Nz), 2*np.pi*f)


    F = F.reshape((1,1,1,w.shape[-1]))


    kz = np.sqrt((w/cp)**2 - kx**2 - ky**2 + 0j)

    if bc == 'stress':


        P = L*W*F*np.sinc(0.5*L*ky/np.pi)*np.sinc(0.5*W*kx/np.pi)*np.exp(1j*kz*z)/(rho*(w**2 - 2*(kx**2 + ky**2)*cs**2))

        P = P/np.amax(np.abs(P))

        n = P.shape

        P[np.abs(np.imag(kz))>0.] = 0.+0j
        P[np.abs((w**2 - 2*(kx**2 + ky**2)*cs**2))<eta] = 0.+0j

    elif bc == 'displacement':

        P = L*W*F*np.sinc(0.5*L*ky/np.pi)*np.sinc(0.5*W*kx/np.pi)*np.exp(1j*kz*z)/(1j*kz)

        P = P/np.amax(np.abs(P))

        n = P.shape

        P[np.abs(np.imag(kz))>0.] = 0.+0j
        P[np.abs(kz)<eta] = 0.+0j

    elif bc == 'potential':

        P = L*W*F*np.sinc(0.5*L*ky/np.pi)*np.sinc(0.5*W*kx/np.pi)*np.exp(1j*kz*z)

        P = P/np.amax(np.abs(P))

        n = P.shape

        P[np.abs(np.imag(kz))>0.] = 0.+0j



    P = P.reshape(n)


# Field averaged over passive aperture

    p = (np.fft.fftshift(np.sum(np.abs(np.sum(np.fft.ifft(P,axis=0),axis=1)),axis=2),axes=(0))).transpose()


    return p

def ContactArray2d(f,F,p,resolution,Lx,Ly,rho=7.8,cL=5.91,cT=3.24,bc='stress',eta=1e-4,output='averagefield'):

    N = len(F)
    Nx = int(Lx/resolution)

    kx,y,w = np.meshgrid(2*np.pi*np.linspace(-1/(2*resolution),1/(2*resolution),Nx)+0j,np.arange(0.,Ly,resolution)+0j,2*np.pi*f + 0j)

    xn = np.linspace(-(N-1)*p/2, (N-1)*p/2, N) + 0j
    ky = np.sqrt((w/cL)**2 - kx**2)

    A = reduce(lambda x,y: x+y, (F[n].reshape((1,1,len(f)))*np.exp(-1j*kx*xn[n]) for n in range(len(F))))

    # A = F[8].reshape((1,1,len(f)))


    if bc == 'stress':

        A = (p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)/(rho*(2*(cT*kx)**2 - w**2))

        # A = np.sqrt((rho*(2*(cT/cL)**2 - 1)*w**2 - 2*cT**2*kx**2)**2 + 1)*(p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)

        # A = (p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)

        # A = (p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)/(rho*ky**2)

        A = A/np.amax(np.abs(A))


        s = A.shape

        # A[np.abs(np.imag(ky))>0.] = 0. + 0j

        A[np.abs(kx**2 - 0.5*(w/cT)**2)<eta] = np.nan




        # A[np.abs(2*(cT*kx)**2 - w**2)<eta] = 0. +0j



    # elif bc == 'potential':
    #
    #     A = (p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)
    #
    #     s = A.shape
    #
    #     A[np.abs(np.imag(ky))>0.] = 0. + 0j
    #
    #
    # # elif bc == 'stressdisplacement':
    # #
    # #     A = (p+0j)*A*np.sinc(0.5*p*kx/(np.pi+0j))*np.exp(1j*ky*y)/((1j*ky)*(rho*(2*(cT*kx)**2 - w**2)))
    # #
    # #     s = A.shape
    # #
    # #     A[np.abs(np.imag(ky))>0.] = 0. + 0j
    # #
    # #     A[np.abs(ky)<0.] = 0. +0j
    # #
    # #     A[np.abs(rho*(2*(cT*kx)**2 - w**2))<0.] = 0. +0j
    #
    #
    # A = A.reshape(s)




    np.nan_to_num(A,False)

    if output == 'averagefield':

        return np.fft.fftshift(np.sum(np.abs(np.fft.ifft(A,axis=1)),axis=2),axes = (1))

    elif output == 'timefield':

        return np.fft.fftshift(np.fft.fft(np.fft.ifft(A,axis=1),axis=2,n=int(2*len(f)+1)),axes = (1))

    elif output == 'frequencyfield':

        return np.fft.fftshift(np.fft.ifft(A,axis=1),axes = (1))


def BornScatteredFieldContact(f,A,I,resolution,Lx,Ly,c=5.92):

    from numpy.fft import ifft, fftn, ifftn, fft

    """ Takes incident field, A, and computes field scattered by scattering
    distribution I. Incident field is assumed to be generated by a contact array
    and should be computed over the same domain as is requested here.
    f - frequency vector, resolution spacing of x and y points, Lx is length of x domain (y E [-Lx/2, Lx/2])
    Ly is length of y domain, y E [-Ly/2, Ly/2]. Returns time domain scattered field (analytic signal thereof)"""

    # w = (2*np.pi*f).reshape((1,1,len(f)))


    Nx = int(np.round(Lx/resolution))

    Ny = int(np.round(Ly/resolution))

    Zx = np.zeros((Ny,int(Nx/2),len(f)))

    Zy = np.zeros((int(Ny/2),2*Nx,len(f)))

    x,y,w = np.meshgrid(np.linspace(-Lx*resolution,Lx*resolution,2*Nx), np.linspace(-Ly*resolution, Ly*resolution, 2*Ny), 2*np.pi*f)

    r = np.sqrt(x**2 + y**2).reshape((x.shape[0],x.shape[1],len(f)))

    g = np.exp(-1j*w*r/c)/r

    G = fftn(g,axes=(0,1))

    # I =

    I = fftn(np.vstack((Zy[:,:,0],np.hstack((Zx[:,:,0],I,Zx[:,:,0])),Zy[:,:,0])),axes=(0,1))

    S = -I*(w**2)

    A = fftn(np.vstack((Zy,np.hstack((Zx,A,Zx)),Zy)),axes=(0,1))


    P = S*A*G

    p = ifftn(P,axes=(0,1))

    p = fft(p[int(np.round(0.5*Ny)):int(np.round(1.5*Ny)),int(np.round(0.5*Nx)):int(np.round(1.5*Ny)),:],n=2*len(f) - 1,axis=2)

    return p


















# def ContactArray(f,F,p,l,resolution,Lx,Ly,rho=7.8,cL=5.91,cT=3.24,output='averagefield'):
#
#     N = len(F)
#     # Nx = int(Lx/resolution)
#
#     x,y,w = np.meshgrid(np.arange(-Lx/2,Lx/2,resolution),np.arange(0.,Ly,resolution),2*np.pi*f)
#
#     xn = np.linspace(-(N-1)*p/2, (N-1)*p/2, N)
#
#
#     kx = w/(np.sqrt(2)*cT)
#
#     # kx = w/cL
#
#     A1 = reduce(lambda x,y: x+y, (F[n].reshape((1,1,len(f)))*np.exp(1j*xn[n]*kx) for n in range(len(F))))
#
#     A2 = reduce(lambda x,y: x+y, (F[n].reshape((1,1,len(f)))*np.exp(-1j*xn[n]*kx) for n in range(len(F))))
#
#     A = (np.pi/(4*rho*w))*1j*(A1*np.exp(-1j*kx*x)*np.sinc(-kx*0.5*l/(np.pi)) + A2*np.exp(1j*kx*x)*np.sinc(kx*0.5*l/(np.pi)))*np.exp(1j*np.sqrt(1/cL**2 - 1/(2*cT**2) + 0j)*y*w)
#
#
#
#
#     if output == 'averagefield':
#
#         return np.sum(np.abs(A),axis=2)
#
#
#     elif output == 'frequencyfield':
#
#         return A
