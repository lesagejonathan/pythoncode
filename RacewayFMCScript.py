import numpy as np
from scipy.optimize import minimize
import FMC
import _pickle
import multiprocessing
import argv

pth = '/mnt/d/FMCScans/Raceway/'

pd = _pickle.load('/mnt/d/FMCScans/ReferenceScans/4M-16x2-A27-Delays.p','rb')

p = (1.0,3.0)

cw = 2.33
cs = 5.92

W = 29.

L = 30.
R = 32.

phi = 12.*np.pi/180.

w = 5.7/np.cos(phi)

H = 32.8524 - 14./np.cos(phi)

f = lambda x: - R + np.sqrt(R**2 - x**2)

def Delay(x,e,Y,Z):

    return np.sqrt((e[0] - x[0])**2 + (e[1] - f(x[0]))**2 + (e[2]-x[1])**2)/cw \
            + np.sqrt( x[0]**2 + (Y - f(x[0]))**2 + (Z - x[1])**2)/cs


def ElementCoords(n, numbering):

    if numbering == 'Even':

        return (W/2 - w*np.cos(phi), H + w*np.sin(phi), n*p[0])

    elif numbering == 'Odd':

        return (W/2 - (w + p[1])*np.cos(phi), H + (w + p[1])*np.sin(phi), n*p[0])


eeven = [ElementCoords(n,'Even') for n in range(16)]
eodd = [ElementCoords(n,'Odd') for n in range(16)]

yrng = np.linspace(f(W/2),10.,round((abs(f(W/2)+10.)/0.096)))
zrng = np.linspace(0.,p[0]*16,100)

dEven = [[[minimize(Delay,x0=(e[0]/2,0.5*(e[2]+z)), args=(e,y,z) , method='BFGS') for y in yrng] for z in zrng] for e in eeven]
dOdd = [[[minimize(Delay,x0=(e[0]/2,0.5*(e[2]+z)), args=(e,y,z) , method='BFGS') for y in yrng] for z in zrng] for e in eodd]


Delays =  []

for n in range(16):

    Delays.append(dEven[n])
    Delays.append(dOdd[n])

def ProcessScans(x):

    a = _pickle.load(open(pth+x,'rb'))

    F = FMC.LinearCapture(25.,a['AScans'],0.6,16,probedelays=pd)

    F.ProcessScans(200)

    F.Delays = Delays

    I = [ F.ApplyTFM(i) for i in range(len(F.AScans)) ]

    _pickle.dump({'Images': I, 'xGrid': zrng, 'yGrid': yrng} , open(pth+x+'-Images.p','wb'))

x = argv[1::]

p = multiprocessing.pool(len(x))

I = p.map(ProcessScans,x)