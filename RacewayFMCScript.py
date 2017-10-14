import numpy as np
from scipy.optimize import minimize
import FMC
import _pickle
import multiprocessing
import sys

pth = '/mnt/d/FMCScans/Raceway/12DegRoofPC/'

pd = 0.6*np.ones((32,32))

p = (1.0,3.0)

cw = 2.33
cs = 5.92

W = 30.


l = 8.8

R = 32.

phi = 12.*np.pi/180.

w = 5.7/np.cos(phi)

H = 32.8524

f = lambda x: -R + np.sqrt(R**2 - x**2)

# def Delay(x,e,Y,Z):
#
#     return np.sqrt((e[0] - x[0])**2 + (e[1] - f(x[0]))**2 + (e[2]-x[1])**2)/cw \
#             + np.sqrt( x[0]**2 + (Y - f(x[0]))**2 + (Z - x[1])**2)/cs
#
#
# def ElementCoords(n, numbering):
#
#     if numbering == 'Even':
#
#         return (W/2 - w*np.cos(phi), H + w*np.sin(phi), n*p[0])
#
#     elif numbering == 'Odd':
#
#         return (W/2 - (w + p[1])*np.cos(phi), H + (w + p[1])*np.sin(phi), n*p[0])
#

def Delay(X,e,x,y):

    return np.sqrt((X[0]-e[0])**2 + (f(X[0]) - e[1])**2 + (X[1]-e[2])**2)/cw \
           + np.sqrt((x-X[0])**2 + (y - f(X[0]))**2 + X[1]**2)/cs

def ElementCoords(n, numbering):

    if numbering == 'Even':

        return (n*p[0] - 15*p[0]*0.5, H - (l*np.tan(phi) + 0.5*p[1]*np.sin(phi)) \
                , l + 0.5*p[1]*np.cos(phi))

    elif numbering == 'Odd':

        return (n*p[0] - 15*p[0]*0.5, H - (l*np.tan(phi) - 0.5*p[1]*np.sin(phi)) \
                , l - 0.5*p[1]*np.cos(phi))


eeven = [ElementCoords(n,'Even') for n in range(16)]
eodd = [ElementCoords(n,'Odd') for n in range(16)]

xrng = np.linspace(-7.5,7.5,150)
yrng = np.linspace(f(7.5),10.-f(7.5), int(round((10.-f(7.5))/0.1)))


dEven = [[[minimize(Delay,x0=(0.5*(e[0]+x), 0.5*e[2]), args=(e,x,y) , method='BFGS').fun if y>=f(x) else np.nan for x in xrng] for y in yrng] for e in eeven]
dOdd = [[[minimize(Delay,x0=(0.5*(e[0]+x), 0.5*e[2]), args=(e,x,y) , method='BFGS').fun if y>=f(x) else np.nan for x in xrng] for y in yrng] for e in eodd]


Delays =  []

for n in range(16):

    Delays.append(dEven[n])
    Delays.append(dOdd[n])

# print('Delays computed')
#
# print(Delays)

def ProcessScans(x):

    a = _pickle.load(open(pth+x+'.p','rb'))

    F = FMC.LinearCapture(25.,a['AScans'],p[0],32,probedelays=pd)

    F.ProcessScans(200)

    F.Delays = Delays

    I = [ F.ApplyTFM(i) for i in range(len(F.AScans)) ]

    _pickle.dump({'Images': I, 'xGrid': xrng, 'yGrid': yrng} , open(pth+x+'-Images.p','wb'))

x = sys.argv[1::]

p = multiprocessing.Pool(len(x))

I = p.map(ProcessScans,x)
