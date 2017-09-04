import FMC
import pickle
from numpy import *
import sys
from scipy.misc import imrotate
import multiprocessing

def AdaptiveImage(x):

    a = pickle.load(open('/mnt/d/FMCScans/L-FBH-CapScans/5L-32-A11/'+x+'.p','rb'))

    F = FMC.LinearCapture(25.,a['AScans'],0.6,32,d)

    F.ProcessScans(100)

    F.KeepElements(range(16))


    Ic = []

    Is = []

    H = []

    Dc = []

    Ds = []

    for i in range(len(F.AScans)):

        h = F.FitInterfaceLine(i,(-30.,0.,0.5),(5.,15.),cw)

        F.GetAdaptiveDelays(linspace(0,16*0.6,100), linspace(h(0),h(0)+9.6,100), h, cw, 3.24)

        Is.append(abs(F.ApplyTFM(i,['Band',0.,10.,cw])))

        Ds.append(F.Delays)

        F.GetAdaptiveDelays(linspace(0,16*0.6,100), linspace(h(0),h(0)+9.6,100), h, cw, 5.92)

        Dc.append(F.Delays)

        Ic.append(abs(F.ApplyTFM(i,['Band',-10.,10.,cw])))

        H.append(h)

    pickle.dump({'ShearImages':Is, 'CompressionImages':Ic, 'Heights':H, 'ShearDelays': Ds, 'CompressionDelays': Dc, 'Resolution':0.096},open('/mnt/d/FMCScans/L-FBH-CapScans/5L-32-A11/OverCapImages-'+x+'.p','wb'))

    return 0

fl = sys.argv[1::]
d = pickle.load(open('/mnt/d/FMCScans/ReferenceScans/MP-5L32-A11-Delays.p','rb'))

cw = 1.49

p = multiprocessing.Pool(len(fl))

r = p.map(AdaptiveImage, fl)
