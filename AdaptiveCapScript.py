import FMC
import pickle
from numpy import *
import sys

fl = sys.argv[1]


a = pickle.load(open('/mnt/d/FMCScans/L-FBH-CapScans/5L-32-A11/'+fl+'.p','rb'))

d = pickle.load(open('/mnt/d/FMCScans/ReferenceScans/MP-5L32-A11-Delays.p','rb'))

F = FMC.LinearCapture(25.,a['AScans'],0.6,32,d)

F.ProcessScans(100)

cw = 1.49

I = []

for i in range(len(F.AScans)):

    print(i)

    F.KeepElements(range(16))

    h = F.FitInterfaceLine(i,(-30.,0.,0.5),(5.,15.),cw)

    F.GetAdaptiveDelays(i, linspace(0,16*0.6,150), linspace(h(0),h(0)+9.6,150), cw, 3.24, AsParallel=True)

    I.append(abs(F.ApplyTFM(i,['Band',0.,30.,cw], AsParallel=True)))

pickle.dump(I,open('/mnt/d/FMCScans/L-FBH-CapScans/5L-32-A11/OverCapImages-'+fl+'.p','wb'))
