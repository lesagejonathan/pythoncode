import FMC
import pickle
from numpy import *


a = pickle.load(open('/mnt/d/FMCScans/L-FBH-CapScans/10L-32-A1/LA1.p','rb'))

d = pickle.load(open('/mnt/d/FMCScans/ReferenceScans/MP-10L32-A1-Delays.p','rb'))

F = FMC.LinearCapture(50.,a['AScans'],0.3,5.92,32,d)

F.ProcessScans()

I = []

for i in range(len(F.AScans)):

    print(i)

    F.GetAdaptiveDelays(1.49,linspace(5.,10.,50),i,(6.5,0.2))

    I.append(abs(F.ApplyTFM(i)))

pickle.dump(I,open('/mnt/d/FMCScans/L-FBH-CapScans/10L-32-A1/CapImages-LA.p','rb'))
