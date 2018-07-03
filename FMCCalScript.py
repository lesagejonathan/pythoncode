import numpy as np
import _pickle as pickle
import sys
import FMC
from multiprocessing import Pool
import FieldSimulation as fs

cal=sys.argv[1]


s = pickle.load(open('/mnt/e/FMCCalibration/SDHBlock-5L64-A2.p','rb'))
d = pickle.load(open('/mnt/e/FMCCalibration/A2-5L-64-ProbeDelays.p','rb'))
a = pickle.load(open('/mnt/e/FMCCalibration/StepBlockCorrections.p','rb'))


f = FMC.LinearCapture(25.,s['AScans'],0.6,64,probedelays=d[0])

f.ProcessScans(70,bp=15)

f.GetContactDelays(np.arange(-5.,42.,0.1),np.arange(10.,60.,0.1), 5.91)

if cal=='Calibrated':

    f.GetContactCorrections(a['x'],a['y'],a['Amplitude'],a['Sensitivity'],isongrid=False)

def GetImage(i):

    return f.ApplyTFM(i,stablecoeff=1e-4)


p = Pool(15)
I = p.map(GetImage,range(len(s['AScans'])))

D = {'Images':I, 'x': f.xRange, 'y': f.yRange}

pickle.dump(D,open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','wb'))





# f = FMC.LinearCapture(25.,s['AScans'],0.6,64,probedelays=d[0])
#
# f.ProcessScans(70,bp=15)
#
# f.GetContactDelays(np.arange(0.,38.,0.1),np.arange(10.,65.,0.1), 5.91)
#
# if cal=='Calibrated':
#
#     f.GetContactCorrections(a['x'],a['y'],a['Amplitude'],a['Sensitivity'],isongrid=False)
#
# I = [f.ApplyTFM(i,stablecoeff=1e-4) for i in range(len(f.AScans))]
#
# D = {'Images':I, 'x': f.xRange, 'y': f.yRange}
#
# pickle.dump(D,open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','wb'))
