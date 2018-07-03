import numpy as np
import _pickle as pickle
import sys
import FMC
from multiprocessing import Pool

cal=sys.argv[1]


s = pickle.load(open('/mnt/e/FMCCalibration/SDH-5L64-A2.p','rb'))
d = pickle.load(open('/mnt/e/FMCCalibration/A2-5L-64-ProbeDelays.p','rb'))
a = pickle.load(open('/mnt/e/FMCCalibration/StepBlockCorrections.p','rb'))


f = FMC.LinearCapture(25.,s['AScans'][0:5],0.6,64,probedelays=d[0])

f.ProcessScans(70,bp=15)

f.GetContactDelays(np.arange(-5.,42.,0.1),np.arange(10.,30.,0.1), 5.91)

if cal=='Calibrated':

    f.GetContactCorrections(a['x'],a['y'],a['Amplitude'],a['Sensitivity'],isongrid=False)


I = [f.ApplyTFM(i,stablecoeff=1e-4) for i in range(5)]

II = []
X = []
Y = []

X.append(f.xRange)
Y.append(f.yRange)

II.append(I)

D = {'Images':II, 'x': X, 'y': Y}

pickle.dump(D,open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','wb'))



f = FMC.LinearCapture(25.,s['AScans'][5:10],0.6,64,probedelays=d[0])

f.ProcessScans(70,bp=15)

f.GetContactDelays(np.arange(-5.,42.,0.1),np.arange(25.,45.,0.1), 5.91)

if cal=='Calibrated':

    f.GetContactCorrections(a['x'],a['y'],a['Amplitude'],a['Sensitivity'],isongrid=False)


I = [f.ApplyTFM(i,stablecoeff=1e-4) for i in range(5)]

D = pickle.load(open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','rb'))

D['Images'].append(I)
D['x'].append(f.xRange)
D['y'].append(f.yRange)

pickle.dump(D,open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','wb'))


f = FMC.LinearCapture(25.,s['AScans'][10:15],0.6,64,probedelays=d[0])

f.ProcessScans(70,bp=15)

f.GetContactDelays(np.arange(-5.,42.,0.1),np.arange(45.,65.,0.1), 5.91)

if cal=='Calibrated':

    f.GetContactCorrections(a['x'],a['y'],a['Amplitude'],a['Sensitivity'],isongrid=False)


I = [f.ApplyTFM(i,stablecoeff=1e-4) for i in range(5)]

D = pickle.load(open('/mnt/e/FMCCalibration/SDHblock'+cal+'.p','rb'))

D['Images'].append(I)
D['x'].append(f.xRange)
D['y'].append(f.yRange)

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
