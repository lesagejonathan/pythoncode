import MicroPulse as mp
import MotionController as mc
import sys
import numpy as np
import FMC
import copy
import matplotlib.pylab as plt



scanpth = '/mnt/c/Users/jlesage/Documents/ANSFeederTubeProject/'

samplename = sys.argv[1]
diameter = float(sys.argv[2])*25.4

scanspeed = 1.0


scanres = 0.5

circ = np.pi*diameter

NScan = int(np.round(circ/scanres))

# drot = (2.*scanres/diameter)*(180./np.pi)

dt = scanres/scanspeed

dtmin = 32*32/20000.

els = list(range(1,17)) + list(range(65,65+17))

if dt<dtmin:

    print("Scan Speed Must be Less Than "+str(drot/dtmin))
    sys.exit()

p = mp.MicroPulse(fsamp=25.)


p.SetFMCCapture((els,els), Gate = (5., 50.), Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1))

g = mc.MotionController(Instrument = 'ZMC4')

g.MoveAbsolute('Rotation', scanspeed , circ)

p.ExecuteCapture(NScan, dt)

p.ReadBuffer()

pos = np.linspace(0.,circ,NScan)


F = FMC.LinearCapture(25., p.AScans, 32, 0.6)

# F.ProcessScans(T0 = p.PulserSettings['Gate'][0])
#
# Acopy = copy.deepcopy(F.AScans)

F.KeepElements(range(16))

I0 = np.abs(np.array([F.PlaneWaveSweep(i, [-39.], 2.33) for i in range(len(p.AScans))]).transpose())

# F.AScans = Acopy
#
# del(Acopy)

F = FMC.LinearCapture(25., p.AScans, 32, 0.6)


F.KeepElements(range(17,33))

I1 = np.abs(np.array([F.PlaneWaveSweep(i, [-39.], 2.33) for i in range(len(p.AScans))]).transpose())


fig, ax = plt.subplots(nrows=2)

ax[0].imshow(I0)

ax[1].imshow(I1)

plt.show()

yn = input("Scan Acceptable (y/n)")

if yn =='y':

    p.SaveScans(scanpth+samplename+'.p', {'ScanPositions': pos})

del(p)
del(g)
