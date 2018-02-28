import MicroPulse as mp
import MotionController as mc
import sys
import numpy as np
import FMC
import copy
import matplotlib.pylab as plt



scanpth = '/mnt/c/Users/jlesage/Documents/ANSFeederTubeProject/'

samplename = sys.argv[1]
# diameter = float(sys.argv[2])*25.4
circumference = float(sys.argv[2])


index = sys.argv[3]

if index=='A':

    info = {'Circumference':circumference, 'IndexOffset': 'Index offset of 90 degrees skew covers HAZ'}

elif index=='B':

    info = {'Circumference':circumference, 'IndexOffset': 'Index offset of 270 degrees skew covers HAZ'}

else:

    print('Index must be A or B')
    sys.exit()


resolution = 0.5
scanspeed = 4.0

els = list(range(1,17))[-1::] + list(range(65,65+17))[-1::]

NScan = int(np.round((circumference + 5.)/0.5))

dt = resolution/scanspeed


p = mp.PeakNDT(fsamp=25.)

p.SetFMCCapture((els,els), Gate = (0., 75.), Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1))

ss = input('Press any key to start capturing')

p.ExecuteCapture(NScan, dt)
p.ReadBuffer()

pos = np.linspace(0.,circumference,NScan)

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

F.KeepElements(range(16,33))

I1 = np.abs(np.array([F.PlaneWaveSweep(i, [-39.], 2.33) for i in range(len(p.AScans))]).transpose())

fig, ax = plt.subplots(nrows=2)

ax[0].imshow(I0[int(5*25.)::,:])

ax[1].imshow(I1[int(5*25.)::,:])

plt.show()

yn = input("Scan Acceptable ? (y for yes, any key for no)")

if yn=='y':

    p.SaveScans(scanpth+samplename+index+'.p',info)

    del(p)

else:

    del(p)
    sys.exit()
