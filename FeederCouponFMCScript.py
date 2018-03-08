import MicroPulse as mp
# import MotionController as mc
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

# els = list(range(1,17))[-1::] + list(range(65,65+17))[-1::]

els = list(range(1,17))+ list(range(65,65+16))

NScan = int(np.round((circumference/resolution)))

dt = resolution/scanspeed

p = mp.PeakNDT(fsamp=25.)

p.SetFMCCapture((els,els), Gate = (0., 50.), Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1))

pc = input('Ensure Probes Connected to MicroPulse\n' + 'Ensure Start Position is Correct \n' + 'Ensure Couplant is Flowing \n' + '... Press Enter to Continue')

ss = input('Press Enter to Start Scanning')

print('Scan in Progress ... ')

p.ExecuteCapture(NScan, dt)

print('Finished Scan, Reading Data ... OK to Change Setup')
p.ReadBuffer()

gate = (int(25*6), int(25*15))

p.AScans = [p.AScans[i][::-1,::-1,:] for i in range(len(p.AScans))]

A = [p.AScans[i][0:15,0:15,gate[0]:gate[1]] for i in range(len(p.AScans))]

F = FMC.LinearCapture(25.,A,0.5,16)

I0 = F.PlaneWaveSweep(0,np.array([-39.]),2.33)

I0 = np.abs(np.array(I0)[:,0,:]).transpose()

del(A)
del(F)

A = [p.AScans[i][16::,16::,gate[0]:gate[1]] for i in range(len(p.AScans))]

F = FMC.LinearCapture(25.,A,0.5,16)

I1 = F.PlaneWaveSweep(0,np.array([-39.]),2.33)
I1 = np.abs(np.array(I1)[:,0,:]).transpose()

ax,fig = plt.subplots(nrows=2)

ax[0].imshow(I0)
ax[1].imshow(I1)

plt.show()


yn = input('Save Scan? Enter (y/n)')

if yn=='y':

    print('Saving Data ...')
    p.SaveScans(scanpth+samplename+'_'+index+'.p',info)

    print('Finished Saving '+scanpth+samplename+'_'+index+'.p')

elif yn=='n':
    print('Scan not Saved')
del(p)


# pos = np.linspace(0.,circumference,NScan)
#
# F = FMC.LinearCapture(25., p.AScans, 0.5, 32)
#
# # F.ProcessScans(T0 = p.PulserSettings['Gate'][0])
# #
# # Acopy = copy.deepcopy(F.AScans)
#
# F.KeepElements(range(16))
#
# I0 = np.abs(np.array([F.PlaneWaveSweep(i, np.array([39.]), 2.33) for i in range(len(p.AScans))]))
#
# I0 = I0[:,0,:].transpose()


# F.AScans = Acopy
#
# del(Acopy)

# F = FMC.LinearCapture(25., p.AScans, 0.5, 32)
#
# F.KeepElements(range(16,32))
#
# I1 = np.abs(np.array([F.PlaneWaveSweep(i, np.array([39.]), 2.33) for i in range(len(p.AScans))]))
#
# I1 = I1[:,0,:].transpose()
#
# fig, ax = plt.subplots(nrows=2)
#
# ax[0].imshow(I0[100:360,:], aspect=0.1)
#
# ax[1].imshow(I1[100:360,:], aspect=0.1)
#
# plt.show()
#
# del(F)
#
# yn = input("Scan Acceptable ? (y for yes, n for no)")
#
# if yn=='y':
#
#     p.SaveScans(scanpth+samplename+index+'.p',info)
#
#     del(p)
#
# else:
#
#     del(p)
#
#     pass
