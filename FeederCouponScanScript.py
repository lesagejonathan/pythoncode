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

index = float(sys.argv[3])

indexspeed = 0.5


scanspeed = 3.


scanres = 3
NScan = int(np.round(circumference/scanres))

# drot = (2.*scanres/diameter)*(180./np.pi)

dt = scanres/scanspeed

dtmin = 32*32/20000.

els = list(range(1,17))[-1::] + list(range(65,65+17))[-1::]

pos = np.linspace(0.,circumference,NScan)


if dt<dtmin:

    print("Scan Speed Must be Less Than "+str(scanres/dtmin))
    sys.exit()


# First skew

g = mc.Controller(instrument = 'ZMC4')

g.MoveToLimit('Index', indexspeed, 'Forward', Limit=30.)

g.MoveRelative('Index', -21., indexspeed, Wait=True)

def Scan(mc):

    p = mp.PeakNDT(fsamp=25.)

    p.SetFMCCapture((els,els), Gate = (0., 75.), Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1))


    mc.MoveAbsolute('Rotation', circumference, scanspeed, Wait=True)

    p.ExecuteCapture(NScan, dt)


    mc.MoveAbsolute('Rotation', 0., scanspeed, Wait=True)

    p.ReadBuffer()


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

    yn = input("Scan Acceptable ? (y/n)")

    if yn == 'y':

        return p

    else:

        del(p)

        return None



p = Scan(g)


while p is None:

    p = Scan(g)


p.SaveScans(scanpth+samplename+'A.p', {'ScanPositions': pos, 'IndexOffset':'90 degree skew at HAZ'})

del(p)

g.MoveRelative('Index', index, indexspeed, Wait=True)

p = Scan()


while p is None:

    p = Scan()


p.SaveScans(scanpth+samplename+'B.p', {'ScanPositions': pos, 'IndexOffset':'270 degree skew at HAZ'})

del(p)
del(g)
