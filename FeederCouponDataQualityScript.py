import MicroPulse as mp
import MotionController as mc
import sys
import numpy as np
import FMC
import copy
import matplotlib.pylab as plt
import _pickle as pickle
import os
import sys
from scipy.signal import detrend

"""

Usage Steps:

1. Copy scans for data quality check into '/mnt/d/FMCScans/ANSFeederScans/QualityCheck/'
2. Run this script: python3 FeederCouponDataQualityScript.py
3. Review images output by script into '/mnt/d/FMCScans/ANSFeederScans/QualityCheck/'
4. Delete copy of scans and verified images from '/mnt/d/FMCScans/ANSFeederScans/QualityCheck/'

* Put a copy of the scans in '/mnt/d/FMCScans/ANSFeederScans/QualityCheck/' prior to running
DO NOT cut or delete original scan file !!

"""

pth = '/mnt/d/FMCScans/ANSFeederScans/QualityCheck/'

# pth = '/mnt/d/ANS_FEEDER_FMC_DATA/'

d = os.listdir(pth)

d = [dd for dd in d if dd.endswith('.p')]

for dd in d:

    f = pickle.load(open(pth+dd, 'rb'))

    F = FMC.LinearCapture(25., f['AScans'], 0.5, 32)

    F.ProcessScans(10,bp=10)

    F.KeepElements(range(16))

    F.AScans = [F.AScans[i][::-1,::-1,:] for i in range(len(F.AScans))]

    I0 = np.array([np.abs(detrend(F.PlaneWaveSweep(i, np.array([-39.]), 2.33))) for i in range(len(f['AScans']))])


    I0 = I0[:,0,:]

    I0 = I0.transpose()

    F = FMC.LinearCapture(25., f['AScans'], 0.5, 32)

    F.ProcessScans(10,bp=10)

    F.KeepElements(range(16,32))

    F.AScans = [F.AScans[i][::-1,::-1,:] for i in range(len(F.AScans))]


    I1 = np.array([np.abs(detrend(F.PlaneWaveSweep(i, np.array([-39.]), 2.33))) for i in range(len(f['AScans']))])

    I1 = I1[:,0,:]

    I1 = I1.transpose()

    fig, ax = plt.subplots(nrows=2)

    # ax[0].imshow(I0[int(7.5*25.):int(15.*25),:], aspect=13.)
    #
    # ax[1].imshow(I1[int(7.5*25.):int(15.*25),:], aspect=13.)

    ax[0].imshow(I0, extent=[0.,I0.shape[1],5.9*I0.shape[0]/50.,0.])

    ax[1].imshow(I1,extent=[0.,I0.shape[1],5.9*I0.shape[0]/50.,0.])

    plt.savefig(pth+dd.strip('.p')+'.png', format='png', dpi=200)

    plt.close()
