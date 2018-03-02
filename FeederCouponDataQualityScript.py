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

d = os.listdir(pth)

d = [dd if dd.endswith('.py') for dd in d]

for dd in d:

    f = pickle.load(open(pth+dd, 'rb'))

    F = FMC.LinearCapture(25., f['AScans'], 0.5, 32)

    F.KeepElements(range(16))

    I0 = np.array([np.abs(detrend(F.PlaneWaveSweep(i, [39.], 2.33))) for i in range(len(p.AScans))])

    I0 = I0[:,0,:].transpose()

    F = FMC.LinearCapture(25., f['AScans'], 0.5, 32)

    F.KeepElements(range(16,31))

    I1 = np.array([np.abs(detrend(F.PlaneWaveSweep(i, [39.], 2.33))) for i in range(len(p.AScans))])

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(I0[int(7.5*25.):int(15.*25),:], aspect=13.)

    ax[1].imshow(I1[int(7.5*25.):int(15.*25),:], aspect=13.)

    plt.savefig(pth+dd.strip('.p')+'.png', format='png', dpi=200)

    plt.close()
