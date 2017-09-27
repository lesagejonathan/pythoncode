import MicroPulse as mp
import MotionController as mc
import FMC
import numpy as np
import time
import pickle

""" Script is used to collect required FMC data from a reference block
to fit directivity and attenuation/geometric spreading correction functions """

h = [ (25., 6.), (50., 12.), (75., 18.), (100., 24.),
     (125., 30.), (150., 36.) ] # List defining position of hole w.r.t. edge of plate

thmax = np.pi*60/180. # Steering range of the array w.r.t. centre of the aperture (radians)

p = 0.6  # Pitch of the array in mm

N = 64  # Number of Elements

dx = 1. # Spacing of Capture Points in mm

Lx = 255. # Length of Scan in mm

X0 = 25. # Initial Position of Centre of Aperture (mm)

X = np.linspace(X0, X0+Lx, int(np.round(Lx/dx))) # Vector of capture positions (mm)

T = 2*1.25*np.sqrt(( X[-1] + (N-1)*p/2 - h[-1][0])**2 + h[-1][1]**2)/5.92 # Time gate (mircoseconds)

F = mp.PhasedArray(N, fsamp=25., pwidth=1/20.) # MicroPulse object for FMC acquisition

G = mc.Controller() # MotionController object for Galil axis control

xy = [] # List to store position of hole to be imaged w.r.t. to first element

for XX in X:

    th = np.array([np.arctan2(np.abs(XX - hh[0]), hh[1]) for hh in h])

    indh = np.argmin(th)

    if th < thmax:

        F.GetFMCData((0., T), 80., 200., 16)

        xy.append((h[indh][0] - (XX - (N-1)*p/2.),h[indh][1]))

    G.MoveAbsolute('Y', XX , 1.)

    time.sleep(1.*dx*1.5)

pickle.dump({'HolePositions':xy, 'AScans': F.AScans, 'PulserSettings': F.PulserSettings}, open('/','wb'))
