import MicroPulse as mp
import MotionController as mc
import sys
import numpy as np


scanpth = '/mnt/d/FMCScans/FeederCoupons/'

samplename = argv[1]
diameter = argv[2]
scanres = argv[3]




p = mp.MicroPulse(fsamp=25.)


p.SetFMCCapture(32, Gate, Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1))

g = mc.MotionController(Instrument= 'FeederController')
