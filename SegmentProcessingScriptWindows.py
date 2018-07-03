import SegmentAnalysis as sa
import Signal
import pickle
import os
import multiprocessing
import FMC
from numpy import *

# def ImageWeld(x):
#
#     print(x)
#
#     wld = pickle.load(open(pth+x,'rb'))
#
#     F = sa.FMC(wld['AScans'],wld['SamplingFrequency'])
#
#     F.Calibrate()
#
#     F.FusionLineFocus()
#
#     return F.FusionLineImages
#


pth = '/mnt/c/Users/jlesage/Documents/MicroPulseFBHBacksideScans/ToProcess/'

# pth = '/mnt/c/Users/jlesage/Documents/MicroPulseFBHBackSideScans/Remaining/'


pd = pickle.load(open('/mnt/c/Users/jlesage/Dropbox/Eclipse/FMCReferenceScans/MP-5L32-Delays.p','rb'))

# dref = pickle.load(open('/mnt/c/Users/jlesage/Dropbox/Eclipse/FMCReferenceScans/5L32-25mmReferenceBlock.p','rb'))
# pd = Signal.EstimateProbeDelays(dref['AScans'][0],dref['SamplingFrequency'],0.6,25.)

d = os.listdir(pth)

d = [dd for dd in d if dd.endswith('.p')]

# d = [d[0]]
# d = d[0:2]

wld = pickle.load(open(pth+d[0],'rb'))

w = FMC.LinearCapture(25.,wld['AScans'],0.6,3.24,32,probedelays=pd)

w.GetWedgeDelays(linspace(40-15,40+15,120),linspace(25,25+30,120))


D = w.Delays.copy()

def ImageWeld(x):

    print(x)

    wld = pickle.load(open(pth+x,'rb'))

    w = FMC.LinearCapture(25.,wld['AScans'],0.6,3.24,32,probedelays=pd)

    w.ProcessScans()

    w.Delays = D

    I = w.ApplyTFM()

    I = [abs(II).transpose() for II in I]

    pickle.dump(I,open('/mnt/c/Users/jlesage/Documents/MP-TFM-'+x,'wb'))



    #
    # F = sa.FMC(wld['AScans'],wld['SamplingFrequency'])
    #
    # F.Calibrate(ProbeDelays=pd)
    #
    # F.FusionLineFocus()
    #
    # outpt = {'WeldParameters':F.WeldParameters,'Images':F.FusionLineImages}

    # pickle.dump(outpt,open('/mnt/c/Users/jlesage/Documents/MicroPulseFBHScans/MP-CustomTFM-'+x,'wb'))

    return 0


# pool = multiprocessing.Pool(multiprocessing.cpu_count())

pool = multiprocessing.Pool(3)

II = pool.map(ImageWeld, d)

# I = {}
#
# for i in range(len(d)):
#
#     I[d[i].strip('.p')] = II[i]
#
#
# pickle.dump(I,open('/mnt/c/Users/jlesage/Documents/CustomTFM-MicroPulse-L-FBH-10DegreeWedge.p','wb'))
