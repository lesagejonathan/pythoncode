import SegmentAnalysis as sa
import Signal
import pickle
import os
import multiprocessing

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


pth = '/home/jbox/Documents/MicroPulse-L-FBHScans-10DegreeWedge/'

dref = pickle.load(open('/home/jbox/Documents/5L32-25mmReferenceBlock.p','rb'))
pd = Signal.EstimateProbeDelays(dref['AScans'][0],dref['SamplingFrequency'],0.6,25.)

d = os.listdir(pth)

d = [dd for dd in d if dd.endswith('.p')]
d = d[0:3]

def ImageWeld(x):

    print(x)

    wld = pickle.load(open(pth+x,'rb'))

    F = sa.FMC(wld['AScans'],wld['SamplingFrequency'])

    F.Calibrate(ProbeDelays=pd)

    F.FusionLineFocus()

    return F.FusionLineImages


pool = multiprocessing.Pool(multiprocessing.cpu_count())
II = pool.map(ImageWeld, d)

I = {}

for i in range(len(d)):

    I[dd.strip('.p')] = II[i]


pickle.dump(I,open('/home/jbox/Documents/CustomTFM-MicroPulse-L-FBH-10DegreeWedge.p','wb'))
