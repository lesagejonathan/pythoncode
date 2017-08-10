import MotionController as mc
import MicroPulse as mp
import numpy as np

pth = 'E:/MicroPulseFBHBacksideScans/'

SampleName = 'D01-03-Backside'

m = mp.PhasedArray(32,pwidth=1/10.)
c = mc.Controller()

Npos = 80

speed = 3.

p = np.linspace(0,Npos-1,Npos)

for pp in p:

    c.MoveAbsolute('Y',pp,speed)

    m.GetFMCData((0.,85.),dB=50.,Voltage=200.,Averages=8)


m.SaveScan(pth+SampleName+'1.p')

m.ClearAScans()

for pp in p[::-1]:

    c.MoveAbsolute('Y',pp,speed)

    m.GetFMCData((0.,85.),dB=50.,Voltage=200.,Averages=8)




m.SaveScan(pth+SampleName+'2.p',Reversed=True)

m.ClearAScans()

for pp in p:

    c.MoveAbsolute('Y',pp,speed)

    m.GetFMCData((0.,85.),dB=50.,Voltage=200.,Averages=8)


m.SaveScan(pth+SampleName+'3.p')

c.MoveRelative('Z',60.,5.)
c.MoveAbsolute('Y',0.,5.)


del(m)
del(c)
