import FMC
import _pickle as pickle
import numpy as np

ci = 1.541
cs = 5.9

s = pickle.load(open('/mnt/d/FMCScans/EPRIBlocks-5L64-NW1.p','rb'))['AScans']

F = FMC.LinearCapture(25., s, 0.6, 64)

F.ProcessScans(80)

h1 = F.FitInterfaceCurve(0,np.linspace(2.75,15.,150),ci)[0]

h2 = F.FitInterfaceCurve(1,np.linspace(3.,15.,150),ci)[0]

print('Block1 Processing')


F.GetAdaptiveDelays(np.linspace(-5.,45.,200),np.linspace(3.,50.,212),h1,ci,cs)

D = {}


I1 = F.ApplyTFM(0)

D['Block1'] = {'Image':I1, 'X':F.xRange, 'Y':F.yRange, 'h': h1(F.xRange)}

pickle.dump(D,open('/mnt/d/FMCScans/EPRIAdaptiveImages.p','wb'))


print('Block2 Processing')

F.GetAdaptiveDelays(np.linspace(-5.,45.,200),np.linspace(3.,51.,215),h2,ci,cs)

I2 = F.ApplyTFM(1)

D = pickle.load(open('/mnt/d/FMCScans/EPRIAdaptiveImages.p','rb'))

D['Block2'] = {'Image':I2, 'X':F.xRange, 'Y':F.yRange, 'h': h2(F.xRange)}

pickle.dump(D,open('/mnt/d/FMCScans/EPRIAdaptiveImages.p','wb'))
