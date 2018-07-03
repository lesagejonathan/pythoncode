import numpy as np
import FieldSimulation as fs
import _pickle as pickle

spec = pickle.load(open('/mnt/e/FMCCalibration/5L64-A2-Spectrum.p','rb'))
#
# #
# # A = [fs.CompressionElementField3d(0.6,10.,spec['Frequency'],spec['Spectrum'],0.25,150.,65.,cp=5.91,bc=b,eta=1e-4) for b in ('potential','displacement','stress')]
# #
# # D = {'Fields':{'potential':A[0],'displacement':A[1],'stress':A[2]}, 'x':np.arange(-75.,75.,0.25), 'y':np.arange(0.,65.,0.25)}
#
#
# A = fs.CompressionElementField3d(0.6,10.,spec['Frequency'],spec['Spectrum'],0.20,150.,100.,cp=5.91,bc='potential',eta=1e-4)
#
# D = {'Field':A, 'x':np.arange(-75.,75.,0.25), 'y':np.arange(0.,65.,0.25)}
#
# #
# pickle.dump(D, open('/mnt/e/FMCCalibration/5L64-A2-Fields.p','wb'))

x = np.arange(-18.9,18.9,2.5)
y = np.arange(5.,65.,5.)

Fn = [[fs.PointFocalLaw(spec['Frequency'],spec['Spectrum'],0.6,64,x[i],y[j],5.91) for i in range(len(x))] for j in range(len(y))]

A = [[fs.ContactArray2d(spec['Frequency'],Fn[i][j],0.6,0.25,150.,75.,cL=5.91,bc='potential') for i in range(len(x))] for j in range(len(y))]

B = np.array([[np.amax(A[i][j]) for i in range(len(x))] for j in range(len(y))])

D = {'PSF':A,'MaxPSF':B,'FocalPoints':(x,y), 'x':np.arange(-75.,75.,0.25), 'y':np.arange(0.,75.,0.25)}

pickle.dump(D, open('/mnt/e/FMCCalibration/5L64-A2-PSFs.p','wb'))
