import FMC
import numpy as np
import _pickle as pickle
import os
import copy

pth = '/Users/jlesage/Dropbox/Eclipse/HighTemp/'

#
# F = FMC.LinearCapture(25.,[np.ones((64,64,100))],1.,64,WedgeParameters={'Velocity':2.978,'Angle':0.,'Height':30.})
# #
# #
# F.SetRectangularGrid(0.,63.,0.,110.,0.1,0.1)
#
# F.GetWedgeDelays(5.92,0.)




#

# Nx = len(F.xRange)
# d = pickle.load(open(pth+'HT-Delays.p','rb'))

# Nx = dl[0][0].shape[1]
# Ny = dl[0][0].shape[0]
#
# x = np.linspace(0.,63.,Nx)
# y = np.linspace(0.,120.,Ny)
#
# rdl = np.array([(30./2.978 + y/5.92)*np.ones((1,Nx)) for y in F.yRange])
#
# D = {}
#
# D['FMCDelays'] = F.Delays
# D['xRange'] = F.xRange
# D['yRange'] = F.yRange
# D['FORDelays'] = (rdl,F.Delays[0])
#
# pickle.dump(D,open(pth+'HT-Delays.p','wb'))





#
# rdl = (rdl,dl[0])
#
# D = {}
#
# D['xRange'] = x
# D['yRange'] = y
# D['FORDelays'] = rdl
# D['FMCDelays'] = dl

# pickle.dump(D,open('/Users/jlesage/Dropbox/Eclipse/HighTemp/HTDelays.p','wb'))


# fls = os.listdir(pth)
#
#
#
# fls = [fl for fl in fls if (fl.endswith('.p'))&(not(fl=='HTDelays.p'))]
#
# Imgs = {}
#
# for fl in fls:
#
#     Fl = pickle.load(open(pth+fl,'rb'))
#
#     F = FMC.LinearCapture(25.,Fl['AScans'],1.,64)
#
#     I = []
#
#     F.xRange = d['xRange']
#     F.yRange = d['yRange']
#
#     F.ProcessScans(200,100)
#
#     if F.AScans[0].shape[0] > 1:
#
#         # F.Delays = (d['FMCDelays'][0][350:650,:],d['FMCDelays'][1][350:650,:])
#
#         F.Delays = d['FMCDelays']
#
#
#         # F.yRange = d['yRange'][350:650,:]
#
#         I.append(F.ApplyTFM(0))
#
#         # F.Delays = (d['FMCDelays'][0][850:1050,:],d['FMCDelays'][1][850:1050,:])
#         #
#         # F.yRange = d['yRange'][850:1050,:]
#
#         I.append(F.ApplyTFM(1))
#
#
#
#     else:
#
#             F.Delays = d['FORDelays']
#             #
#             # F.yRange = d['yRange'][350:650,:]
#
#             I.append(F.FocusOnReception(0))
#
#             # F.Delays = copy.deepcopy(d['FORDelays'][850:1050,:])
#             #
#             # F.yRange = d['yRange'][850:1050,:]
#
#             I.append(F.FocusOnReception(1))
#
#     Imgs[fl.strip('.p')] = I
# #
# pickle.dump(Imgs,open(pth+'HTImages.p','wb'))
