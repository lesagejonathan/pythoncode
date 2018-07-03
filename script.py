# import Eclipse as ec
# from os import listdir
# import os
# import shutil
# from numpy import *
# from skimage.measure import regionprops
# # import matplotlib.pylab as plt
# import sys
# import pickle
# from multiprocessing import Pool
# import FMC
# from scipy.io  import loadmat,savemat
#
# from skimage.measure import label,regionprops
#
# import SegmentImageAnalysis as sa
import os
from numpy import *
from numpy import sum as asum
import pickle
import SegmentAnalysis as sa
from functools import reduce

pth = '/Volumes/STORAGE/CustomTFMRendering/'

d = os.listdir(pth)

for dd in d:

    f = pickle.load(open(pth+dd,'rb'))

    if type(f) is list:

        I = sa.FMC(f)

    else:

        I = sa.FMC(f['AScans'])


    I.Calibrate()

    I.FusionLineFocus()

    pickle.dump({'FusionLine':I.FusionLineImages},open(pth+dd+'FusionLine.p','wb'))





# pth = 'C:/Users/jlesage/Documents/POD-FBHS-5MHZ/L/SweepAndCustomTFM/'
# #
# # I = array([0.])
# #
# d = os.listdir(pth)
# #
# # for dd in d:
# #
# #     f = pickle.load(open(pth+dd,'rb'))
# #
# #     II = array(f['FusionLine'][0])[:,:,0] + array(f['FusionLine'][1])[:,:,0] + array(f['FusionLine'][2])[:,:,0]
# #
# #     I = concatenate((I,II))
# #
# # I = I[1::]
#
#
#
#
#
# for dd in d:
#
#     f = pickle.load(open(pth+dd,'rb'))
#     #
#     # print(dd.strip('.p'))
#     # print(len(f['DisbondLengths']))
#
#     if type(f) is dict:
#
#         F = sa.FMC(f['AScans'])
#
#     elif type(f) is list:
#
#         F = sa.FMC(f)
#
#
#     F.Calibrate()
#
#     F.FusionLineFocus()
#
#     sw = [F.PlaneWaveSweep(i,linspace(-2.5,2.5,10),linspace(-2.5,2.5,10)) for i in range(len(F.AScans))]
#
#     sw = array(sw)
#
#     sw = asum(asum(abs(sw),axis=2),axis=1)
#
#     pickle.dump({'Sweep':sw, 'FusionLine': F.FusionLineImages}, open(pth+dd.strip('.p')+'tfmandsweep.p','wb'))


    # pickle.dump({'DisbondLine':f['DisbondLengths'],'Sweep':sw, 'FusionLine': F.FusionLineImages}, open(pth+dd.strip('.p')+'tfmandsweep.p','wb'))

#
# f = pickle.load(open('G:/CIVAAscans/AScans30/CIVAAscansTh30.p','rb'))
# #
# #
# l = []
#
# for ff in f:
#
#
#     F = sa.FMC(25.,ff['WeldParameters'])
#
#     F.LoadAScans(ff['AScans'],'array')
#
#     F.Calibrate()
#
#     Isc = F.SymmetricModeImage('SideWallCap',((3.,8.),(3.,8.)))
#     Ics = F.SymmetricModeImage('CapSideWall',((3.,8.),(3.,8.)))
#
#     l.append({'VFL':ff['WeldParameters']['VerticalLOF'],'HFL':ff['WeldParameters']['HorizontalLOF'],'DisbondLength':ff['DisbondLength'],'SideWallCap':Isc,'CapSideWall':Ics})
#
# pickle.dump(l,open('G:/SymmetricPathsTh30.p','wb'))
#
#
# f = pickle.load(open('G:/CIVAAscans/AScans275/CIVAAscansTh275.p','rb'))
#
# l = []
#
# for ff in f:
#
#
#     F = sa.FMC(25.,ff['WeldParameters'])
#
#     F.LoadAScans(ff['AScans'],'array')
#
#     F.Calibrate()
#
#     Isc = F.SymmetricModeImage('SideWallCap',((3.,8.),(3.,8.)))
#     Ics = F.SymmetricModeImage('CapSideWall',((3.,8.),(3.,8.)))
#
#     l.append({'VFL':ff['WeldParameters']['VerticalLOF'],'HFL':ff['WeldParameters']['HorizontalLOF'],'DisbondLength':ff['DisbondLength'],'SideWallCap':Isc,'CapSideWall':Ics})
#
# pickle.dump(l,open('G:/SymmetricPathsTh275.p','wb'))
#
#
# f = pickle.load(open('G:/CIVAAscans/AScans325/CIVAAscansTh325.p','rb'))
#
# l = []
#
# for ff in f:
#
#
#     F = sa.FMC(25.,ff['WeldParameters'])
#
#     F.LoadAScans(ff['AScans'],'array')
#
#     F.Calibrate()
#
#     Isc = F.SymmetricModeImage('SideWallCap',((3.,8.),(3.,8.)))
#     Ics = F.SymmetricModeImage('CapSideWall',((3.,8.),(3.,8.)))
#
#     l.append({'VFL':ff['WeldParameters']['VerticalLOF'],'HFL':ff['WeldParameters']['HorizontalLOF'],'DisbondLength':ff['DisbondLength'],'SideWallCap':Isc,'CapSideWall':Ics})

# pickle.dump(l,open('G:/SymmetricPathsTh325.p','wb'))

# d = []
#
# for ff in f:
#
#     fl = os.listdir(pth+ff+'/')
#
#     d = []
#
#     for ffl in fl:
#
#         p = ffl.split('_')[1::2]
#
#         a = Civa.LoadAScansFromTxt(pth+ff+'/'+ffl)
#
#         w={'Thickness':float(p[0]), 'VerticalLOF':float(p[1]), 'HorizontalLOF':float(p[2]), 'SideWallPosition':float(p[1])+26.16, 'Velocity':{'Longitudinal':5.9,'Transverse':3.24}, 'Impedance':{'Longitudinal':5.9*7.8,'Transverse':3.23*7.8}}
#
#         d.append({'AScans':a,'WeldParameters':w,'DisbondLength':float(p[3]),'SamplingFrequency':25.})
#
#     pickle.dump(d,open(pth+ff+'/CivaAScansTh'+p[0]+'.p','wb'))







#
#
# # pth = 'C:/Users/jlesage/Documents/FractureTestScans/FractureTest2Scans/AScans/Images/'
#
#
# # f = [['A1','A2','A3'],['B1','B2','B3'],['C1','C2','C3'],['D1','D2','D3']]#,['E1','E2','E3']]
#
# # f = [['A1','A2'],['B1','B2']]
#
# # f = [['E1','E2','E3']]
#
# pth = 'C:/Users/jlesage/Documents/RepeatabilityStudy/Images/'
#
# f = os.listdir(pth)
#
# if os.path.exists(pth+'LRepeat.p'):
#     d = pickle.load(open(pth+'LRepeat.p','rb'))
#     f = list(set(f)-set(d.keys()))
#     SZ = d.copy()
# else:
#     SZ = {}
#
# f = [ff for ff in f if not(ff.endswith('.p'))]
#
# for ff in f:
#
#     print(ff)
#
#     # w = sa.Weld(pth+fff+'/',[[5.,55.],[5.,55.]])
#
#     w = sa.Weld(pth+ff+'/',[[5.,75.],[35.,105.]])
#
#
#     # w.DefineSubRegions(True)
#
#     # rs = w.SubRegions[0]['BinarySize']/(2*sqrt(2))
#     #
#     # inds = w.SubRegions[1]['BinarySize']
#
#     w.RectangularCropImages()
#
#     w.RectangularCropImages()
#
#     w.SmoothImages(0.6/w.Period[0])
#
#     w.GetBinaryImages()
#
#     w.HorizontalStack()
#
#     imshow(w.BinaryHStack,aspect=0.1,cmap='gray')
#
#     pts = ginput(3,timeout=0)
#
#     close()
#
#
#     sz = []
#
#     for i in range(len(w.Images)):
#
#
#
#         l = label(w.BinaryImages[i])
#
#         r = regionprops(l,w.Images[i])
#
#
#         if len(r)>0:
#
#             rroot = [rr for rr in r if (rr.centroid[0]>pts[0][1])&(rr.centroid[0]<pts[1][1])]
#
#             if len(rroot)>0:
#
#                 rroot = array(rroot)[argsort(array([rr.max_intensity for rr in rroot]))[-1]]
#
#             rind = [rr for rr in r if (rr.centroid[0]>pts[1][1])&(rr.centroid[0]<pts[2][1])]
#
#             if len(rind)>0:
#
#                 rind = array(rind)[argsort(array([rr.max_intensity for rr in rind]))[-1]]
#
#                 sz.append(rind.major_axis_length*w.Period[0])
#
#             else:
#
#                 try:
#
#
#                     sz.append(rroot.major_axis_length*w.Period[0]/(2*sqrt(2)))
#
#                 except:
#
#                     sz.append(0.)
#
#         else:
#
#             sz.append(0.)
#
#     SZ[ff] = sz
#
# pickle.dump(SZ,open(pth+'LRepeat.p','wb'))


        # if len(r)>0:
        #
        #     # r = array(r)[argsort(array([rr.max_intensity for rr in r]))[-3::]]
        #     #
        #     # rroot = r[-1]
        #     #
        #     # rind = [rr for rr in r[0:2] if rr.centroid[0]>rroot.bbox[2]]
        #
        #     if len(rind)>0:
        #
        #         sz.append(rind[-1].major_axis_length*w.Period[0])
        #
        #     else:
        #
        #         sz.append(rroot.major_axis_length*w.Period[0]/(2*sqrt(2)))
        #
        # else:
        #
        #     sz.append(0.)
        #
        #








        #
        # sz = []
        #
        # for i in range(len(rs)):
        #
        #     if inds[i]>0:
        #
        #         sz.append(inds[i])
        #
        #     else:
        #
        #         sz.append(rs[i])


    #     plot(sz)
    #
    #
    # xlabel('Weld Position (mm)')
    # ylabel('Estimated Dis-bond (mm)')
    #
    # legend(['Trial 1','Trial 2','Trial 3'],loc='best')
    #
    # show()
    #
    # savefig('C:/Users/jlesage/Documents/'+ff[0][0]+'.png',format='png',dpi=250)
    #
    # close()




















# pth = 'C:/Users/jlesage/Documents/LProcessedScans/AScans/'
#
#
# # pth = pth[1::]
#
# d = os.listdir(pth)
# #
# #
# #
# # d = [dd for dd in d if dd.endswith('.p')]
# #
# # for dd in d:
# #
# #     f = pickle.load(open(pth+dd,'rb'))
# #
# #     F = []
# #
# #     for ff in f:
# #
# #         F.append(sum(sum(abs(ff),axis=0),axis=0))
# #
# #     F = array(F).transpose()
# #
# #     imshow(F)
# #
# #
#
#
#
#
#
# F = sa.FMC(25.)
#
#
# for dd in d:
#
#     fl = os.listdir(pth+dd)
#
#     fl = [int(f.split('.')[0]) for f in fl]
#
#     fl.sort()
#
#
#     P = []
#
#     for f in fl:
#
#
#         a = fromfile(pth+dd+'/'+str(f)+'.txt',dtype=int16)
#
#
#         a = a.reshape((32,32,1626))
#
#         F.LoadAScans(a,'array')
#
#         F.Calibrate()
#
#         P.append(F.PlaneWaveSweep(linspace(-5.,5.,10),linspace(-5.,5.,10)))
#
#
#     pickle.dump(P,open(pth+dd+'.p','wb'))
#








# pth = '/Volumes/Storage/BallScans/AScans/'
#
# # pth = pth[1::]
#
# d = os.listdir(pth)
#
#
# d = [dd for dd in d if dd.endswith('.txt')]
#
# xrng = linspace(0.,5.,int(5/0.25))
# yrng = linspace(0.,5.,int(5/0.25))
# zrng = linspace(0.,20.,int(20/0.25))
#
# D = {}

# a = loadmat('C:/Users/jlesage/Dropbox/Eclipse/AKSonics/flatMetal3.mat')
# a = moveaxis(a['test'],0,2)

# F = FMC.MatrixCapture(40.,a,2.5,1.49,(10,5))



# for dd in d:
#
#     a = fromfile(pth+dd,dtype=int16)
#
#     a = a.reshape((64,64,1126))
#
#     F = FMC.MatrixCapture(25.,a,0.6,1.49,(8,8))
#
#     F.ProcessScans(4.)
#
#     F.GetDelays(xrng,yrng,zrng)
#
#     F.ApplyTFM()
#
#     D[dd.split('.')[0]] = abs(F.TFMImage)
#
#
#
# D['xrange'] = xrng
# D['yrange'] = yrng
# D['zrange'] = zrng
#
# savemat(pth+'BallScans.m',D)
#





# import SegmentAnalysis as sa

# import SegAnalysis as sa


# pth = 'C:/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/AScans/'
#
# # depthsfl = 'C:/Users/jlesage/Documents/SmallPlateDepths.csv'
#
# outptha = 'C:/Users/jlesage/Documents/LProcessedScans/AScans/'
# outpthc = 'C:/Users/jlesage/Documents/LProcessedScans/Correlation/'
# outpthcp = 'C:/Users/jlesage/Documents/LProcessedScans/CorrelationPeaks/'
#
#
# d = os.listdir(pth)
#
# fin = os.listdir(outptha)
#
# d = list(set([dd.split('.')[0] for dd in d])-set(fin))
#
#
# for dd in d:
#
#     f = pickle.load(open(pth+dd+'.p','rb'))
#
#     F = sa.FMC(25.)
#
#     fl = dd.split('.')[0]
#
#     os.makedirs(outptha+fl)
#     os.makedirs(outpthc+fl)
#     os.makedirs(outpthcp+fl)
#
#
#
#     for i in range(len(f)):
#
#         F.LoadAScans(f[i],'array')
#
#         F.Calibrate()
#
#
#         F.AScans.tofile(outptha+fl+'/'+str(i)+'.txt')
#         F.Correlation.tofile(outpthc+fl+'/'+str(i)+'.txt')
#         F.CorrelationPeaks.tofile(outpthcp+fl+'/'+str(i)+'.txt')
#


# d = d[1::]

# print(d)
#
#
# if os.path.exists(depthsfl):
#
#     f = open(depthsfl,'r')
#     l = f.readlines()
#     finished = [ll.split(',')[0] for ll in l]
#     f.close()
#
#     f = open(depthsfl,'a')
#
#     d = list(set(d)-set(finished))
#
#     print(d)
#
# else:
#
#     f = open(depthsfl,'w')
#
#
#
# d.sort()


# for dd in d:
#
#     # fn = os.listdir(pth+'/'+dd)[0]
#
#     try:
#
#
#         print(pth+dd+'/')
#
#         F = sa.Weld(pth+dd+'/',[[0.,30.],[0.,30.]])
#
#
#         dy = F.Period[1]
#
#         F.DefineSubRegions(True)
#
#         if len(F.SubRegions)==0:
#
#             depth = 0.
#
#
#         elif len(F.SubRegions)==1:
#
#             I = F.SubRegions[0]['Images'][0]
#
#             imax = unravel_index(argmax(I),I.shape)
#
#             # depth = 14.-dy*imax[0]
#             #
#             # print(depth)
#
#             l = I>0.5*amax(I)
#
#             l = l.astype(int)
#
#             r = regionprops(l,I)
#
#             r = r[array([rr.max_intensity for rr in r]).argmax()]
#
#             depth = dy*(imax[0]-r.bbox[0])
#
#
#         elif len(F.SubRegions)==2:
#
#             I1 = F.SubRegions[0]['Images'][0]
#             I2 = F.SubRegions[1]['Images'][0]
#
#             ind1 = unravel_index(argmax(I1),I1.shape)
#             ind2 = unravel_index(argmax(I2),I2.shape)
#
#             depth = dy*(abs(ind1[0]-ind2[0]))
#
#         f.write(dd+','+str(depth)+'\n')
#
#     except:
#
#         pass
#
# f.close()
#
#











# def GeometryEstimation(d):
#
#     F = sa.FMC(25.)
#
#     F.LoadAScans(d,'array')
#     F.Calibrate()
#     F.EstimateSideWall()
#     F.EstimateCap()
#
#     return (F.WeldParameters['Thickness'],F.WeldParameters['SideWallPosition'],F.WeldParameters['VerticalLOF'],F.WeldParameters['HorizontalLOF'])
#
#
# if __name__ == '__main__':
#
#     D = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/LControlledAScans/1H1L-1.p','rb'))
#
#     D = D[10:-10:5]
#
#     p = Pool(processes=None)
#
#     g = p.map(GeometryEstimation, [ d for d in D])
#
#     pickle.dump(g,open('/Users/jlesage/Dropbox/Eclipse/LControlledAScans/Geo1H1L-1.p','wb'))
#
#







# pth = 'C:/Users/jlesage/CivaOutput/th_32p5/'
#
# l = os.listdir(pth)
#
#
# for ll in l:
#
#     f = os.listdir(pth+ll)[0]
#
#     print(ll)
#     print(f)
#
#     shutil.copy(pth+ll+'/'+f,pth+ll+'.png')
#






# import pickle
# import itertools
# import multiprocessing
# import sys
#
# pth = 'C:/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/AScans/'
# l = os.listdir(pth)
#
# th = {}
# sw = {}
#
#
#
# for ll in l:
#
#     F = sa.FMC(25.)
#     d = pickle.load(open(pth+ll,'rb'))
#
#     s = []
#     t = []
#
#     F.LoadAScans(d[25],'array')
#
#     F.Calibrate()
#     F.EstimateSideWall()
#
#     s.append(F.WeldParameters['SideWallPosition'])
#     t.append(F.WeldParameters['Thickness'])
#
#     F.LoadAScans(d[45],'array')
#
#     F.Calibrate()
#     F.EstimateSideWall()
#
#     s.append(F.WeldParameters['SideWallPosition'])
#     t.append(F.WeldParameters['Thickness'])
#
#     F.LoadAScans(d[60],'array')
#
#     F.Calibrate()
#     F.EstimateSideWall()
#
#     s.append(F.WeldParameters['SideWallPosition'])
#     t.append(F.WeldParameters['Thickness'])
#
#     th[ll.split('.')[0]] = mean(t)
#     sw[ll.split('.')[0]] = mean(s)
#



# th = float(sys.argv[1])
# h1 = int(sys.argv[2])
# h2 =int(sys.argv[3])
# v1 = int(sys.argv[4])
# v2 = int(sys.argv[5])
#
# flname = sys.argv[6]
#
#
# def findpath(p):
#
#     from numpy import sum as asum
#
#     print('VFL: '+str(p[1]['VerticalLOF']))
#     print('HFL: '+str(p[1]['HorizontalLOF']))
#     print('\n')
#     print(p[0][1])
#
#     d = sa.Delays(p[0],(range(0,32),range(0,32)),p[1])
#
#     nconv = asum(isfinite(array([ d[m][n][0] for m in range(15,16) for n in range(15,16)])))
#
#     print('\n')
#     print(nconv)
#
#     return (p[0],p[1],nconv,d)
#
# if __name__ == '__main__':
#
#     # th = float(sys.argv[1])
#     # h1 = int(sys.argv[2])
#     # h2 =int(sys.argv[3])
#     # v1 = int(sys.argv[4])
#     # v2 = int(sys.argv[5])
#     #
#     # flname = sys.argv[6]
#
#     allpaths = []
#
#
#     pth1c = list(itertools.product(['L','T'],repeat=6))
#
#
#     for i in range(0,len(pth1c)):
#         allpaths.append(('SideWallCap',pth1c[i]))

    # for j in range(0,len(pth2c)):
    #     allpaths.append(('path2',pth2c[j]))
    #
    # for k in range(0,len(pth3c)):
    #     allpaths.append(('path3',pth3c[k]))

    # for i in range(0,len(pth1c)):
    #     allpaths.append(('path4',pth1c[i]))
    #
    # for j in range(0,len(pth3c)):
    #     allpaths.append(('path5',pth3c[j]))


    # #
    # allpaths.append(('path1',('L','L','L','L','L')))
    # allpaths.append(('path2',('L','L','L','L','L','L')))
    # allpaths.append(('path3',('L','L','L','L','L','L','L')))
    # allpaths.append(('path4',('L','L','L','L','L')))
    # allpaths.append(('path5',('L','L','L','L','L','L','L')))
    #
    #
    # p = multiprocessing.Pool(processes=2)
    #
    # WP = lambda th,vlof,hlof: {'Thickness':th,'SideWallPosition':26.1189+vlof,'VerticalLOF':vlof,'HorizontalLOF':hlof}
    #
    # wp = [WP(th,float(v),float(h)) for v in range(v1,v2+1) for h in range(h1,h2+1) if h<=v]
    #
    # convergence = p.map(findpath,[(pth,w) for w in wp for pth in allpaths])
    #
    # pickle.dump(convergence,open(flname,'wb'))

# l = listdir('/Users/jlesage/Dropbox/Eclipse/PathConvergence/')
# l = ['/Users/jlesage/Dropbox/Eclipse/PathConvergence/'+ll for ll in l if ll.endswith('.p')]
#
# paths = {}
#
#
# for ll in l:
#
#     f = pickle.load(open(ll,'rb'))
#
#
#     for ff in f:
#
#         wp = ff[0]
#
#         p = ff[1]
#
#         T = ff[2]
#
#         T = [ T[m][n][0] for n in range(32) for m in range(32) ]
#
#         nT = sum(isfinite(array(T)).astype(int))
#
#
#
#         print(wp)
#         print(nT)
#
#         if nT>0:
#
#             try:
#
#                 paths['th'+str(int(wp['Thickness']))+'v'+str(int(wp['VerticalLOF']))+'h'+str(int(wp['HorizontalLOF']))] = paths['th'+str(int(wp['Thickness']))+'v'+str(int(wp['VerticalLOF']))+'h'+str(int(wp['HorizontalLOF']))] + [(p,nT)]
#
#             except KeyError:
#
#                 paths['th'+str(int(wp['Thickness']))+'v'+str(int(wp['VerticalLOF']))+'h'+str(int(wp['HorizontalLOF']))] = [(p,nT)]


    # for d in f:
    #
    #     p = d[1]
    #
    #     T = d[2][0]
    #
    #     T=[[t[i][0] for i in range(len(t[0]))] for t in T]
    #
    #     nT=[sum(isfinite(array(TT)).astype(int)) for TT in T][0]
    #
    #     ind = where(nT>0)
    #
    #     ind = argsort(ind)
    #
    #     nT = nT[ind]
    #
    #     p = [p[i] for i in ind]
    #
    #     paths['th'+str(d[0]['Thickness'])+'v'+str(d[0]['VerticalLOF'])+'h'+str(d[0]['HorizontalLOF'])] = (p,nT)
    #




# P = '/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/T/JSeries/'
#
# pth = listdir(P)
#
# pth = pth[1::]
#
# I = [ec.FMCImage(P+f+'/',[[3.,55.],[5.,40.]]) for f in pth]
#
# height = []
# call = []
#
# for II in I:
#
# 	II.AxialStack()
# 	II.RectangularCropImages()
# 	II.AxialStack()
#
# 	II.DisbondLengths = II.DisbondLengths[10:-9]
#
# 	II.Images = II.Images[10:-9]
#
# 	II.GetBinaryImages()
# 	II.HorizontalStack()
#
# 	h = II.DisbondLengths[II.DisbondLengths>1.]-1.
# 	call.append((II.BimodalityCoefficient[II.DisbondLengths>1.]>(5./9.)).astype(int))
# 	height.append(h)
#
#
#
# 	fig,ax = plt.subplots(4,sharex=True)
#
# 	# ax[0].axis('off')
# 	ax[0].set_title('Side View')
# 	# ax[1].axis('off')
# 	ax[1].set_title('Binary Side View')
#
# 	ax[2].set_title('Bimodality Coefficient')
#
# 	ax[3].set_title('Disbond Lengths')
#
# 	ax[0].imshow(II.HStack,extent=[II.Region[2,0],II.Region[2,1],II.Region[1,1],II.Region[1,0]])
#
# 	ax[1].imshow(II.BinaryHStack,cmap='gray',extent=[II.Region[2,0],II.Region[2,1],II.Region[1,1],II.Region[1,0]])
#
#
#
# 	ax[2].plot(II.BimodalityCoefficient)
# 	ax[2].plot((5/9)*ones(II.BimodalityCoefficient.shape))
#
# 	ax[3].plot(II.DisbondLengths)
# 	ax[3].plot(ones(II.DisbondLengths.shape))
#
#
# 	fig.savefig(II.File+'.png',format='png')
#
# 	plt.close(fig)
#
#
#
# h = [h[i] for h in height for i in range(len(h))]
#
# c = [c[i] for c in call for i in range(len(c))]
#
# f = hstack((array(h).reshape((len(h),1)), array(c).reshape((len(c),1))))
#
# savetxt('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/T/JPOD.csv',f,delimiter=',')
