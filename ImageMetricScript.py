
import os
import shutil
from numpy import *
# import matplotlib.pylab as plt
import sys
import pickle
from sklearn.linear_model import LinearRegression
import SegmentImageAnalysis as sa
from skimage.filters import threshold_li
from functools import reduce
from skimage.measure import regionprops
from matplotlib.pylab import *



pth = 'C:/Users/jlesage/Documents/MP-BacksideTFM/'

# pth = '/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/T/Test/'
# pth = '/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/L-Compression/'

f = os.listdir(pth)

# f = f[1::]

# f = [ff for ff in f if not(ff is ".DS_Store")]
# f = f[1::]

# print('ba')
#
# H = array([])


# H = {'I14':1.27,'I15':1.29,'I16':0.66,'I17':0.94,'I18':2.13,'I19':0.81,'I20':0.34,'I21':0.69,'I22':0.4,'I23':0.61,'I24':0.91,'I25':1.33,'I26':1.16,'I27':0.74}
#
# T = array([3.67,4.65,2.9,4.45,4.01,3.59,3.71,4.08,4.03,5.40,4.29,4.45,4.3,4.41])
# H = array([ 1.17,  0.85,  1.07,  0.97,  0.64,  1.  ,  1.24,  0.77,  0.84,
#         1.47,  1.33,  0.86,  0.89,  1.09,  0.99,  1.23,  1.21,  1.21,
#         0.55,  0.44])

# T = array([4.23,4.35,3.88,4.39,4.39,4.03,5.12,3.82,4.51,4.09,4.18,3.21,3.72,3.94,4.38,3.79,4.15,4.06,4.75,4.97])
# T['I14']
# H = array([2.13])
# T = array([4.01])

# e = []
# lr = []
# ld = []
# h = []
#
# pdtr = []
# pde = []

# d = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/T/TDisbondLengths.p','rb'))

c = {}

for i in range(len(f)):

    w = pickle.load(open(pth+f[i],'rb'))

    w = w[::-1]

    w = [ww.transpose() for ww in w]

    print(f[i])

    fls  = sa.Weld(w,[[25.,55.],[25.,55.]])

    # H[f[i]] = fls.DisbondLengths*(H[f[i]]/mean(fls.DisbondLengths))


    fls.SmoothImages(0.5)

    fls.AxialStack()
    
    fls.RegisterImages()

    fls.AxialStack()

    fls.RectangularCropImages()

    fls.GetBinaryImages()

    fls.HorizontalStack()

    imshow(fls.BinaryHStack)

    bi = array(fls.BinaryHStack.copy())

    imshow(bi,aspect=0.1)

    pts = ginput(n=0,timeout=0)

    close()

    # print(pts)

    ec = array([ any(bi[int(pts[0][1]):int(pts[1][1]),j]>0.) for j in range(bi.shape[1]) ]).astype(int)

    fn = f[i].split('/')[0]

    # dl = d[f[i]]

    ecp = 100*sum(ec)/len(ec)

    # dlp = 100*sum(dl>1.5)/len(ec)
    #
    # err = ecp-dlp

    c[fn] = {'Calls':ec,'CallPercentage':ecp}


# pickle.dump(H,open(pth+'TDisbondLengths.p','wb'))
# pickle.dump(c,open(pth+'TCalls1p5.p','wb'))

# pickle.dump(c,open(pth+'TCalls1p5.p','wb'))

pickle.dump(c,open(pth+'MP-L-FBH-BacksideCalls.p','wb'))







#     dl = [ff for ff in fls if ff.endswith('.csv')][0]
#
#
#     N = len(fls)-1
#
#     h = loadtxt(pth+f[i]+'/'+dl,delimiter=',')
#
#     r = int(round(len(h)/N))
#
#     h = h[0::r]
#
#     N = min([len(h),N])
#
#     h = h[0:N]
#
#     h = h/(mean(h)/H[i])
#
#     # h = h[::-1]
#
#     d[f[i]] = h
# #
# pickle.dump(d,open('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/DisbondLengths.p','wb'))

    # I.HorizontalStack()

#     I.RemoveSlices()
#
#     print('Register')
#
#     I.RegisterImages()
#
#     print('Crop')
#     # I.SmoothImages(0.6)
#
#     # I.CropImages()
#
#     I.DefineSubRegions(True)
#
#     d[f[i]] = {'SubRegions':I.SubRegions,'DisbondLengths':I.DisbondLengths}
#
#
#
# pickle.dump(d,open('/Users/jlesage/Dropbox/Eclipse/LSubRegions.p','wb'))

    # bs1 = I.BinarySize()
    #
    #
    #
    # bs2 = I.BinarySize()






    # e.append(I.SubRegions[0]['Energy'][5:-5])
    # lr.append(I.SubRegions[0]['MaxDeviation'][5:-5])
    # ld.append(I.SubRegions[1]['MaxDeviation'][5:-5])
    # h.append(I.DisbondLengths[5:-5])

    # h.append(mean(I.DisbondLengths[5:-5]))
    # print(I.SubRegions[0]['BinarySize'][5:-5])

#     lr = I.SubRegions[0]['BinarySize'][5:-5]
#     lr = array(lr)*sqrt(2)/4
#
#     ld = array(I.SubRegions[1]['BinarySize'][5:-5])
#
#     print(lr)
#     print(ld)
#
#     l = array([max(array([lr[i],ld[i]])) for i in range(len(lr))])
#
#     l[l>T[i]] = T[i]
#
#     pdtr.append(100*abs(mean(I.DisbondLengths[5:-5])-T[i])/T[i])
#
#     pde.append(100*abs(mean(l)-T[i])/T[i])
#
#
# pde = array(pde)
# pdtr = array(pdtr)


# e = reduce(lambda x,y:concatenate((x,y)), [ee for ee in e])
# ld = reduce(lambda x,y:concatenate((x,y)), [ll for ll in ld])
# lr = reduce(lambda x,y:concatenate((x,y)), [ll for ll in lr])
#
# h = reduce(lambda x,y:concatenate((x,y)), [hh for hh in h])

# e = array(e)
# e = e>threshold_li(e)
# e = e.astype(int)
# e = e.reshape((len(e),1))

# l = array(l)



# ld = array(ld).reshape(-1,1)
# lr = array(lr).reshape(-1,1)
#
# # l = l<threshold_li(l)
# # l = l.astype(int)
# # l = l.reshape((len(l),1))
#
#
#
# # h = array(h)
# h = h.reshape((len(h),1))
#
# # he = hstack((h,e))
# hl = hstack((h,lr,ld))  #,L.reshape((L,1))))
#


# p = polyfit(L,h,)

# savetxt()

# savetxt('/Users/jlesage/Dropbox/Eclipse/EnergyHitMiss.txt',he,delimiter=',')


# savetxt('/Users/jlesage/Dropbox/Eclipse/LLengthMetrics.txt',hl,delimiter=',')









# p = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/ModelLTest/ModelImageMetrics.p','rb'))
#
# X = p[0].astype(float)
# y = p[1].astype(float)
#
# th = [25.,27.5,30.,32.5]



# vh = [(float(v),float(h)) for v in range(5,9) for h in range(3,9) if h<=v]

# for t in th:
#
#     l = LinearRegression()
#
#     ind = X[:,]



# l = os.listdir(pth)
#
# l = [ll for ll in l if ll.endswith('.png')]
#
# X = []
# y = []
#
#
# for ll in l:
#
#     # f = os.listdir(pth+ll)[0]
#
#     d = ll.split('_')
#
#     if d[1]==275:
#
#         d[1] = 27.5
#
#
#
#     # break
#     I = sa.Weld(pth+ll,[[5.,75.],[35.,105.]])
#
#     I.DefineSubRegions(True)
#
#
#
#     X.append([d[1],d[3],d[5],I.SubRegions[0]['MaxDeviation'][0],I.SubRegions[0]['Energy'][0],I.SubRegions[0]['SNR1'][0],I.SubRegions[0]['Properties'][0].max_intensity])
#     y.append([d[7]])
#
# X = array(X)
# y = array(y)
#
# d = (X,y,('Thickness','VFL','HFL','MaxDeviation','Energy','SNR','MaxIntensity'))
#
# pickle.dump(d,open('/Users/jlesage/Dropbox/Eclipse/ModelLTest/ModelTestImageMetrics.p','wb'))
