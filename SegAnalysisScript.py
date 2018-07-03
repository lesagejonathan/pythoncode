import os
import shutil
import time

import misc
from numpy import *
# import matplotlib.pylab as plt
import sys
import pickle
from multiprocessing import Pool
from functools import reduce
from sklearn.linear_model import LinearRegression,LogisticRegression
from matplotlib.pylab import *
from skimage.filters import threshold_otsu
from numpy import sum

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



import SegAnalysis as sa

modes = []
modes.append([(5,3),[('path2',('T','L','L','L','L','L')),('path2',('T','L','L','L','L','T'))]])
modes.append([(5,4),[('path2',('L','T','T','T','T','L')),('path2',('L','T','T','T','T','T'))]])
modes.append([(5,5),[('path2',('L','L','T','T','T','T')),('path2',('T','T','T','T','L','L'))]])

modes.append([(6,3),[('path4',('T','L','L','L','L','T')),('path1',('L','T','T','L','T'))]])
modes.append([(6,4),[('path2',('T','L','L','L','L','L')),('path2',('T','L','L','L','L','T'))]])
modes.append([(6,5),[('path2',('L','T','T','T','T','L')),('path2',('L','T','T','T','T','T'))]])
modes.append([(6,6),[('path2',('L','L','T','T','T','T')),('path2',('T','T','T','T','L','L'))]])

modes.append([(7,3),[('path2',('T','T','L','L','L','T')),('path2',('T','T','L','L','T','L'))]])
modes.append([(7,4),[('path1',('T','L','T','L','L')),('path1',('T','L','T','L','T'))]])
modes.append([(7,5),[('path2',('T','L','L','L','L','L')),('path2',('T','L','L','L','L','T'))]])
modes.append([(7,6),[('path2',('L','T','T','T','T','L')),('path2',('L','T','T','T','T','T'))]])
modes.append([(7,7),[('path2',('L','L','T','T','T','T')),('path2',('T','T','T','T','L','L'))]])

modes.append([(8,3),[('path2',('T','T','L','L','T','L')),('path2',('T','T','L','L','T','T'))]])
modes.append([(8,4),[('path1',('L','T','T','L','L')),('path4',('L','L','L','L','L','T'))]])
modes.append([(8,5),[('path2',('T','L','L','L','L','L')),('path2',('T','L','L','L','L','T'))]])
modes.append([(8,6),[('path2',('T','T','T','T','T','L')),('path2',('T','T','T','T','T','T'))]])
modes.append([(8,7),[('path2',('L','T','T','T','T','L')),('path2',('L','T','T','T','T','T'))]])
modes.append([(8,8),[('path2',('L','L','T','T','T','T')),('path2',('T','T','T','T','L','L'))]])


def AnalyzeWeldFrame(d):

    F = sa.FMC(25.)

    F.LoadAScans(d,'array')
    F.Calibrate()

    y = []
    I = []
    IP = []

    for p in modes:

        yy = []
        II = []
        IIP = []

        for pp in p[1]:

            d = F.FusionLineFocus(F.GetDelays(pp,[('VerticalLOF',p[0][0]),('HorizontalLOF',p[0][1])]))

            yy.append(d[0])
            II.append(d[1])
            IIP.append(d[2])

        I.append(II)
        y.append(yy)
        IP.append(IIP)


    return {'Thickness':F.WeldParameters['Thickness'], 'SideWallPosition':F.WeldParameters['SideWallPosition'], 'Intensities':I, 'YCoordinates':y, 'InnerProducts':IP}

def AnalyzeWeld(d):

    from scipy.signal import convolve,tukey

    w = sa.AnalyzeWeld(d[0])
    w.LoadDisbondLengths(d[1])

    w.StackMetrics()
    w.GetMaxMetrics()

    sm = w.GetMoments(('StackedMetrics','InnerProducts'),d[2])
    mm = w.GetMoments(('MaxMetrics','InnerProducts'),d[2])

    # sm = w.GetMoments(('StackedMetrics','Intensities'),d[2])
    # mm = w.GetMoments(('MaxMetrics','Intensities'),d[2])

    # for i in range(sm.shape[1]):
    #
    #     sm[:,i] = convolve(sm[:,i],tukey(5,0.1),mode='same')/sum(tukey(5,0.1))
    #     mm[:,i] = convolve(mm[:,i],tukey(5,0.1),mode='same')/sum(tukey(5,0.1))

    # c = array([modes[ind][0] for ind in w.MaxMetrics['Intensities'][2]])

    c = array([modes[ind][0] for ind in w.MaxMetrics['InnerProducts'][2]])


    return c,sm,mm,w.DisbondLengths


if __name__ == '__main__':

    # pth = '/Users/jlesage/Dropbox/Eclipse/ModeledResults/L/'
    #
    # fl = os.listdir(pth)
    #
    # fl = [f for f in fl if f.endswith('.p')]
    #
    # # fl = [fl[0]]
    #
    # for f in fl:
    #
    #     t0 = time.time()
    #
    #     print(f)
    #
    #     D = pickle.load(open(pth+f,'rb'))
    #
    #     print(len(D))
    #
    #     # D = [D[20]]
    #
    #     p = Pool(processes=None)
    #
    #     # g = [AnalyzeWeldFrame(d) for d in D]
    #
    #     g = p.map(AnalyzeWeldFrame, [d for d in D])
    #
    #     pickle.dump(g,open(pth+'Metrics'+f,'wb'))
    #
    #     print(time.time()-t0)

    pth = '/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/Metrics/'
    # pth = '/Users/jlesage/Dropbox/Eclipse/ModeledResults/L/'


    fl = os.listdir(pth)

    flp = [f for f in fl if f.endswith('.p')]

    thresh = 0.0

    # flcsv = [f for f in fl if fl.endswith('.csv')]

    w = [AnalyzeWeld((pth+flp[i],pth+flp[i].split('.')[0][-3::]+'.csv',thresh)) for i in range(len(flp))]

    # w = [AnalyzeWeld((pth+flp[i],pth+flp[i].split('.')[0]+'.csv',thresh)) for i in range(len(flp))]

    # h = reduce(lambda x,y: concatenate((x,y)), [ww[3] for ww in w])
    #
    # h = array(h)
    #
    # h = [sum(h[])]
    h = [sum(ww[3][i*10:10*(i+1)]) for i in range(4) for ww in w]

    # print(w[1][3])


    # h = h-mean(h)

    # h = h-1.

    # sm = reduce(lambda x,y: concatenate((x,y),axis=0), [ww[1] for ww in w])

    # sm = sm[h>0,:]

    # mm = reduce(lambda x,y: concatenate((x,y),axis=0), [ww[2] for ww in w])


    mm = [sum(ww[2][i*10:10*(i+1)],axis=0) for i in range(4) for ww in w]




    # print(len(h))
    # print(len(mm))

    # mm = mm[h>0,:]

    # h = h[h>0]

    # mm = mm[0:83,:]

    h = array(h)
    mm = array(mm)

    print(h)

    # print(h.shape)
    # print(mm.shape)

    R = {}
    keys = ['SumAboveThreshold','Centroid','RMS','Mean','STD','Max']

    for i in range(1,2):


        m = mm[:,i]

        m = m.reshape((len(m),1))




        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        sthresh = 35.

        h = 100*h/(70*5/4)

        # print(threshold_otsu(mm[:,i]))

        # s = LinearDiscriminantAnalysis()
        #
        # s.fit(mm[:,i].reshape(-1,1),h>sthresh)

        s = SVC(C=1.,kernel='linear',class_weight='balanced')
        s.fit(m,h>sthresh)

        # sthresh = threshold_otsu(m)

        print(sthresh)

        # sum()

        # print(sthresh)

        print(keys[i])

        l = LogisticRegression()
        l.fit(h.reshape(-1,1),s.predict(m))

        c = l.coef_[0]
        a = l.intercept_

        hh = linspace(min(h),max(h),100)

        plot(hh,1/(1+exp(-(c*hh+a))))

        ylabel('POD')
        xlabel('Percent Disbond (%)')

        show()


        # print(threshold_otsu(mm[:,i]))


        # print(s.decision_function(mm[:,i].reshape(-1,1)))

        # print(s.score(mm[:,i].reshape(1,-1),h>sthresh))
        #
        # print(s.predict(mm[:,i].reshape(1,-1)))


        print(s.score(m,h>sthresh))

        M = s.predict(m)


        print(M)
        M = M.astype(int)




        # X = hstack((array(s.predict(s.predict(mm[:,i].reshape(-1,1)))).reshape((len(mm[:,1],1))).astype(int),h.reshape((len(h),1))))

        X = hstack((M.reshape((len(M),1)),h.reshape((len(h),1))))


        savetxt('/Users/jlesage/Desktop/SVMSplit4.txt',X,delimiter=',')
        #
        # Thresh = threshold_otsu(mm[:,i])


    #
    #     # l = LinearRegression()
    #     #
    #     # l.fit(mm[:,i].reshape(-1,1),h)
    #     #
    #     # r2 = l.score(mm[:,i].reshape(-1,1),h)
    #
    #     p = polyfit(h,mm[:,i],1)
    #
    #     # print(p)
    #     #
    #     # print(l.coef_)
    #
    #     hgrid = linspace(min(h),max(h),len(h))
    #
    #     scatter(h,mm[:,i])
    #     # p = [l.coef_[0],l.intercept_]
    #
    #     # plot(hgrid,)
    #     plot(hgrid,polyval(p,hgrid))
    #
    #     r2 = misc.CoeffDet(p,h,mm[:,i])
    #
    #     print(keys[i])
    #
    #     print(r2)
    #
    #     # ax.annotate('y = '+str(p[0])+'x +'+str(p[1]),xy=(0,0))
    #     ax.annotate('R^2 = '+str(r2),xy=(0,10))
    #     # xlabel('Dis-bond Length (mm), Past Root Gap')
    #
    #     xlabel('Dis-bonded Area (mm^2)')
    #
    #
    #
    #     ylabel(keys[i])
    #
    #     savefig('/Users/jlesage/Dropbox/Eclipse/Figures/LMetricFits/Max/'+keys[i]+'Overall.png',dpi=300)
    #
    #     close('all')
    #
    #     Thresh = threshold_otsu(mm[:,i])
    #
    #     R[keys[i]] = {'RSquared':r2,'Coefficients':p,'Value':mm[:,i],'DisbondLengths':h,'AutoThreshold':Thresh}
    #
    #     mth = (mm[:,i]>Thresh).astype(int)
    #
    #     # savetxt('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/'+keys[i]+'Intensities'+str(thresh)+'MaxHitMiss.txt',hstack((mth.reshape(len(mth),1),h.reshape(len(h),1))),delimiter=',')
    #
    # # pickle.dump(R,open('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/MaxMetricModeled.p','wb'))
    #
    # pickle.dump(R,open('/Users/jlesage/Dropbox/Eclipse/DestructiveTesting/L/OverallMetrics.p','wb'))
