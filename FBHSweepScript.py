import pickle
from numpy import *
from matplotlib.pylab import *
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
from misc import moments
from functools import reduce
import os
import SegmentAnalysis as sa


from skimage.filters import threshold_li

# pth = 'C:/Users/jlesage/Documents/POD-FBHS-5MHZ/L/FBH/AScans/Images/'
#
#
# d = os.listdir(pth)
#
# d = [dd for dd in d if not(dd.endswith('.p'))]
#
# m = {}
#
# for dd in d:
#
#     I = si.Weld(pth+dd+'/')
#
#     print(dd[0:-10])
#
#     I.DefineSubRegions()
#
#     m[dd[0:-10]] = {}
#
#     m[dd[0:-10]]['MaxDeviation'] = I.SubRegions[0]['MaxDeviation'][5:-5]
#
#
#
# mtr = reduce(lambda x,y: concatenate((y,x)), [m[k]['MaxDeviation'] for k in m.keys()])
#
# mtrthresh = threshold_li(mtr)
#
#
# for k in list(m.keys()):
#
#     m[k]['HitMiss'] = [any(m[k]['MaxDeviation'][8:18]<=mtrthresh),any(m[k]['MaxDeviation'][52:63]<=mtrthresh)]
#
#     x = linspace(5,len(m[k]['MaxDeviation'])-5,len(m[k]['MaxDeviation']))
#
#     plot(x,m[k]['MaxDeviation'])
#     plot(mtrthresh*ones(len(m[k]['MaxDeviation'])))
#
#     xlabel('Axial Location (mm)')
#     ylabel('Indication Length (mm)')
#     legend(['Metric','Threshold'],loc='best')
#
#
#     savefig(pth+str(k)+'.png',dpi=250,box_inches='tight')
#
#     close('all')
#
#
#
#
# m['Threshold'] = mtrthresh
#
# pickle.dump(m,open(pth+'/FBHMaxDeviations.p','wb'))
#
#
#








# pth = '/Volumes/STORAGE/FBH/'
#
# pth = 'G:/DestructiveResults/'

# pth = 'G:/FBH/'

# pth = '/Volumes/STORAGE/DestructiveResults/AScans/'

pth = 'C:/Users/jlesage/Documents/POD-FBHS-5MHZ/L/FBH/Test/'

d = os.listdir(pth)

# d = d[0:4]+d[6::]

# Xtfm = []
# Xsw = []

# dl = pickle.load(open('/Volumes/STORAGE/DisbondLengths.p','rb'))


# dl = pickle.load(open('G:/DisbondLengths.p','rb'))



# D = {}

# SW = []
#
# TFM0 = []
#
# TFM1 = []
#
# TFM2 = []

d = [dd for dd in d if dd.endswith('.p')]


ps = {}

SS = []

SWn = []

for dd in d:

    f = sa.FMC(pickle.load(open(pth+dd,'rb'))['AScans'][5:-5])

    f.Calibrate()

    f.Sweep(linspace(-2.5,2.5,10),linspace(-2.5,2.5,10),(2.5,4.5))

    # sw = f.SweepPeakSum
    #
    # th = threshold_li(sw)
    #
    # s0 = std(sw[sw<th])
    # mu0 = mean(sw[sw<th])
    #
    # swn = (sw-mu0)/s0
    #
    SWn.append(f.SweepMaxStackSum)

    SS.append(f.SweepMaxStack)


    # ps[dd.strip('.p')] = {'AScans':f.AScans, 'Reference':f.Reference}


    # # f = pickle.load(open(pth+dd,'rb'))
    #
    # # sw = f['Sweep']
    #
    # # sw = (sw - mean(sw,axis=0)).transpose()
    #
    # # sw = sw-mean(sw,axis=0)
    # #
    # # # D[dd[0:3]] = {}
    # # # D[dd[0:3]]['Sweep'] = sw
    # # #
    # # # D[dd[0:3]]['DisbondLengths'] = dl[dd[0:3]]
    # #
    # # sw = sw[:,750:1250]
    # #
    # #
    # # sw = sum(sw*(sw>0),axis=1)
    #
    #
    #
    #
    #
    # tfm = [array(ff)[:,:,0] for ff in f['FusionLine']]
    #
    #
    # tfm0 = sum(tfm[0],axis=1)
    #
    # tfm1 = sum(tfm[1],axis=1)
    #
    # Xtfm += [[tfm0[i],tfm1[i]] for i in range(10,len(tfm0)-10)]
    #
    # tfm2 = sum(tfm[2],axis=1)
    #
    #
    # tfm01 = sum(tfm[0]+tfm[1],axis=1)
    #

    #
    # fsw,axsw = subplots(nrows=1,sharex=True)
    #
    # fsw.suptitle(dd[0:3])
    #
    # axsw[0].plot(dl[dd[0:3]])
    #
    # axsw[0].set_title('Disbond Line')
    #
    # plot(sw)
    #
    # title('Sum of Synthetic Sweep (Above Axial Mean)')
    #
    # xlabel('Axial Position (mm)')
    #
    # savefig(pth+dd[0:3]+'Sweep.png',dpi=250,box_inches='tight')
    #
    # close('all')
    #
    # ftfm,axtfm = subplots(nrows=3,sharex=True)
    #
    # fsw.suptitle(dd[0:3])
    #
    # axtfm[0].plot(dl[dd[0:3]])
    #
    # axtfm[0].set_title('Disbond Line')
    #
    # axtfm[0].plot(tfm0)
    #
    # axtfm[0].set_title('TTTLTT')
    #
    # axtfm[1].plot(tfm1)
    #
    # axtfm[1].set_title('TLTTLL')
    #
    # axtfm[2].plot(tfm01)
    #
    # axtfm[2].set_title('TTTLTT + TLTTLL')
    #
    # xlabel('Axial Position (mm)')
    #
    # savefig(pth+dd[0:3]+'TFM.png',dpi=250,box_inches='tight')
    #
    # close('all')

# SWn = reduce(lambda x,y: concatenate((y,x)),[s for s in SWn])

# pickle.dump(ps,open(pth+'CalibratedDestructiveScans.p','wb'))

# plot(f.Reference[10])
# plot(f.Reference[20])
#
# show()



th = threshold_li(reduce(lambda x,y: concatenate((y,x)),[s for s in SWn]))

SW = {}
SW['Threshold'] = th

for i in range(len(d)):

    k = d[i].strip('.p')



    SW[k] = {'SweepStack':SS[i],'PeakSum':SWn[i]}
    #
    #

    f,ax = subplots(nrows = 5,sharex=True)
    #
    f.suptitle(k)

    ax[0].plot(SWn[i],marker='o')

    ax[0].set_title('Sweep Stack Sum (Relative To Background)')

    ax[1].plot(SWn[i]>=2.,marker='o')

    ax[1].set_ylim((-0.1,1.1))


    ax[1].set_title('Hit/Miss (2 Standard Deviations Above Background)')

    ax[2].plot(SWn[i]>=3.,marker='o')

    ax[2].set_ylim((-0.1,1.1))


    ax[2].set_title('Hit/Miss (3 Standard Deviations Above Background)')

    ax[3].plot(SWn[i]>=4.,marker='o')

    x[3].set_ylim((-0.1,1.1))


    ax[3].set_title('Hit/Miss (4 Standard Deviations Above Background)')

    ax[4].imshow(SS[i],aspect=0.3)

    ax[4].set_title('Sweep Stack')


    # ax[1].set_title('Hit/Miss')
    #
    # ax[1].set_ylim((-0.1,1.1))
    #
    # ax[2].plot(SWn[i],marker='o')
    #
    # ax[2].plot(th*ones(len(SWn[i])))
    #
    # ax[2].set_title('Sweep Stack Sum (Relative To Background)')
    #
    # ax[3].imshow(SS[i],aspect=0.5)
    #
    # ax[3].set_title('Sweep Stack')
    #
    xlabel('Axial Position (mm)')
    #
    savefig(pth+k+'SweepStack.png',dpi=250,box_inches='tight')

    close('all')


pickle.dump(SW,open(pth+'SweepStackData.p','wb'))


# gm = GaussianMixture(n_components=2)
#
# gm.fit(array(Xtfm))
#
# y = array(gm.predict(Xtfm))
#
# # print(y)
#
# scatter(array(Xtfm)[y==0,0],array(Xtfm)[y==0,1],color='blue',alpha=0.1)
#
# scatter(array(Xtfm)[y==1,0],array(Xtfm)[y==1,1],color='red',alpha=0.1)
#
#
# savefig(pth+'TFMScatterFBH.png',dpi=250)
#
# close('all')

# for dd in d:
#
#     f = pickle.load(open(pth+dd,'rb'))
#
#
#
#
#     tfm = [array(ff)[:,:,0] for ff in f['FusionLine']]
#
#
#
#     tfm0 = sum(tfm[0],axis=1)
#
#     L = len(tfm0)
#
#     tfm0 = tfm0[10:len(tfm0)-10]
#
#     tfm1 = sum(tfm[1],axis=1)
#
#     tfm1 = tfm1[10:len(tfm1)-10]
#
#     X = hstack((tfm0.reshape((len(tfm0),1)),tfm1.reshape((len(tfm1),1))))
#
#     # fig,ax = subplots(nrows=3,sharex=True)
#     #
#     # fig.suptitle(dd[0:3])
#     #
#     # ax[0].plot(dl[dd[0:3]][10:L-10],marker='o')
#     #
#     # ax[0].plot(2.0*ones(L-20))
#     #
#     # ax[0].set_title('Disbond Line + Threshold')
#     #
#     # ax[1].plot(((dl[dd[0:3]][10:L-10])>2.).astype(int),marker='o')
#     #
#     #
#     #
#     # ax[1].set_title('Disbond Hit/Miss')
#     #
#     # ax[1].set_ylim((-0.1,1.1))
#     #
#     # ax[2].plot(gm.predict(X),marker='o')
#     #
#     # ax[2].set_title('Metrics Hit/Miss')
#     #
#     # ax[2].set_ylim((-0.1,1.1))
#
#     plot(gm.predict(X),marker='o')
#
#     title(dd[0:3])
#
#     ylim((-0.1,1.1))
#
#
#     xlabel('Axial Position (mm)')
#
#     savefig(pth+dd[0:3]+'TFMBinary.png',dpi=250,box_inches='tight')
#
#     close('all')







    # TFM = zeros((30,len(sw)+10))
    #
    # for i in range(len(tfm)):
    #
    #     T = (tfm[i]-mean(tfm[i],axis=0)).transpose()
    #     plot(T)
    #     show()
    #     T = T*(T<0)
    #
    #     TFM += TFM


    # tfm = reduce(lambda x,y: x+(y-mean(y,axis=0)).transpose(), tfm, zeros((30,len(sw)+10)))

    # tfm = reduce(lambda x,y: x+y.transpose(), tfm, zeros((30,len(sw)+10)))


    # D[dd[0:3]]['FusionLine'] = tfm

    # D[dd[0:3]]['FusionLine'] = TFM


    # x = linspace(0.,6.,30)
    #
    # Xsw += list(sw)
    #
    # D[dd[0:3]]['SweepMetrics'] = sw
    #

    # tfm = [[moments(tfm[0:30,i],x)[0],moments(tfm[0:30,i],x)[1],sqrt(moments(tfm[0:30,i],x,0.)[2])/2] for i in range(5,tfm.shape[1]-5)]

    # tfm = [[moments(tfm[0:30,i],x)[0],moments(tfm[0:30,i],x)[1]] for i in range(5,tfm.shape[1]-5)]


    # TFM = [[moments(TFM[0:30,i],x)[0],moments(TFM[0:30,i],x)[1],sqrt(moments(TFM[0:30,i],x,0.)[2])/2] for i in range(5,TFM.shape[1]-5)]


    # D[dd[0:3]]['TFMMetrics'] = tfm

    # Xtfm += tfm

    # Xtfm += TFM



# Xsw = array(Xsw).reshape((len(Xsw),1))
#
#
# Xtfm = array(Xtfm)
#
# plot(Xtfm[:,0])
# show()
#
#
# ksw = KMeans(n_clusters=2, random_state=0).fit(Xsw)
#
# ktfm = KMeans(n_clusters=2, random_state=0).fit(Xtfm)
#
# for k in D.keys():
#
#
#     D[k]['SweepPredictions'] = ksw.predict(array(D[k]['SweepMetrics']).reshape((len(D[k]['SweepMetrics']),1)))
#
#     D[k]['TFMPredictions'] = ktfm.predict(D[k]['TFMMetrics'])
#
#
# # D['TFMClusters'] = ktfm
# D['SweepClusters'] = ksw
#
# pickle.dump(D,open('/Volumes/STORAGE/DestructiveSweepTFMClusters.p','wb'))
#
#







# import tensorflow as tf
#
# # Model parameters
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x:x_train, y:y_train})
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# l = pickle.load(open('/Volumes/STORAGE/CIVAAScans/CIVAAScansTh275.p','rb'))
#
#
# GetSlice = lambda d,v,h: [ ll for ll in l if ((ll['WeldParameters']['VerticalLOF']==v))&(int(ll['WeldParameters']['HorizontalLOF']==h))&(int(ll['DisbondLength']==d))][0]['AScans']
#
# vfl = linspace(5.,7.,3)
# hfl = linspace(3.,6.,4)
# D = linspace(0.,5.,6)
#
# # print(GetSlice(0.,5.,3.))
#
# # vfl = linspace(5.,6.,1)
# # hfl = linspace(3.,4.,1)
# # D = linspace(0.,3.,4)
#
# for v in vfl:
#
#     h = [h for h in hfl if h<v]
#
#     for hh in h:
#
#         s = [GetSlice(dd,v,hh) for dd in D if dd<hh]
#
#         dd = [dd for dd in D if dd<hh]
#
#         dd = dd[1::]
#
#         N = (len(s)-1)*5+10*len(s)
#
#         w = [s[0]]*N
#
#         dl = zeros(len(w))
#
#         s = s[1::]
#
#
#         for n in range(len(s)):
#
#             w[9+n*15:9+n*15+5] = [s[n]]*5
#
#             dl[9+n*15:9+n*15+5] = dd[n]
#
#         pickle.dump({'AScans':w,'DisbondLengths':dl},open('/Volumes/STORAGE/Th275_VFL_'+str(int(v))+'_HFL_'+str(int(hh))+'.p','wb'))
