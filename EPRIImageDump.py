import _pickle as pickle
import numpy as np
import matplotlib.pylab as plt
from os import listdir

ndtcmap = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/ndtcolourmap.p','rb'))

# pth = '/Volumes/STORAGE/EPRI/'

pth = '/Users/jlesage/Dropbox/Eclipse/EPRI/'

holes = pickle.load(open(pth+'HolePositions.p','rb'))

data = pickle.load(open(pth+'CorrectedScans.p','rb'))

k = list(data.keys())

# print(k)

k = [kk for kk in k if (kk.split('-')[-1]=='Upstream')] #&(kk.split('-')[1]=='PAUT')]

# for kk in k:
#
#     K = kk.split('-')
#
#     if (K[0]=='B1')&(K[2]=='Ref'):
#
#         Iu = data[kk][0][0::2,0::2]
#
#         Id = data['-'.join(K[0:-1])+'-Downstream'][0][0::2,0::2]
#
#         Iref1u = np.amax(Iu[10::,10:-10])
#
#         Iref1d = np.amax(Id[10::,10:-10])
#
#         #
#         #
#         #
#         # Iref1u = np.amax(data[kk][0][20::,20:-20])
#         #
#         # Iref1d = np.amax(data['-'.join(K[0:-1])+'-Downstream'][0][20::,20:-20])
#         #
#         # print(Iref1d)
#
#         fig,ax = plt.subplots(nrows=2)
#
#
#         ax[0].imshow(Iu/Iref1u,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,255.,1.85*25.4,0.])
#
#
#         ax[0].scatter(holes['Block1'][0], holes['Block1'][1], facecolor='none', edgecolor='m')
#
#         ax[1].imshow(Id/Iref1d,cmap=ndtcmap, vmin=0., vmax=1., extent=[0.,255.,1.85*25.4,0.])
#
#
#         ax[1].scatter(holes['Block1'][0], holes['Block1'][1], facecolor='none', edgecolor='m')
#
#         ax[0].set_xlim((0.,255.))
#
#         ax[0].set_ylim((25.4*1.85,0.))
#
#         ax[1].set_xlim((0.,255.))
#
#         ax[1].set_ylim((25.4*1.85,0.))
#
#         ax[0].set_xlabel('X (mm)')
#         ax[0].set_ylabel('Y (mm)')
#
#         ax[0].set_title('Upstream Scan')
#
#         ax[1].set_xlabel('X (mm)')
#         ax[1].set_ylabel('Y (mm)')
#
#         ax[1].set_title('Downstream Scan')
#
#
#         fig.savefig(pth+'Phase2Images/'+'-'.join(kk.split('-')[0:-1])+'.png', dpi=250, bbox='tight', bbox_inches='tight', pad_inches=0)
#
#
#         plt.close()
#
#     elif (K[0]=='B2')&(K[2]=='Ref'):
#
#         Iu = data[kk][0][0::2,0::2]
#
#         Id = data['-'.join(K[0:-1])+'-Downstream'][0][0::2,0::2]
#
#         Iref2u = np.amax(Iu[10::,10:-10])
#
#         Iref2d = np.amax(Id[10::,10:-10])
#
#         fig,ax = plt.subplots(nrows=2)
#
#         ax[0].imshow(Iu/Iref2u,cmap=ndtcmap, extent=[0.,305.,2.0*25.4,0.],vmin=0.,vmax=1.)
#
#         ax[0].scatter(holes['Block2'][0], holes['Block2'][1], facecolor='none', edgecolor='m')
#
#         ax[1].imshow(Id/Iref2d,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,305.,2.0*25.4,0.])
#
#         ax[1].scatter(holes['Block2'][0], holes['Block2'][1], facecolor='none', edgecolor='m')
#
#         ax[0].set_xlim((0.,305.))
#
#         ax[0].set_ylim((25.4*2.0,0.))
#
#         ax[1].set_xlim((0.,305.))
#
#         ax[1].set_ylim((25.4*2.0,0.))
#
#         ax[0].set_xlabel('X (mm)')
#         ax[0].set_ylabel('Y (mm)')
#
#         ax[0].set_title('Upstream Scan')
#
#         ax[1].set_xlabel('X (mm)')
#         ax[1].set_ylabel('Y (mm)')
#
#         ax[1].set_title('Downstream Scan')
#
#
#
#
#         fig.savefig(pth+'Phase2Images/'+'-'.join(kk.split('-')[0:-1])+'.png', dpi=250,bbox='tight', bbox_inches='tight', pad_inches=0)
#
#
#         plt.close()

for kk in k:

    K = kk.split('-')


    if (K[0]=='B2')&(K[2]!='Ref'):


        print(kk)

        Iu = data[kk][0][0::2,0::2]

        Id = data['-'.join(K[0:-1])+'-Downstream'][0][0::2,0::2]

        Iuref2 = np.amax(Iu[60::,:])

        Idref2 = np.amax(Id[60::,:])


        fig,ax = plt.subplots(nrows=2)

        ax[0].imshow(Iu/Iuref2,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,305.,2.0*25.4,-45.])

        # ax[0].imshow(data[kk][0],cmap=ndtcmap, extent=[0.,305.,2.0*25.4,-45.])


        ax[0].scatter(holes['Block2'][0], holes['Block2'][1], facecolor='none', edgecolor='m')

        ax[1].imshow(Id/Idref2,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,305.,2.0*25.4,-45.])

        # ax[1].imshow(data['-'.join(K[0:-1])+'-Downstream'][0],cmap=ndtcmap, extent=[0.,305.,2.0*25.4,-45.])



        ax[1].scatter(holes['Block2'][0], holes['Block2'][1], facecolor='none', edgecolor='m')

        ax[0].set_xlim((0.,305.))

        ax[0].set_ylim((25.4*2.0,-10.))

        ax[1].set_xlim((0.,305.))

        ax[1].set_ylim((25.4*2.0,-10.))

        ax[0].set_xlabel('X (mm)')
        ax[0].set_ylabel('Y (mm)')

        ax[0].set_title('Upstream Scan')

        ax[1].set_xlabel('X (mm)')
        ax[1].set_ylabel('Y (mm)')

        ax[1].set_title('Downstream Scan')




        fig.savefig(pth+'Phase2Images/Unnormalized/'+'-'.join(kk.split('-')[0:-1])+'.png', dpi=250, bbox='tight', bbox_inches='tight', pad_inches=0)


        plt.close()

    elif (K[0]=='B1')&(K[2]!='Ref'):

        Iu = data[kk][0][0::2,0::2]

        Id = data['-'.join(K[0:-1])+'-Downstream'][0][0::2,0::2]

        Iuref1 = np.amax(Iu[60::,:])

        Idref1 = np.amax(Id[60::,:])

        fig,ax = plt.subplots(nrows=2)

        ax[0].imshow(Iu/Iuref1,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,255.,1.85*25.4,-45.])

        # ax[0].imshow(data[kk][0],cmap=ndtcmap, extent=[0.,255.,1.85*25.4,-45.])


        ax[0].scatter(holes['Block1'][0], holes['Block1'][1], facecolor='none', edgecolor='m')

        ax[1].imshow(Id/Idref1,cmap=ndtcmap, vmin=0., vmax=1.,extent=[0.,255.,1.85*25.4,-45.])

        # ax[1].imshow(data['-'.join(K[0:-1])+'-Downstream'][0],cmap=ndtcmap, extent=[0.,255.,1.85*25.4,-45.])


        ax[1].scatter(holes['Block1'][0], holes['Block1'][1], facecolor='none', edgecolor='m')

        ax[0].set_xlim((0.,255.))

        ax[0].set_ylim((25.4*1.85,-10.))

        ax[1].set_xlim((0.,255.))

        ax[1].set_ylim((25.4*1.85,-10.))

        ax[0].set_xlabel('X (mm)')
        ax[0].set_ylabel('Y (mm)')

        ax[0].set_title('Upstream Scan')

        ax[1].set_xlabel('X (mm)')
        ax[1].set_ylabel('Y (mm)')

        ax[1].set_title('Downstream Scan')


        fig.savefig(pth+'Phase2Images/Unnormalized/'+'-'.join(kk.split('-')[0:-1])+'.png', dpi=250, bbox='tight', bbox_inches='tight', pad_inches=0)


        plt.close()
