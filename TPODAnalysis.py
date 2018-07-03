from os import listdir
from os.path import isfile, join
import Eclipse as ec
from numpy import ones
import matplotlib.pyplot as plt
import misc


# pth = '/Users/jlesage/Dropbox/Eclipse/POD-FBHS1-T-Images/'

pth = '/Users/jlesage/Dropbox/Eclipse/PODImages/'


dirs = listdir(pth)



prfl=[f.strip('.png').split('-')[-1] for f in listdir(pth) if f.endswith('.png')]


if len(prfl)>0:

    dirs = [d for d in dirs if (not(d.endswith('.png')))&(not(d.split('/')[-1] in prfl))&(d.split('/')[-1]!='.DS_Store')&(not(d.endswith('.csv')))]

else:

    dirs = [d for d in dirs if (d.split('/')[-1]!='.DS_Store')&(not(d.endswith('.csv')))]


# print(dirs)


# resultsfl = pth.split('/')
# resultsfl = pth+resultsfl[-4]+'-'+resultsfl[-3]+'-'+resultsfl[-2]+'.csv'
#
# if isfile(resultsfl):
#     open(resultsfl, 'w')

s = lambda x: misc.AutoThreshold(x)[1][0]
b = lambda x: misc.BimodalityCoefficient(x)


for d in dirs:

    fl = ec.FMCImage(pth+d+'/',[[3.,45.],[5.,35.]])

    flname = (pth+d).split('/')
    flname = flname[-1]


    # fl.SmoothImages(5.5)

    fl.AxialStack()

    fl.RectangularCropImages()

    fl.GetBinaryImages()
    #
    # S = fl.GetMetric(s)
    # B = fl.GetMetric(b)


    fl.HorizontalStack()



    fig,ax = plt.subplots(3,sharex=True)

    # ax[0].axis('off')
    ax[0].set_title('Side View')
    # ax[1].axis('off')
    ax[1].set_title('Binary Side View')

    ax[2].set_title('Bimodality Coefficient')

    ax[0].imshow(fl.HStack,extent=[fl.Region[2,0],fl.Region[2,1],fl.Region[1,1],fl.Region[1,0]])

    ax[1].imshow(fl.BinaryHStack,cmap='gray',extent=[fl.Region[2,0],fl.Region[2,1],fl.Region[1,1],fl.Region[1,0]])


    # ax[1].plot(S)
    # ax[1].plot(1.5*ones(S.shape))
    # ax[1].plot(2*ones(S.shape))
    # ax[1].plot(2.5*ones(S.shape))


    ax[2].plot(fl.BimodalityCoefficient)
    ax[2].plot((5/9)*ones(fl.BimodalityCoefficient.shape))



    plt.ion()

    plt.show()

    # ans = input('Enter Weld Classifcation - 0 for None, 1 for Minor, 2 for Major, 3 for Poor Data: ')

    # fig.xlabel(flname+' - '+anskey[ans])
    fig.savefig(pth+flname+'.png',format='png')

    plt.close(fig)

    del(fl)


    # open(resultsfl,'a').write(flname+','+anskey[int(ans)]+'\n')
