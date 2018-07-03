import numpy as np
from scipy.interpolate import griddata, interp1d
import _pickle as pickle
from misc import DiffCentral, ClosestIndex
from scipy.optimize import brentq
from os import listdir
from Ultravision import ReadScan
# import matplotlib.pylab as plt
from numpy.linalg import norm
import os.path

# pth = '/Volumes/Storage/EPRI/RawData/'

pth = '/Volumes/Storage/EPRI-Phase2-Data/Test/'

ndtcmap = pickle.load(open('/Users/jlesage/Dropbox/Eclipse/ndtcolourmap.p','rb'))

d = listdir(pth)

d = [dd for dd in d if dd.endswith('.txt')]

print(d)


for dd in d:

    print(dd)

    ddd = dd.split('-')

    # Probe,Block,Polymer,UpDown = ddd[0],ddd[1],ddd[2],ddd[3].split('.')[0]

    Block,Probe,Polymer,UpDown = ddd[0],ddd[1],ddd[2],ddd[3].split('.')[0]

    if Block[0]=='.':

        Block = Block[2]


    cs = {'B1': 3.23, 'B2': 3.23}

    cw = {'PAUT': 2.33, 'Conventional': 2.735}

    hw = {'PAUT': 7.76, 'Conventional': 6.5}

    ci = {'ACE400':1.541,'Aqualink100':1.489, 'Aqualene300': 1.585, 'Aqualene320': 1.569, 'Ref':1.5}

    th = {'PAUT': np.pi*np.linspace(40.,70.,31)/180., 'Conventional':np.array([np.pi/4])}

    phi = {'PAUT': np.pi*(38.)/180., 'Conventional': np.pi*(37.)/180.}

    X = {'B1': np.linspace(0.,254.,255), 'B2': np.linspace(0.,305.,306)}


    if Polymer=='Ref':

        H = np.ones(len(X[Block]))*0.01

    else:

        H = pickle.load(open('/Volumes/STORAGE/EPRI-Phase2-Data/BlockSurfaces.p','rb'))[Block][Polymer]



    x = {'B1': np.linspace(0.,254.,len(H)), 'B2': np.linspace(0.,305.,len(H))}


    th = th[Probe]
    ci = ci[Polymer]
    cw = cw[Probe]
    x = x[Block]
    phi = phi[Probe]
    cs = cs[Block]
    X = X[Block]
    hw = hw[Probe]

    # xygrid = {'Block1':np.meshgrid(np.linspace(0.,254.,int(255*2)), np.linspace(0.,7.6+1.85*25.4,int((7.6+1.85*25.4)/0.5))), 'Block2':np.meshgrid(np.linspace(0.,305.,int(306*2)), np.linspace(0.,7.6+2.0*25.4,int((7.6+2.0*25.4)/0.5)))}

    # thw = np.arcsin(cw*np.sin(th)/cs)
    #
    # thi = np.arcsin(ci*np.sin(thw)/cw)

    A = ReadScan(pth+dd)
    #
    # if Probe == 'PAUT':
    #
    #     A = A[:,0::2,:]
    #


        # th = th[0:25]

    thw = np.arcsin(cw*np.sin(th)/cs)

    thi = np.arcsin(ci*np.sin(thw)/cw)

    if UpDown=='Downstream':

        H = H[::-1]

    h = interp1d(x,H,fill_value=0.,bounds_error=False)

    dhdx = interp1d(x,DiffCentral(H)/DiffCentral(x),fill_value=0.,bounds_error=False)

    # dt = 130./A.shape[1]

    # dt = 1.0/(12.5)

    dt = 1/25.


    # print(1/dt)

    # xpts = []
    # ypts = []
    #
    # a = []
    #
    # xxi = []
    #
    # hhi = []
    dx = 1.
    dy = 1.

    if (Polymer=='Reference')&(Block=='B1'):

        I = np.zeros((int(round(1.85*25.4/dy)),int(round(255./dx))))
        II = np.zeros((int(round(1.85*25.4/dy)),int(round(255./dx))))


    elif (Polymer=='Reference')&(Block=='B2'):

        I = np.zeros((int(round(2.*25.4/dy)),int(round(305./dx))))
        II = np.zeros((int(round(2.*25.4/dy)),int(round(305./dx))))


    elif (Polymer!='Reference')&(Block=='B1'):

        I = np.zeros((int(round((H[0]+1.85*25.4)/dy)),int(round(255./dx))))
        II = np.zeros((int(round((H[0]+1.85*25.4)/dy)),int(round(255./dx))))


    elif (Polymer!='Reference')&(Block=='B2'):

        I = np.zeros((int(round((H[0]+2.0*25.4)/dy)),int(round(305./dx))))
        II = np.zeros((int(round((H[0]+2.0*25.4)/dy)),int(round(305./dx))))



    for i in range(len(X)):

        for j in range(len(thw)):

            # Xi = X[i] + hw*(1./np.cos(thw[j]) - 1./np.cos(phi))

            if thw[j]>phi:

                Xi = X[i] + (hw*np.tan(thw[j]) - hw*np.tan(phi))

            else:

                Xi = X[i] - (hw*np.tan(phi) - hw*np.tan(thw[j]))




            def Yi(x):

                return (x - Xi)/np.tan(thi[j])

            def IntersectionEqn(x):

                return Yi(x) - h(x)



            if (Xi>=0)&(Xi<=X[-1]):

                try:

                    xi = brentq(IntersectionEqn,Xi-10.,Xi+10.)

                    T = 2*np.sqrt((xi-Xi)**2 + h(xi)**2)/ci

                            # print(T)
                            #
                            # print(A.shape[1])


                    n = np.array([-dhdx(xi), 1.])

                    vi = np.array([xi-Xi, h(xi)])

                            # print(n)
                            # print(vi)

                    csi = np.dot(n,vi)/(norm(n)*norm(vi))

                            # angarg = np.sqrt(np.abs(1. - (cs**2/ci**2)*(1.-csi**2)))

                    angarg = np.sqrt(np.abs(1. - (cs**2/ci**2)*(1.-csi**2)))

                    hi = h(xi)

                except:

                    hi = np.nan
                    angarg = 2.



                # # T = np.sqrt((xi-X[i])**2 + h(xi)**2)/ci
                #
                # T = 2*np.sqrt((xi-Xi)**2 + h(xi)**2)/ci
                #
                # # print(T)
                # #
                # # print(A.shape[1])
                #
                #
                # n = np.array([-dhdx(xi), 1.])
                #
                # vi = np.array([xi-Xi, h(xi)])
                #
                # # print(n)
                # # print(vi)
                #
                # csi = np.dot(n,vi)/(norm(n)*norm(vi))
                #
                # # angarg = np.sqrt(np.abs(1. - (cs**2/ci**2)*(1.-csi**2)))
                #
                # angarg = np.sqrt(np.abs(1. - (cs**2/ci**2)*(1.-csi**2)))
                #
                #
                # # print(angarg)
                #
                # # ths = np.arcos(1. - (cs**2/ci**2)*(1.-csi**2))
                #
                # # thh = np.arctan(dhdx(xi))
                # #
                # angarg = (cs/ci)*np.sin(thi[j]+thh)

                if (np.isfinite(hi))&(abs(angarg)<1)&(hi>0):


                    # ths = thh - np.arcsin(angarg)

                    ths = np.arccos(angarg)

                    AA = A[j,int(round(T/dt))::,i]

                    r = np.linspace(0.,dt*len(AA),len(AA))*cs/2.


                    xpts = xi+r*np.sin(ths)
                    ypts = h(xi) + r*np.cos(ths)

                    # xpts += list(xi+r*np.sin(ths))
                    # ypts += list(h(xi) + r*np.cos(ths))

                    for p in range(len(xpts)):

                        ix = int(round(xpts[p]/dx))
                        iy = int(round(ypts[p]/dy))

                        if (ix<I.shape[1])&(iy<I.shape[0]):

                            II[iy,ix] = max([AA[p],I[iy,ix]])

                            I[iy,ix] = sum([AA[p],I[iy,ix]])



                    # xpts += list(xi+r*np.sqrt(1-angarg**2))
                    # ypts += list(h(xi) + r*angarg)

                    # xypts.append([xi+r*np.sin(ths), h(xi) + r*np.cos(ths)])
                    #
                    # print(AA.shape)

                    # a += list(AA)

            # else:
            #
            #     AA = A[j,int(round(T/dt))::,i]
            #
            #     r = np.linspace(0.,dt*len(AA),len(AA))*cs/2.
            #
            #     xpts += list(xi+r*np.sin(ths))
            #     ypts += list(h(xi) + r*np.cos(ths))



# plt.plot(X,h(X))
# plt.scatter(np.array(xxi),np.array(hhi))
#
# plt.show()

    # #
    # xypts = np.hstack((np.array(xpts).reshape(-1,1), np.array(ypts).reshape(-1,1)))
    # a = np.array(a)


    # plt.imshow(I,cmap=ndtcmap)
    #
    # plt.show()



    #
    # # print(xypts.shape)
    # # print(a.shape)
    #
    # I = griddata(xypts, a,  np.hstack((xygrid[Block][0].reshape(-1,1),xygrid[Block][1].reshape(-1,1))), fill_value=0.,method='linear').reshape(xygrid[Block][0].shape)
    #

    if UpDown=='Downstream':

        I = I[:,::-1]

    # plt.imshow(I,cmap=ndtcmap)
    #
    # plt.show()


    try:

        f = pickle.load(open(pth+'CorrectedScans.p','rb'))
        f[dd.strip('.txt')] = (I,II)
        pickle.dump(f,open(pth+'CorrectedScans.p','wb'))

    except:

        f = {}
        f[dd.strip('.txt')] = (I,II)
        pickle.dump(f,open(pth+'CorrectedScans.p','wb'))


    # plt.imshow(I,cmap=ndtcmap)
    #
    # plt.show()
    #
