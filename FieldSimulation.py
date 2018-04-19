
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import griddata
import time
import pickle

from matplotlib import animation



L = 10.
W = 0.6
cp = 5.92*(1-0.01*1j)
cs = 3.24
rho = 7.8
w0 = 2*np.pi*5.
Lx = 60.
Lz = 45.
dx = 0.5
Nx = int(Lx/dx)
Nz = int(Lz/dx)

# print(Nx)
# print(Nz)

# kx,ky,z,w = np.meshgrid(2*np.pi*np.linspace(-1/(2*dx),1/(2*dx),Nx),2*np.pi*np.linspace(-3/L,3/L,10), np.linspace(0,Lz,Nz), 2*np.pi*np.linspace(0,12.5,50))

kx,ky,z,w = np.meshgrid(2*np.pi*np.linspace(-1/L,1/L,40), 2*np.pi*np.linspace(-1/(2*dx),1/(2*dx),Nx),np.linspace(0,Lz,Nz), 2*np.pi*np.linspace(0.,12.5,300))


# print(ky.shape)
# print(z.shape)
# print(w.shape)

#
# F = np.fft.fftshift(np.exp(-0.1*(w-w0)**2)+0j,axes=(3)) # + np.exp(-0.1*(w+w0)**2)-0j

F = np.exp(-0.01*(w-w0)**2)+0j # + np.exp(-0.1*(w+w0)**2)-0j


# plt.plot(w[0,0,0,:]/(2*np.pi),abs(F[0,0,0,:]))

# plt.plot(np.fft.ifft(F[0,0,0,:],F.shape[-1]*2-1).flatten())
# #
# #
# plt.show()

#
# F = F[:,:,:,40]
#



kz = np.sqrt((w/cp)**2 - kx**2 - ky**2 + 0j)

# kz = np.real(kz)

P = L*W*F*np.sinc(0.5*L*ky/np.pi)*np.sinc(0.5*W*kx/np.pi)*np.exp(1j*kz*z)/(rho*(w**2 - 2*(kx**2 + ky**2)*cs**2))


# Field averaged over passive aperture

p = (np.fft.fftshift(np.sum(np.abs(np.sum(np.fft.ifft(P,axis=0),axis=1)),axis=2),axes=(0))).transpose()

# plt.imshow(p)
#
# plt.show()

x = np.linspace(-Lx/2,Lx/2,Nx).reshape(1,-1)

z = np.linspace(0., Lz, Nz).reshape(-1,1)

th = np.arctan2(x,z)*180./np.pi

# print(max(th))

# print(p)

r = np.sqrt(x**2 + z**2)

thgrid,rgrid = np.meshgrid(np.linspace(-60.,60.,100), np.linspace(0.,np.max(z),100))

DR = griddata((th.flatten(),r.flatten()), p.flatten(), (thgrid.flatten(),rgrid.flatten()), method='cubic', fill_value=0., rescale=True).reshape(thgrid.shape)

DR = DR/(np.max(np.abs(DR)))




pickle.dump({'AmplitudeCorrections':DR, 'Angles':np.linspace(-60.,60.,100), 'Range': np.linspace(0.,np.max(z),100), 'Wavespeed':cp, 'ElementDimensions':(W,L)}, open('/Users/jlesage/Dropbox/Eclipse/0p6x10-5MHzProbeCorrections.p','wb'))

# plt.plot(DR[5,:])
# plt.plot(DR[10,:])
# plt.plot(DR[25,:])
# plt.plot(DR[50,:])

# plt.plot(p[:,60])

for i in range(5,DR.shape[0],5):

    plt.plot(DR[i,:])

plt.show()


# plt.imshow(p.transpose())

# plt.plot(p.transpose()[:,60])
#
#
#
# plt.show()


# p = np.fft.fftshift(np.fft.ifft(np.sum(P,axis=1),axis=0),axes=(0))
#
# p = np.fft.fft(p, n=2*p.shape[-1] - 2, axis=-1)

# pmax = np.abs(p[:,:,0])
# # pmax = np.amax(np.abs(p),axis=-1)
#
# # plt.plot(pmax[int(pmax.shape[0]/2),:])
# plt.imshow(np.amax(np.abs(p),axis=-1).transpose())
# #
# plt.show()




# p = np.fft.fftshift(np.fft.ifftn(np.sum(P,axis=1),axes=(0,2),s=(P.shape[0],2*P.shape[3]-1)),axes=(0,2))

# p = np.fft.fftshift(np.fft.ifft(np.sum(np.sum(P,axis=3),axis=1),axis=0),axes=(0))

# print(p.shape)
#
# plt.ion()
#
# # fig = plt.figure()
# #
# # ax = fig.add_subplot(111)
# #
# # I, = ax.imshow(np.abs(p[:,:,0]))
#
#

# fig = plt.figure()
#
# ims = []
#
# for i in range(p.shape[2]):
#     print(i)
#     im = plt.imshow(np.abs(p[:,:,i].transpose()), animated = True)
#     ims.append(im)
#
# ani = animation.FuncAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# plt.show()


# for i in range(p.shape[2]):
#
#     plt.imshow(np.abs(p[:,:,i].transpose()))
#
#     plt.pause(1e-40)
#


#
#
# # print(p.shape)
# # # plt.plot(np.abs(p[250,:]))
# #
# # # plt.plot(np.abs(p[:,0]))
# #
# # # plt.imshow(np.abs(p.transpose()))
# #
# # x,z = np.meshgrid(np.linspace(-Lx/2,Lx/2,p.shape[1]),np.linspace(0,Lz,p.shape[0]))
# #
# # r = np.sqrt(x**2 + z**2)
# #
# # # plt.plot(r[:,0])
# # # plt.plot(r[:,1])
# # #
# # # plt.show()
# #
# # th = np.arctan2(x,z)
# #
# # Th,R = np.meshgrid(np.pi*np.linspace(-40,40,80)/180.,np.linspace(0.,np.amax(r),50))
# #
# # ppolar = griddata(np.hstack((th.reshape(-1,1),r.reshape(-1,1))),np.abs(p.reshape(-1,1)),np.hstack((Th.reshape(-1,1),R.reshape(-1,1))),'linear')
# #
# # ppolar = ppolar.reshape(Th.shape)
# #
# # plt.plot(ppolar)
# #
# #
# #
# # plt.show()
# #
# # # print(np.amax(th))
# #
# # # f = interp2d(th,r,np.abs(p))
# #
# # # print('bah')
# # #
# # # Th = np.pi*np.linspace(-40.,40.,80)/180.
# # # R = np.linspace(0.,np.amax(r),20)
# # #
# # # DR = f(Th,R)
# # #
# # # print(DR.shape)
# # #
# # #
# # # for i in range(DR.shape[0]):
# # #
# # #     plt.plot(Th,DR[i,:])
# # #
# # #
# # #
# # #
# # # plt.show()
