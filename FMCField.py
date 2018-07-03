import numpy as np
import Civa
from functools import reduce
import matplotlib.pylab as plt

H = Civa.LoadAScansFromTxt('/Users/jlesage/Dropbox/Eclipse/Notch2mmInBlock_0_0.txt')

c = 5.9
p = 0.6
rho = 7.9

M = rho*c**2
L = rho*(c**2 - 2*(3.24)**2)

# H = np.fft.fft(H,axis=2)

H = np.fft.rfft(H,axis=2)


N = H.shape[0]

xn = np.linspace(-N*p/2,N*p/2,N)
xn = xn.reshape((N,1))

f = np.linspace(0.,12.5,H.shape[2])

# f = np.linspace(-12.5,12.5,H.shape[2])

w = np.pi*2*f

# print(len(w))
#
# w = w[30:70]

# w = w[1::]


# w = w[5::]

# if5 = np.argmin(np.abs(f-5.))
#
# H = H[:,:,if5]
# w = 2*np.pi*f[if5]

kx = 2*np.pi*np.linspace(-1/(2*p),1/(2*p),32)
kx = kx.reshape(1,-1)

# F = 1.+0j

F = (np.exp(-0.01*(w-np.pi*2*5.)**2) + 0j) #+ np.exp(-0.01*(w+np.pi*5.)**2)

# plt.plot(np.fft.fftshift(np.real(np.fft.ifft(F,n=2*len(F)-1))))
#
# plt.show()

# plt.plot(np.abs(F))
# plt.show()


def SolveContactCoeffs(H,F,w,m):

    kz = np.sqrt((w/c)**2 - kx**2 + 0j)

    # kz = kz[np.abs(np.imag(kz))<1e-16]


    # kz = np.real(kz)

    # kz = kz.reshape((1,N))
    # kx = kx.reshape((1,N))

    H = H.reshape(-1,1)

    # xn = np.linspace(-N*p/2,N*p/2,N)
    # xn = n.reshape((N,1))

    a = -(2*kz/kx)*(np.exp(1j*kx*(xn+0.5*p)) - np.exp(1j*kx*(xn-0.5*p)))

    b = H-np.sum(0.5*p*np.sinc(0.25*p*kx/np.pi)*np.exp(-1j*kx*xn[m])*F*(np.exp(1j*kx*(xn+0.5*p)) - np.exp(1j*kx*(xn-0.5*p)))/(-M*kz**2 - L*kx**2),axis=1).reshape(-1,1)



    return np.linalg.solve(a,b)

def DisplacementField(w,m,F,B,zrng):

    z = np.linspace(zrng[0],zrng[1],int((zrng[1]-zrng[0])/zrng[2]))

    z = z.reshape(-1,1)

    # print(len(kx))

    Nx = int(np.round(N*p/zrng[2])) - N

    # Nx = int(np.ceil(p/zrng[2])/2) - N

    # print(Nx)

    kz = np.sqrt((w/c)**2 - kx**2 + 0j)

    # print(np.sign(np.real(kz)))

    # print(w)
    # print(np.imag(kz))

    # kz = kz[np.abs(np.imag(kz))<1e-16]

    # kz = np.real(kz) + 0j

    B[:,np.abs(np.imag(kz.ravel()))>0.] = 0+0j


    A = -0.5*F*p*np.sinc(0.25*p*kx/np.pi)*np.exp(-1j*kx*xn[m])/(-M*kz**2 - L*kx**2) - B

    A[:,np.abs(np.imag(kz.ravel()))>0.] = 0+0j

    Ux = np.fft.fftshift(1j*kx*(A*np.exp(1j*kz*z) + B*np.exp(-1j*kz*z)),axes=(1,))

    # Ux = np.fft.fftshift(A*np.exp(1j*kz*z) + B*np.exp(-1j*kz*z),axes=(1,))


    xpad = np.zeros((len(z),Nx),dtype=complex)

    indxhalf = int(np.floor(len(kx)/2))


    Ux = np.hstack((Ux[:,0:indxhalf],xpad,Ux[:,indxhalf+1:-1]))

    Uz = 1j*kz*(A*np.exp(1j*kz*z) - B*np.exp(-1j*kz*z))

    Uz = np.hstack((Uz[:,0:indxhalf],xpad,Uz[:,indxhalf+1:-1]))

    ux = np.fft.ifft(Ux,axis=1)

    uz = np.fft.ifft(Uz,axis=1)

    # print(ux.shape)


    return np.hstack((ux.reshape((ux.shape[0]*ux.shape[1],1)),uz.reshape((uz.shape[0]*uz.shape[1],1))))

ushape = (75,94)

# B = np.array([SolveContactCoeffs(H[m,:],F,w,m) for m in range(int(N/2))])


# u = reduce(lambda x,y: x+y, [DisplacementField(ww,m,B[m].reshape((1,len(B[m]))),(0.,20.,0.1)) for m in range(int(N/2))])

u = [reduce(lambda x,y: x+y, [DisplacementField(w[i],m,F[i],SolveContactCoeffs(H[m,:,i],F[i],w[i],m).reshape(1,-1),(0.,15.,0.2)) for m in range(32)]) for i in range(len(w))]

u = np.array(u)

print(u.shape)

u = np.fft.ifft(u,n=2*len(w)-1,axis=0)

# u = np.abs(np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)).reshape((2*len(w)-1,ushape[0],ushape[1]))

u = np.abs(u[:,:,1]).reshape((2*len(w)-1,ushape[0],ushape[1]))



u = u[85,:,:]

# u = np.amax(u,axis=0)

plt.imshow(u)

# fig,ax = plt.subplots(nrows=2)
#
# # ax[0].imshow(np.abs(np.sqrt(u[:,0]**2 + u[:,1]**2)).reshape(ushape))
# ax[0].imshow(np.abs(u[:,0]).reshape(ushape))
# ax[1].imshow(np.abs(u[:,1]).reshape(ushape))


# plt.plot(np.abs(np.sqrt(u[:,0]**2 + u[:,1]**2)).reshape(100,190)[:,int(190/2)])

plt.show()
