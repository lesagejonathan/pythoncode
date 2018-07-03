import numpy as np
from matplotlib.pylab import plot,show

p = 7.8
c = 5.92

v = 0.0

f = np.linspace(0.,12.5,1000)
f = f.reshape(-1,1)

d = np.linspace(0.,2.,1000)
d = d.reshape(1,-1)

Z = p*c


pp = 15.63*v + (1-v)*1.0

cc = np.sqrt((1.0*(1.5)**2)/pp)

ZZ = pp*cc

zeta = 0.5*np.abs(ZZ/Z - Z/ZZ)

print(zeta**2)


# T = np.trapz(np.exp(-3.0*(f-2.)**2)/(np.sqrt((zeta**2)*np.sin(2*np.pi*f*d/cc)**2 + 1)),dx=f[1]-f[0],axis=0)*12.5

f = 2.25

# d = 0.5

T = 1./(np.sqrt((zeta**2)*np.sin(2*np.pi*f*d/cc)**2 + 1))

# T = np.exp(-2.0*(f-2.)**2)*T




#
# print(d.shape)
# print(T.shape)
#
plot(d.ravel(),T.ravel())

# plot(f.ravel(),T[
show()
