import numpy as np
from skimage.io import imread
from skimage.filters import threshold_li
from skimage.measure import moments, moments_central

I = imread('/Users/jlesage/Dropbox/AirLightBoltProject/OrangeBoltsRef2.jpg')

Ibinary = I[:,:,2]>threshold_li(I[:,:,2])
Ibinary = Ibinary.astype(float)

m = moments(Ibinary)

r = np.sqrt(m[0,0]/np.pi)

R = int(r*1.1)

Ly,Lx = Ibinary.shape

c = (int(round(m[0,1]/m[0,0])), int(round(m[1,0]/m[0,0])))

I = I[max[0,c[0]-r], min[0,c[0]+r], max[0,c[0]-r], min[0,c[0]+r], :]

Ibinary = I[:,:,2]>threshold_li(I[:,:,2])
Ibinary = Ibinary.astype(float)

Ly,Lx = Ibinary.shape

x,y = np.meshgrid(np.linspace(0,Lx-1,Lx), np.linspace(0,Ly-1,Ly))
