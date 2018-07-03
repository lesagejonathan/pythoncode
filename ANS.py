import numpy as np

class PipeScan():


def ANSDefects(ang,z,T,h=8.5,od=3.5):

    l1 = 6.25

    c = 3.24

    rang = 60.*(np.pi/180.)

    l2 = h*np.tan(rang)

    d = h/np.cos(rang)

    if od<10:

        R = od*25.4/2.

    else:

        R = od/2.

    CircumferentialLength = np.array([ang[0],ang[1]])*(2*np.pi/160.)*R

    if T<=2*d/c:

        Depth = c*T*np.cos(rang)/2.0

        OffCentreLinePosition = l2 + (z-c*T*np.sin(rang)/2.)

    else:

        dd = (T-2*d/c)*c/2.

        Depth = h - dd*np.cos(rang)

        OffCentreLinePosition = z - dd*np.sin(rang)

    return CircumferentialLength,Depth,OffCentreLinePosition
