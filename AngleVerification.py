import numpy as np
import Ultravision
import os
from scipy.signal import hilbert
import pickle

pth = '/Users/jlesage/Dropbox/Eclipse/ANSFeederTubeProject/WedgeParameterVerification/'

d = os.listdir(pth)

print(d)

d = [dd for dd in d if dd.endswith('.txt')]

n = np.linspace(0.,15.,16)

apdv = {}

for dd in d:

    s = Ultravision.ReadScan(pth+dd)

    s = s.reshape((32,32,2504))

    s1 = s[0:15,0:15,:]
    s1 = s[::-1,::-1,:]

    s2 = s[16::,16::,:]
    s2 = s[::-1,::-1,:]

    s1 = hilbert(s1)
    s2 = hilbert(s2)

    T1 = np.array([ 200 + np.argmax(abs(s1[i,i,200::])) for i in range(16) ]).astype(float)/100.

    T2 = np.array([ 200 + np.argmax(abs(s2[i,i,200::])) for i in range(16) ]).astype(float)/100.

    sphi1 = np.polyfit(n/2.33,T1,1)[0]
    sphi2 = np.polyfit(n/2.33,T2,1)[0]

    phi1 = np.arcsin(sphi1)*180./np.pi
    phi2 = np.arcsin(sphi2)*180./np.pi

    v1 = T1-T1[0]-2*n*0.5*sphi1/2.33
    v2 = T2-T2[0]-2*n*0.5*sphi2/2.33

    apdv[dd.split('_')[1].strip('.txt')]={'Angle':(phi1,phi2),'ProbeDelayVariation':(v1,v2)}

pickle.dump(apdv, open(pth+'AngleProbeDelayVariation.p' ,'wb'))
