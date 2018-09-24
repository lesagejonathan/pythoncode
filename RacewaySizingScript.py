import numpy as np
import _pickle as pickle
from scipy.stats import linregress


d = {}

n = 2

Tw = 25.

s = 37.82

# k = ['B','C','D','E','F','G','I','J','K','L']

k = ['B','C']


d['A']={'Diameter':0.5,'Depth':1.}
d['B']={'Diameter':0.5,'Depth':1.5}
d['C']={'Diameter':1.,'Depth':0.5}
d['D']={'Diameter':1.,'Depth':1.}
d['E']={'Diameter':1.,'Depth':1.5}
d['F']={'Diameter':1.,'Depth':2.}
d['G']={'Diameter':1.5,'Depth':0.5}
d['H']={'Diameter':1.5,'Depth':1.}
d['I']={'Diameter':1.5,'Depth':1.5}
d['J']={'Diameter':1.5,'Depth':2.}
d['K']={'Diameter':1.5,'Depth':2.5}
d['L']={'Diameter':1.5,'Depth':3.}


for kk in k:

    print(kk)

    inpt = input('Detected ?')

    if inpt == 'y':

        d[kk]['Detected'] = True

        ms = []
        c = []

        for i in range(n):

            print(str(i)+'\n')

            T1 = float(input('Channel 1 TOF'))
            T2 = float(input('Channel 2 TOF'))
            T3 = float(input('Channel 3 TOF'))

            cc = s/(T3 - Tw)

            l = cc*(T3 - 0.5*(T1+T2))

            print(cc)
            print(l)

            ms.append(l)
            c.append(cc)

        d[kk]['Velocity'] = np.array(c)
        d[kk]['MeasuredSize'] = np.array(ms)

    else:

        d[kk]['Detected'] = False

pickle.dump(d,open('/Users/jlesage/Dropbox/Eclipse/Raceway/SizingResults.p','wb'))


ms = np.array([0.])
dia = []

for kk in k:

    if d[kk]:

        m = d[kk]['MeasuredSize']
        ms = np.concatenate((ms,m))

        dia.append(d[kk]['Diameter'])


ms = ms[1::]
dia = np.array(dia)

r = []

for i in range(n):

    r.append(linregress(dia[i],ms[i::n]))

pickle.dump({'Regression':r},open('/Users/jlesage/Dropbox/Eclipse/Raceway/SizingRegression.p','wb'))
