import os.path
import os
import numpy as np
import pickle
import SegAnalysis as sa
from scipy.io import savemat
from functools import reduce


def ReadEsndtFile(fld,ds=1,navg=None):
    #fld = 'c:\\temp2\\ascanExtractTest\\I40'
    files = [f for f in os.listdir(fld) if ' pos ' in f]

    d = {}

    for fn in files:
        # pos = int(abs(np.floor(float(fn.split(' pos ')[-1].split('$')[0]))))
        pos = abs(float(fn.split(' pos ')[-1].split('$')[0]))
        d[pos] = ReadEsndtSlice(os.path.join(fld,fn),ds)


    k = list(d.keys())
    k.sort()

    d = [ d[kk] for kk in k]

    L = len(d)

    if type(navg) is int:


        d = [reduce(lambda x,y:x+y,d[i:min([i+navg,L])])/navg for i in range(0,L,navg)]

    return d

def ReadEsndtSlice(fn,ds=1):
    #fn = 'c:\\temp2\\ascanExtractTest\\I40\\I40 pos 5$32,6501'
    numElem, ascanLen = [int(x) for x in os.path.basename(fn).split('$')[-1].split(',',1)]
    a = np.fromfile(fn, dtype='int16')
    ascans = a.reshape((numElem,numElem,ascanLen))
    ascans = ascans[:,:,0::ds]
    return ascans


def Esndt2Pickle(infldr,fsamp,outfl='same',navg = None, ds=1):

    a = ReadEsndtFile(infldr,ds,navg)

    if outfl is 'same':


        outfl = infldr+'/'+infldr.split('/')[-1]+'.p'



    d = {'AScans':a,'SamplingFrequency':fsamp}

    pickle.dump(d,open(outfl,'wb'))


def Esndt2Mat(infldr,outfl=None):

    a = np.array(ReadEsndtFile(infldr))

    if a.shape[0]==1:
        a = a[0,:,:,:]

    if outfl is None:

        outfl = infldr+'/'+infldr.split('/')[-1]+'.mat'

    savemat(outfl,{'AScans':a})

def Pickle2BinaryFolder(infl,outdir):

    a = pickle.load(open(infl,'rb'))['AScans']

    for i in range(len(a)):

        a[i].tofile(outdir+str(i)+'.txt')








# d = ReadEsndtFile('c:\\temp2\\ascanExtractTest\\I40')
# ascan = d[5][10][20] # ascan at position 5 for tx element number 11, rx element number 21

# pth = 'C:/Users/jlesage/Documents/RepeatabilityStudy/'
# f = os.listdir(pth)
#
# # f = [f[0]]
#
# F = sa.FMC(25.)
#
# for ff in f:
#
#     d = ReadEsndtFile(pth+ff)
#
#     for i in range(len(d)):
#
#         F.LoadAScans(d[i],'array')
#
#         F.Calibrate()
#
#         a = F.AScans.astype(np.int16).copy()
#         #
#         a.tofile(open(pth+'/'+ff+'/'+str(i)+'.txt','wb'))
#
#
#
#
# # f = [ff for ff in f if os.path.isdir(pth+ff)]
# # #
# # print(f)
#
# # for ff in f:
# #
# #     # d = ReadEsndtFile(pth+ff)
# #
# #     fn = os.listdir(pth+'/'+ff)[0]
# #
# #     d = ReadEsndtSlice(pth+'/'+ff+'/'+fn)
# #
# #     F = sa.FMC(50.,wedgeid='Black')
# #
# #     F.LoadAScans(d,'array')
# #
# #     F.Calibrate(BPParams=(1.,14.,3.),Offset=0.5)
# #
# #     # a = F.AScans.astype(np.int16)
# #
# #     F.AScans.tofile(open(pth+'/'+ff[1::]+'.txt','wb'))
#
#
#
#
#
#
#
#     # print(ff)
#     # pickle.dump(d,open('C:/Users/jlesage/Dropbox/Eclipse/Controlled/L/'+ff+'.p','wb'))
