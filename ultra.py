def nextpow2(n):
	from numpy import log2, ceil
	N=2**int(ceil(log2(n)))
	return N

def GetData(navg=2,nloc=1,chnum=1):
    import visa
    from numpy import linspace,array,mean
    
    rm=visa.ResourceManager()
    s=rm.open_resource('USB0::0x0957::0x1797::MY52160825::INSTR')

    while (chnum!=1) and (chnum!=2):
        chnum=input('Input a valid channel number (1 or 2) :')

    s.write('*CLS')
    
    s.write(':stop')
    

    s.write(':waveform:format ascii')
    s.write(':waveform:source channel'+str(int(chnum)))
    s.write(':waveform:points:mode maximum')
    
    X=[]
    T=[]
    
    for i in range(nloc):
        
        s.write(':acquire:type normal')
        s.write(':run')
        
        print('Location '+str(i+1))
        raw_input('Press Any Key to Collect Signal')
        
        s.write(':acquire:type average')
        s.write(':acquire:count '+str(nextpow2(navg)))
	
        s.write(':digitize channel'+str(int(chnum)))
        xst=s.query(':waveform:data?')
        N=float(s.query(':waveform:points?'))
        t0=float(s.query(':WAVeform:XOrigin?'))
        dt=float(s.query(':waveform:xincrement?'))
        t=linspace(t0,dt*N+t0,N)
        xst=s.query(':waveform:data?')
        xst=xst[10::].split(',')
        x=[float(xx) for xx in xst]
        x=array(x)
        x=x-mean(x)
        T.append(t)
        X.append(x)
        
        
    s.write(':acquire:type normal')
    s.write(':run')
    s.clear()
    return T,X

def GetWavespeedAttenuation(t,x,d):

    from numpy import log
    from scipy.signal import hilbert
    from spr import pkfind
    
    c=[]
    alpha=[]
	
    for i in range(len(d)):
        
        tp,xp=pkfind(t[i],abs(hilbert(x[i])),2)
        
        c.append(2*1e-6*d[i]/(tp[1]-tp[0]))
        
        alpha.append(-log(xp[1]/xp[0])/(2*d[i]))
        
    return c,alpha
     
def SignalIndex(x,indmax,sigma,thresh=1.):
    
    indleft=indmax
    indright=indmax
    
    SNR=x[indmax]/sigma    
   
    while SNR>thresh:
        
        indleft -=1
        
        SNR=x[indleft]/sigma
    
        
    SNR=x[indmax]/sigma
        
    while SNR>thresh:
        
        indright +=1
        
        SNR=x[indright]/sigma
        
        
    return indleft,indright
    
    
def GumbleFit():
    
    
    

# def fsweep(frange,navg,flname,ctype,ch=1,delaytime=0.5,overwrite=False,cycles=10,vamp=5.0):
#     import visa
#     from numpy import linspace,array,floor
#     import os
#     from time import sleep
#
#     while (navg<2) or (navg>65536):
#             navg=input('Input a number of averages between 2 and 65536 :')
#
#     f = 1e6*linspace(frange[0],frange[1],floor((frange[1]-frange[0])/frange[2])+1)
#
#     s.write('*CLS')
#     s.write(':wgen:function sinusoid')
#     s.write(':wgen:voltage:high '+str(vamp/2))
#     s.write(':wgen:voltage:low '+str(-vamp/2))
#     s.write(':acquire:type average')
#     s.write(':acquire:count '+str(nextpow2(navg)))
#     s.write(':wgen:output 1')
#     s.write(':trigger:source wgen')
#
#         s.write(':waveform:source channel'+str(ch))
#         s.write(':timebase:vernier 1')
#
#         for ff in f:
#             print(str(ff*1e-6)+' MHz')
#             s.write(':wgen:frequency '+str(ff))
#             s.write(':timebase:scale '+str(float(cycles/ff/10)))
#             sleep(delaytime)
#             s.write(':digitize channel'+str(ch))
#             X=float(s.query(':measure:vrms? channel'+str(ch)))
#             print(str(X))
#             with open(flname, 'ab') as fl:
#                 fl.write(str(ff*1e-6)+'\t'+str(X/vamp)+'\n')
#
#             with open(flname, 'ab') as fl:
#                 fl.write(str(ff*1e-6)+'\t'+str(X)+'\n')
#
#
#     s.write(':wgen:output 0')

	
# class Sample:
#
#     def __init__(self,h,temp,freq):
#
#         self.CentreFrequency=freq
#         self.Thickness=h
#         self.CureTemperature=temp
#
#     def GetSignal(self,navg):
#
#         t,x=getdata(navg,len(self.Thickness))
#         self.Time=t
#         self.Data=x
#
#     def GetWavespeedDamping(self):
#
#         from numpy import mean,std,array
#
#         c,alpha=get_c_alpha(self.Time,self.Data,self.Thickness)
#
#
#         self.Wavespeed=c
#         self.Damping=alpha
#
#         c=array(c)
#         alpha=array(alpha)
#
#         self.AverageWavespeed=mean(c)
#         self.AverageDamping=mean(alpha)
#         self.WavespeedDeviation=100*std(c)/mean(c)
#         self.DampingDeviation=100*std(alpha)/mean(alpha)
#
#     def Save(self,filename,writemode='append',pth='/Users/jlesage/Dropbox/ShawCorr/'):
#
#         import pickle,os
#
#         fl=pth+filename+'.p'
#
#         if (os.path.isfile(fl))&(writemode is 'append'):
#
#             s=pickle.load(open(fl,'rb'))
#             s.append(self)
#             pickle.dump(s,open(fl,'wb'))
#
#
#         else:
#
#             s=[]
#             s.append(self)
#             pickle.dump(s,open(fl,'wb'))
#
#
#
#
