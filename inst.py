def nextpow2(n):
	from numpy import log2, ceil
	N=2**int(ceil(log2(n)))
	return N

def freqsweep(f1,f2,df,flname=None,amp=20.,delaytime=10.,scopechann='ch2',sgenname='usb::0x0699::0x0340::C011598::instr',scopename='TCPIP0::192.168.50.197::inst0::INSTR'):
	import numpy as np
	import visa
	import time
	import os
	sc=visa.Instrument(scopename)
	sg=visa.Instrument(sgenname)
	sg.write('*rst')
	sg.write('output on')
	sg.write('output:impedance infinity')
	sg.write('voltage '+str(amp))
	sc.write('measurement:immed:source '+scopechann)
	sc.write('measurement:immed:type amplitude')
	f=np.arange(f1,f2+df,df)
	X=np.zeros(f.size)
	for i in range(0,f.size):
		print('Frequency: '+str(f[i])+' Hz')
		print('Time Remaining:'+str((f.size*delaytime-i*delaytime)/3600.0)+'hrs')
		sg.write('frequency '+str(f[i]))
		sc.write('autoset execute')
		sc.write('acquire:numavg 512')
		sc.write('acquire:mode average')
		time.sleep(delaytime)
		X[i]=sc.ask('measurement:immed:value?')
	sg.write('output off')
	if type(flname)==tuple:		
		os.chdir(flname[0])
		np.savetxt(flname[1],(f,X))
	return f,X

def getdata_tek(ch):
	from numpy import linspace, array
	import visa
	import os

	s=visa.Instrument('TCPIP0::192.168.50.197::inst0::INSTR',timeout=60.0)

	s.write('*CLS')
	s.write('data:encdg ascii')
	s.write('data:width 2')

	
	s.write('data:source ch'+str(int(ch)))
	xscl=float(s.ask('wfmpre:ymult?'))
	xoff=float(s.ask('wfmpre:yoff?'))
	
	x=xscl*(array(s.ask_for_values('curve?'))-xoff)
		
		
	t1=float(s.ask('wfmpre:xzero?'))
	ts=float(s.ask('wfmpre:xincr?'))
	t2=t1+ts*len(x)
	t=linspace(t1,t2,len(x))
	return t,x

def getdata(chnum=1,navg=None,ctype='usb'):
    import visa
    from numpy import linspace,array
    
    rm=visa.ResourceManager()
	
    if ctype=='net':
        s=visa.Instrument('TCPIP0::192.168.50.236::inst0::INSTR',timeout=30.0)
    # elif ctype=='usb':
    #     s=visa.Instrument('USB0::0x0957::0x1797::MY52160825::0::INSTR',timeout=30.0)
    elif ctype=='usb':
        s=rm.open_resource('USB0::0x0957::0x1797::MY52160825::INSTR')

    while (chnum!=1) and (chnum!=2):
        chnum=input('Input a valid channel number (1 or 2) :')

    s.write('*CLS')

    # if navg is None:
  #       s.write('acquire:type normal')
  #   else:
  #         while (navg<2) or (navg>65536):
  #             navg=input('Input a number of averages between 2 and 65536 :')
        
    s.write(':acquire:type average')
    s.write(':acquire:count '+str(nextpow2(navg)))

    s.write(':waveform:format ascii')
    s.write(':waveform:source channel'+str(int(chnum)))
    s.write(':waveform:points:mode maximum')
	
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
    s.clear()
    return t,x

def fsweep(frange,navg,flname,ctype,ch=1,delaytime=2.,overwrite=False,cycles=10,vamp=1.0):
	import visa
	from numpy import linspace,array,floor
	import os
	from time import sleep
	
	while (navg<2) or (navg>65536):
			navg=input('Input a number of averages between 2 and 65536 :')

	f = 1e6*linspace(frange[0],frange[1],floor((frange[1]-frange[0])/frange[2])+1)
	
	if overwrite==True:
		os.remove(flname)
	if ctype=='net':
		s=visa.Instrument('TCPIP0::192.168.50.236::inst0::INSTR',timeout=30.0)
	elif ctype=='usb':
		s=visa.Instrument('USB0::0x0957::0x1797::MY52160825::0::INSTR',timeout=30.0)

	s.write('*CLS')
	s.write(':wgen:function sinusoid')
	s.write(':wgen:voltage:high '+str(vamp/2))
	s.write(':wgen:voltage:low '+str(-vamp/2))
	s.write(':acquire:type average')
	s.write(':acquire:count '+str(nextpow2(navg)))
	s.write(':wgen:output 1')
	s.write(':trigger:source wgen')

	if type(ch) is int:
		s.write(':waveform:source channel'+str(ch))
		s.write(':timebase:vernier 1')
		
		for ff in f:
			print(str(ff*1e-6)+' MHz')
			s.write(':wgen:frequency '+str(ff))
			s.write(':timebase:scale '+str(float(cycles/ff/10)))
			sleep(delaytime)
			s.write(':digitize channel'+str(ch))
			X=float(s.ask(':measure:vrms? channel'+str(ch)))
			print(str(X))
			with open(flname, 'ab') as fl:
				fl.write(str(ff*1e-6)+'\t'+str(X/vamp)+'\n')

	else:

		for ff in f:
			print(str(ff*1e-6)+' MHz')
			s.write(':wgen:frequency '+str(ff))
			s.write(':timebase:scale '+str(float(cycles/ff/10)))
			sleep(delaytime)
			s.write(':waveform:source channel'+str(ch[0]))
			s.write(':timebase:vernier 1')
			s.write(':digitize channel'+str(ch[0]))
			X1=float(s.ask(':measure:vamplitude? channel'+str(ch[0])))
			s.write(':waveform:source channel'+str(ch[1]))
			s.write(':timebase:vernier 1')
			s.write(':digitize channel'+str(ch[1]))
			X2=float(s.ask(':measure:vamplitude? channel'+str(ch[1])))
			X=X2/X1
			print(str(X))
			with open(flname, 'ab') as fl:
				fl.write(str(ff*1e-6)+'\t'+str(X)+'\n')
		

	s.write(':wgen:output 0')


	
	
	
	
		


