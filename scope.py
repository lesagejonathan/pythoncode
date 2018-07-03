def nextpow2(n):
	from numpy import log2, ceil
	N=2**int(ceil(log2(n)))
	return N

def getdata(navg=2,nloc=1,chnum=1):
    import visa
    from numpy import linspace,array
    
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
        T.append(t)
        X.append(x)
        
        
    s.write(':acquire:type normal')
    s.write(':run')
    s.clear()
    return T,X