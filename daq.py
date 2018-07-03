def get_accel_time(ch,rate,tend,tpt,trig_level):
    import ctypes
    import numpy
    nidaq = ctypes.windll.nicaiu # load the DLL
    ##############################
    # Setup some typedefs and constants
    # to correspond with values in
    # C:\Program Files\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include\NIDAQmx.h
    # the typedefs
    sens=(105.6,100.1,100.8)
    
    int32 = ctypes.c_long
    uInt32 = ctypes.c_ulong
    uInt64 = ctypes.c_ulonglong
    float64 = ctypes.c_double
    TaskHandle = uInt32
    # the constants
    DAQmx_Val_Cfg_Default = int32(-1)
    DAQmx_Val_Newtons=15875
    DAQmx_Val_mVoltsPerNewton=15891
    DAQmx_Val_Volts = 10348
    DAQmx_Val_Rising = 10280
    DAQmx_Val_FiniteSamps = 10178
    DAQmx_Val_GroupByChannel = 0
    DAQmx_Val_PseudoDiff=12529
    DAQmx_Val_AccelUnit_g=10186
    DAQmx_Val_mVoltsPerG=12509
    DAQmx_Val_Internal=10200
    ##############################
    def CHK(err):
        """a simple error checking routine"""
        if err < 0:
            buf_size = 100
            buf = ctypes.create_string_buffer('\000' * buf_size)
            nidaq.DAQmxGetErrorString(err,ctypes.byref(buf),buf_size)
            raise RuntimeError('nidaq call failed with error %d: %s'%(err,repr(buf.value)))
    # initialize variables
    taskHandle = TaskHandle(0)
    max_num_samples = int(tend*rate)
    t=numpy.linspace(0,tend,max_num_samples)
    data = numpy.zeros((len(ch)*max_num_samples,),dtype=numpy.float64)
    
   
    CHK(nidaq.DAQmxCreateTask("",ctypes.byref(taskHandle)))
    
    CHK(nidaq.DAQmxCreateAIForceIEPEChan(taskHandle,"Dev1/ai0","", DAQmx_Val_PseudoDiff,float64(-10),float64(10),DAQmx_Val_Newtons, float64(1),DAQmx_Val_mVoltsPerNewton,DAQmx_Val_Internal,float64(0.0021),None))
    

    for c in ch:
        
        CHK(nidaq.DAQmxCreateAIAccelChan(taskHandle,"Dev1/ai"+str(c),"",DAQmx_Val_PseudoDiff,float64(-10),float64(10),DAQmx_Val_AccelUnit_g,float64(sens[c-1]),DAQmx_Val_mVoltsPerG,DAQmx_Val_Internal,float64(0.0021),None))
        
 
 
                                        
    CHK(nidaq.DAQmxCfgSampClkTiming(taskHandle,"",float64(rate),
                                    DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,
                                    uInt64(max_num_samples)))
                                    
    CHK(nidaq.DAQmxCfgAnlgEdgeRefTrig(taskHandle,"Dev1/ai0", DAQmx_Val_Rising, float64(trig_level), uInt32(int(tpt*rate))))
             
                                    
    #CHK(nidaq.DAQmxCfgAnlgEdgeStartTrig(taskHandle,"Dev1/ai0",DAQmx_Val_Rising,float64(trig_level)));

                                    

    CHK(nidaq.DAQmxStartTask(taskHandle))
    read = int32()
    CHK(nidaq.DAQmxReadAnalogF64(taskHandle,max_num_samples,float64(10.0),
                                DAQmx_Val_GroupByChannel,data.ctypes.data,
                               (len(ch)+1)*max_num_samples,ctypes.byref(read),None))
    if taskHandle.value != 0:
        nidaq.DAQmxStopTask(taskHandle)
        nidaq.DAQmxClearTask(taskHandle)
        
    x=data.reshape((max_num_samples,len(ch)+1),order='F')
        
    return t,x