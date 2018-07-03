import socket
import struct
from numpy import array,zeros,log2,frombuffer,int8,int16,uint8,uint16,int32
from matplotlib.pylab import plot,show
import _pickle
# import numpy
import time
import threading

def ClosestIndex(x,val):
    from numpy import array

    return abs(array(x)-val).argmin()

def ClosestValue(x,val):

    return x[ClosestIndex(x,val)]

# def BytesToFloat(x,depth):
#
#     converter = {}
#
#     converter['8'] = lambda x : array([xx-2**7 for xx in x]).astype(float)
#
#     converter['16'] = lambda x : array([x[i]+x[i+1]*256 - 2**15 for i in range(0,len(x),2)]).astype(float)
#
#     return converter[str(depth)](x)

def BytesToData(x,depth,datatype='int16'):

    if int(depth) == 8:

        # return array([xx-128 for xx in x]).astype(datatype)

        return (frombuffer(x, uint8).astype(int16)-int16(128)).astype(datatype)

    elif int(depth) == 16:

        # return array([x[i]+x[i+1]*256 - 2**15 for i in range(0,len(x),2)]).astype(datatype)

        return (frombuffer(x, uint16).astype(int32)-int16(32768)).astype(datatype)


def ReadExactly(sock,size):

    buff = bytearray()

    while len(buff) < size:

        data = sock.recv(size-len(buff))

        if not data:

            break

        buff.extend(data)

    return buff

def ReadBuffer(sock,buff,stopcapture,size=4096):

    while not(stopcapture.is_set()):

        data = sock.recv(size)

        if len(data)>0:
            buff.extend(data)

    return

class PeakNDT:

    def __init__(self,ip='192.168.1.150',port=1067, fsamp=25, bitdepth=16):
        # def __init__(self,ip='192.168.1.150',port=1067, fsamp=25, bitdepth=16):

        self.IP = ip
        self.Port = port

        self.Socket = socket.socket()
        self.Socket.connect((ip,port))

        self.PulserSettings = {}
        self.SetSamplingFrequency(fsamp)
        self.SetBitDepth(bitdepth)

        # self.Socket.send(('ENCM 1\r').encode())

        self.ClearScans()
        self.EncodedScan = False

        self.StepsPerMeasure = (1,1,1,9)

        # self.StepsPerMeasure = 1.
        # self.AxisNumber = 4.

    def SetSamplingFrequency(self,fs=25):

        fset = [10,25,40,50,80,100]

        fs = int(ClosestValue(fset,fs))

        self.PulserSettings['SamplingFrequency'] = int(fs)

        self.Socket.send(('SRST '+str(fs)+'\r').encode())

        ReadExactly(self.Socket,32)

    def SetPRF(self,prf):

        if prf<1:
            PRF = 1
        elif 1<=prf<=20000:
            PRF = int(prf)

        elif prf>20000:
            PRF = 20000

        else:
            PRF = 1000

        self.PulserSettings['PRF'] = PRF
        self.Socket.send(('PRF '+str(PRF)+'\r').encode())

    def ValidGain(self,dB):

        """

            Takes (float) dB and returns closest valid Gain setting
            if illegal gain setting is specified, it is corrected to 24

        """

        gain = int(dB/0.25)

        if gain<0:

            Gain = 0

        elif 0<=gain<=70:

            Gain = gain

        elif gain>70:

            Gain = 80

        else:

            Gain = 24

        return Gain

    def ValidPAVoltage(self, voltage):

        vset = list(range(50,205,5))

        return int(ClosestValue(vset, voltage))

    def ValidConventionalVoltage(self, voltage):

        vset = [50, 100, 150, 200, 250, 300]

        return int(ClosestValue(vset, voltage))

    def ValidAverage(self,naverages):

        if naverages<=1:

            Averages = 0

        elif 1<naverages<=256:

            Averages = log2(naverages)

        elif naverages>256:

            Averages = log2(naverages)

        else:

            Averages = 0

        return Averages

    # def SetPAPulseWidth(self,width):
    #
    #     """
    #
    #         Sets pulse widths on all phased array channels to value specified
    #         by argument width (or closest valid value)
    #         to floating point in nanoseconds
    #
    #     """
    #
    #     wdthset = list(range(20,502,2))
    #
    #     self.PulserSettings['PulseWidth'] = int(ClosestValue(wdthset,width*1e3))
    #
    #     self.Socket.send(('PAW 1 128 '+str(self.PulserSettings['PulseWidth'])+'\r').encode())

    def ValidPAPulseWidth(self,width):

        """
            Gets closest valid pulse width for phased array channels to
            value specified in (float) width in microseconds, returns value
            in nanoseconds

        """

        wdthset = list(range(20,502,2))

        return int(ClosestValue(wdthset,width*1e3))

    def SetPAFilter(self,filtersettings):

        fsettings = list(range(1,5))
        ssettings = list(range(1,9))

        self.Socket.send(('FRQ 0 '+str(ClosestValue(fsettings,filtersettings[0]))+' '+str(ClosestValue(ssettings,filtersettings[1]))+'\r').encode())

    def ValidConventionalPulseWidth(self, width):

        """
            Gets closest valid pulse width for conventional channels to
            value specified in (float) width in microseconds, returns value
            in nanoseconds
        """

        wdthset = list(range(16,1012,2))

        return int(ClosestValue(wdthset,width*1e3))

    def ValidConventionalDamping(self, damping):

        """
            Gets valid setting for (float) damping specified in ohms,
            returns integer value for closest setting

        """

        dampsetting = list(range(8))

        dampvalue = [660, 458, 220, 149, 102, 82, 63, 51]

        return dampsetting[ClosestIndex(dampvalue,damping)]

    def SetBitDepth(self,res):

        """

            Sets bit depth for data returned by MicroPulse, specified by
            integer or string valued argument res

            TODO:

            * Add support for specifying 10 and 12 bit modes


        """

        bitd = {'8':(8,'DOF 1\r'),'16':(16, 'DOF 4\r'), 8:(8,'DOF 1\r'),16:(16, 'DOF 4\r')}

        self.PulserSettings['BitDepth'] = bitd[str(res)][0]

        self.Socket.send((bitd[str(res)][1]).encode())

    def SetConventionalCapture(self, Channels, Gate, Voltage = 100., Gain = 20., Averages = 0 , PulseWidth = 1/10., Damping = 660.):

        """ Sets Conventional UT Capture to be executed

        Channels  - Tuple (transmitting conventional channel, recieving conventional channel)

        Gate - Tuple defining the start and end of the time gate to be recorded
               in microseconds

        Voltage - Float value defining desired element voltage to be applied
                  to the transmitting elements (in Volts, adjusted to closest
                  allowed value)

        Gain - Float value defining desired reciever gain to be applied to
                recieve elements (in dB, adjusted to closest allowed value)

        Averages - Integer number of of averages to be taken for the capture
                    (adjusted to closest allowed value)

        PulseWidth - Floating point number specifying the pulse width in
                    microseconds (adjusted to closest allowed value)

        Damping - Floating point number specifying the channel damping in ohms
                    (adjusted to closest allowed setting)

        Todo:

            * Allow multiple channels to be transmitted and received
             simultaneously (if possible)
            * Allow test number to be appended so that more complicated
              sequences can be handled

        """

        # self.Socket.send(('STX 1\r').encode())


        # Currently only setting channel damping and pulse widths to the same value

        self.Socket.send(('PDW 0 '+str(self.ValidConventionalDamping(Damping))+' '+str(self.ValidConventionalPulseWidth(PulseWidth))+'\r').encode())

        self.Socket.send(('PSV '+str(Channels[0])+' '+str(int(self.ValidConventionalVoltage(Voltage)))+'\r').encode())

        gate = (int(Gate[0]*self.PulserSettings['SamplingFrequency']),int(Gate[1]*self.PulserSettings['SamplingFrequency']))

        self.ScanLength = gate[1]-gate[0]

        ReadLength = int(self.ScanLength*(int(self.PulserSettings['BitDepth'])/8) + 8)

        self.SetPRF(1.5e6/(Gate[1]-Gate[0]))

        self.Socket.send(('NUM 1\r').encode())

        self.Socket.send(('TXN 1 '+str(Channels[0])+'\r').encode())

        self.Socket.send(('RXN 1 '+str(Channels[1])+'\r').encode())

        self.Socket.send(('GAN 1 '+str(self.ValidGain(Gain))+'\r').encode())

        self.Socket.send(('GAT 1 '+str(gate[0])+' '+str(gate[1])+'\r').encode())

        self.Socket.send(('AWF 1 1\r').encode())

        self.Socket.send(('AMP 1 3 0 '+str(self.ValidAverage(Averages))+'\r').encode())

        self.Socket.send(('DLY 1 0\r').encode())

        self.Socket.send(('ETM 1 0\r').encode())

        self.CaptureSettings = {'CatpureType': 'Conventional', 'Channels': Channels,
                                'Gate':Gate,'Voltage': Voltage, 'Gain': Gain,
                                'Averages': Averages, 'PulseWidth': PulseWidth, 'Damping': Damping}


        # self.StopCapture = threading.Event()
        # self.BufferThread = threading.Thread(target = ReadBuffer, args = (self.Socket, self.Buffer, ReadLength, self.StopCapture))
        # self.BufferThread.start()

        self.StartBuffering()

    def SetFMCCapture(self, Elements, Gate, Voltage=200., Gain=70., Averages=0, PulseWidth = 1/10., FilterSettings=(4,1)):

        """ Sets FMC Type Capture to be executed

        Elements  - Either integer number of elements each to be used in
                    transmit/recieve or tuple of ranges: the first defining
                    transmit elements and the second recieve elements

        Gate - Tuple defining the start and end of the time gate to be recorded
               in microseconds

        Voltage - Float value defining desired element voltage to be applied
                  to the transmitting elements (in Volts, adjusted to closest
                  allowed value)

        Gain - Float value defining desired reciever gain to be applied to
                recieve elements (in dB, adjusted to closest allowed value)

        Averages - Integer number of of averages to be taken for the capture
                    (adjusted to closest allowed value)

        PulseWidth - Floating point number defining pulse width for the capture
                    (adjusted to the closest allowed value)


        Todo:

            * Allow Gate, Gain, Voltage and Averages to be set separately for
              each element

        """

        self.SetPAFilter(FilterSettings)

        if type(Elements) is int:

            Elements = (range(1,Elements+1), range(1,Elements+1))

        self.Socket.send(('PAW '+str(Elements[0][0])+' '+str(Elements[0][-1])+' '+str(self.ValidPAPulseWidth(PulseWidth))+'\r').encode())

        self.Socket.send(('PAV '+str(Elements[0][0])+' '+str(Elements[0][-1])+' '+str(int(self.ValidPAVoltage(Voltage)))+'\r').encode())

        gate = (int(Gate[0]*self.PulserSettings['SamplingFrequency']),int(Gate[1]*self.PulserSettings['SamplingFrequency']))

        self.ScanLength = gate[1]-gate[0]

        ReadLength = int(self.ScanLength*(int(self.PulserSettings['BitDepth'])/8) + 8)

        self.SetPRF(1.5e6/(Gate[1]-Gate[0]))

        for tr in range(len(Elements[0])):

            self.Socket.send(('TXF '+str(tr+1)+' 0 -1\r').encode())

            self.Socket.send(('TXF '+str(tr+1)+' '+str(Elements[0][tr])+' 0\r').encode())

            self.Socket.send(('TXN '+str(tr+256)+' '+str(tr+1)+'\r').encode())

            self.Socket.send(('RXF '+str(tr+1)+' 0 -1 0\r').encode())

            for rc in range(len(Elements[1])):

                self.Socket.send(('RXF '+str(tr+1)+' '+str(Elements[1][rc])+' 0 0\r').encode())

            self.Socket.send(('RXN '+str(tr+256)+' '+str(tr+1)+'\r').encode())

        self.Socket.send(('SWP 1 '+str(256)+' - '+str(256+len(Elements[0])-1)+'\r').encode())

        self.Socket.send(('GANS 1 '+str(int(self.ValidGain(Gain)))+'\r').encode())
        self.Socket.send(('GATS 1 '+str(gate[0])+' '+str(gate[1])+'\r').encode())
        self.Socket.send(('AMPS 1 13 '+str(int(self.ValidAverage(Averages)))+' 0\r').encode())

        self.Socket.send(('AWFS 1 1\r').encode())

        self.CaptureSettings = {'CaptureType': 'FMC', 'Elements': Elements,
                                'Gate':Gate,'Voltage': Voltage, 'Gain': Gain,
                                'Averages': Averages, 'PulseWidth':PulseWidth, 'FilterSettings':FilterSettings}

        self.StartBuffering()

    def ExecuteCapture(self, NExecutions=1, TimeBetweenCaptures = None):


        """
            Executes capture previously set on MicroPulse the number of times
            specified by NExecutions, waiting TimeBetweenCaptures seconds
            between them
        """

        for n in range(NExecutions):

            if self.CaptureSettings['CaptureType']=='Conventional':

                self.Socket.send(('CAL 0\r').encode())

            else:

                self.Socket.send(('CALS 0\r').encode())

            if TimeBetweenCaptures is not None:

                time.sleep(TimeBetweenCaptures)

            self.ScanCount += 1

    def OneAxisEncoderCapture(self, Start, End, Pitch):

        """IRPM = Input Ratio Per MM
        Pitch = Inspection Increment
        SDT = Stall Detection Time in Second

        Is Written for Axis 4"""

        self.ScanCount = int(round((End-Start)/Pitch))

        self.Socket.send(('ENCM 0\r').encode())

        self.Socket.send(('ENCT 0 0 0 0\r').encode())

        self.Socket.send(('ENCF 0 0 0 0\r').encode())

        # self.Socket.send(('MPE 10 10 10 ' +  str(int(StepsPerMeasure)) + '\r').encode())

        self.Socket.send(('MPE '+str(int(self.StepsPerMeasure[0]))+' '+str(int(self.StepsPerMeasure[1]))+' '+str(int(self.StepsPerMeasure[2]))+' '+str(int(self.StepsPerMeasure[3]))+'\r').encode())

        self.Socket.send(('BKL 20000 20000 20000 20000\r').encode())

        self.Socket.send(('SPA '+str(Pitch)+' '+str(Pitch)+' '+str(Pitch)+' '+str(Pitch)+'\r').encode())

        self.Socket.send(('LCP 1 0\r').encode())

        self.Socket.send(('LCP 2 0\r').encode())

        self.Socket.send(('LCP 3 0\r').encode())

        self.Socket.send(('LCP 4 0\r').encode())

        # enct=''
        # encf=''
        # mpe=''
        # bkl=''
        # spa=''
        #
        # for m in range(len(Axis)):
        #
        #     enct = enct + ' 0'
        #     encf = encf + ' 0'
        #     mpe = mpe + ' ' + str(IRPM[m])
        #     bkl = bkl + ' ' + str(SDT)
        #     spa = spa + ' ' + str(Pitch[m])
        #
        #     self.Socket.send(('LCP ' + str(Axis[m]) + ' 0\r' ).encode())
        #
        # self.Socket.send(('ENCT' + enct + '\r').encode())
        #
        # self.Socket.send(('ENCF' + encf + '\r').encode())
        #
        # self.Socket.send(('MPE' + mpe + '\r').encode())
        #
        # self.Socket.send(('BKL' + bkl + '\r').encode())
        #
        # self.Socket.send(('SPA' + spa + '\r').encode())

        if self.CaptureSettings['CaptureType']=='Conventional':

            self.Socket.send(('FLM 0\r').encode())

        else:

            self.Socket.send(('FLM 3\r').encode())

        self.Socket.send(('FLX 4 ' + str(int(Start)) + ' 0\r').encode())

        # for m in range(len(Axis)):
        #
        #     if Direction[m] == 'Forward':
        #
        #         self.Socket.send(('FLX ' + str(Axis[m]) + ' ' + str(Start[m]) + ' 0\r').encode())
        #
        #     else:
        #
        #         self.Socket.send(('FLX ' + str(Axis[m]) + ' ' + str(Start[m]) + ' 1\r').encode())

        self.Socket.send(('FLZ 4 ' + str(int(End)) +'\r').encode())

        self.EncodedScan = True

    def ReadAxisLocations(self):

        """ Reads Encoder Position in steps for all axes (4 in total)
        returns a tuple of encoder giving the positions of each axis"""

        self.Socket.send(('STS 0\r').encode())

        al = ReadExactly(self.Socket,18)
        #
        # a = al[11] + al[12]*2**8 + al[13]*2**16
        #
        # # a = al[11]*2**8 + al[12]*2**16 + al[13]*2**24
        #
        #
        # if a < 2**23:
        #
        #     return a
        #
        # else:
        #
        #     return (2**24 - a)*(-1)

        return (int.from_bytes([al[2],al[3],al[4]], byteorder='little', signed=True), int.from_bytes([al[5],al[6],al[7]], byteorder='little', signed=True), int.from_bytes([al[8],al[9],al[10]], byteorder='little', signed=True), int.from_bytes([al[11],al[12],al[13]], byteorder='little', signed=True))

        # al = ReadExactly(self.Socket,26)[7:23]

        # return tuple(frombuffer(al,int32))

    def CalibrateEncoder(self,start,stop,axis=3):

        input("Go to Start Position, Press Enter")

        countstart = self.ReadAxisLocations()[axis]
        # countstart = self.ReadAxisLocations()

        input("Got to End Position, Press Enter")

        countstop = self.ReadAxisLocations()[axis]
        # countstop = self.ReadAxisLocations()

        self.StepsPerMeasure[axis] = abs((countstop - countstart)/(stop - start))*10

        # return(abs((countstop - countstart)/(stop - start))*10)


    def ReadBuffer(self):

        """
            Reads data from the buffer - currently only working
            for FMC and Conventional captures

            TODO:

            * Add functionality to read scans from buffer and store them for
              sectorial scans, electronic scans, conventional tests, etc.

             * Fix Conventional capture to read bytearray correctly
        """

        from numpy import frombuffer

        Nt = int(self.ScanLength*(int(self.PulserSettings['BitDepth'])/8)+8)

        if (self.CaptureSettings['CaptureType'] == 'FMC')&(self.EncodedScan==False):

            Ntr = len(self.CaptureSettings['Elements'][0])
            Nrc = len(self.CaptureSettings['Elements'][1])

            totalscanbytes = self.ScanCount*(Nt*Ntr*Nrc+2)

            while len(self.Buffer)<totalscanbytes:

                time.sleep(1e-3)

            self.StopBuffering()

            indstart = int(0)
            indstop = int(0)

            for s in range(self.ScanCount):

                A = zeros((Ntr, Nrc, self.ScanLength),dtype='int'+str(self.PulserSettings['BitDepth']))

                ibstart = int(s*(Nt*Ntr*Nrc+2))
                ibstop = int(ibstart + Nt*Ntr*Nrc)

                a = self.Buffer[ibstart:ibstop]

                for tr in range(Ntr):

                    for rc in range(Nrc):

                        indstart = int(tr*Nrc*Nt+rc*Nt+8)

                        indstop = int(indstart + Nt-8)

                        A[tr,rc,:] = BytesToData(a[indstart:indstop], self.PulserSettings['BitDepth'], 'int'+str(self.PulserSettings['BitDepth']))


                self.AScans.append(A)

        elif (self.CaptureSettings['CaptureType'] == 'FMC')&(self.EncodedScan==True):

            Ntr = len(self.CaptureSettings['Elements'][0])
            Nrc = len(self.CaptureSettings['Elements'][1])

            totalscanbytes = self.ScanCount*(Nt*Ntr*Nrc+5)

            while len(self.Buffer)<totalscanbytes:

                print(100.*len(self.Buffer)/totalscanbytes,end='\r')
                time.sleep(1e-3)

            self.StopBuffering()

            indstart = int(0)
            indstop = int(0)

            for s in range(self.ScanCount):

                A = zeros((Ntr, Nrc, self.ScanLength),dtype='int'+str(self.PulserSettings['BitDepth']))

                ibstart = int(s*(Nt*Ntr*Nrc)+5*(s+1))
                ibstop = int(ibstart + Nt*Ntr*Nrc)

                a = self.Buffer[ibstart:ibstop]

                for tr in range(Ntr):

                    for rc in range(Nrc):

                        indstart = int(tr*Nrc*Nt+rc*Nt+8)

                        indstop = int(indstart + Nt-8)

                        A[tr,rc,:] = BytesToData(a[indstart:indstop], self.PulserSettings['BitDepth'], 'int'+str(self.PulserSettings['BitDepth']))


                self.AScans.append(A)

        elif self.CaptureSettings['CaptureType'] == 'Conventional':

            totalscanbytes = self.ScanCount*(Nt+8) * self.NumberofEncoderCaptures

            while len(self.Buffer)<totalscanbytes:

                time.sleep(1e-3)

            self.StopBuffering()

            indstart = 0
            indstop = 0

            for s in range(self.ScanCount):


                indstart = s*Nt + 8
                indstop = indstart + Nt - 8

                self.AScans.append(BytesToFloat(self.Buffer[indstart:indstop],self.PulserSettings['BitDepth']))

    def StartBuffering(self):

        """
            Starts or restarts reading device buffer to local buffer

        """
        self.StopBuffering()

        self.ScanCount = 0
        self.Buffer = bytearray()

        self.StopCapture = threading.Event()
        self.BufferThread = threading.Thread(target = ReadBuffer, args = (self.Socket, self.Buffer, self.StopCapture))
        self.BufferThread.start()

    def StopBuffering(self):

        try:

            self.StopCapture.set()
            del(self.BufferThread)


        except:
            pass

    def ClearScans(self):

        """

            Removes all scans stored in AScans, zeros ScanCount and stops
            all UT tests in progress + clears MicroPulse data buffer and
            local Buffer variable

        """

        self.AScans = []
        self.ScanCount = 0

        # self.Socket.send(('STX 1\r').encode())
        # ReadExactly(self.Socket, 8)

        self.Buffer = bytearray()

    def SaveScans(self,Filename, ScanInfo = None, Reversed=False):

        """

        Saves all captured scans in AScans to file specified in string
        Filename along with CaptureSettings any additional information passed
        as dictionary in ScanInfo

        if Reversed == True, then AScans are saved in reversed order


        """

        out = self.PulserSettings.copy()
        out.update(self.CaptureSettings)

        if ScanInfo is not None:

            out.update(ScanInfo)


        if Reversed:

            out['AScans'] = self.AScans[::-1]

        else:

            out['AScans'] = self.AScans


        _pickle.dump(out,open(Filename,'wb'))

    def __del__(self):

        self.StopBuffering()
        self.Socket.close()


# class Conventional:
#
#     def __init__(self,ip='10.10.1.2',port=1067,fsamp=25,pwidth=1/10.,bitdepth=16):
#
#
#         self.IP = ip
#         self.Port = port
#
#         self.Socket = socket.socket()
#         self.Socket.connect((ip,port))
#
#         self.PulserSettings = {}
#         self.SetSamplingFrequency(fsamp)
#
#         self.SetPulseWidth(pwidth)
#         self.SetBitDepth(bitdepth)
#
#         self.ClearScans()
#
#
#     def SetSamplingFrequency(self, fs = 25.):
#
#         fset = [10,25,40,50,80,100]
#
#         fs = int(ClosestValue(fset,fs))
#
#         self.PulserSettings['SamplingFrequency'] = fs
#
#         self.Socket.send(('SRST ' + str(fs) + '\r').encode())
#
#
#
#     def SetPRF(self,prf):
#
#         if prf<1:
#             PRF = 1
#         elif 1<=prf<=20000:
#             PRF = int(prf)
#
#         elif prf>20000:
#             PRF = 20000
#
#         else:
#             PRF = 1000
#
#         self.PulserSettings['PRF'] = PRF
#         self.Socket.send(('PRF '+str(PRF)+'\r').encode())
#
#
#     def SetPulsewidth(self, pw = 1/5.):
#
#         pwset = list(range(16,1010,2))
#
#         pw = ClosestValue(pwset,pw)
#
#         self.PulserSettings['PulseWidth'] = pw
#
#     def SetBitDepth(self, bitdepth = 16.):
#
#         if bitdepth == 8:
#
#             self.Socket.send(('DOF 0\r').encode())
#
#         else:
#
#             self.Socket.send(('DOF 4\r').encode())
#
#         self.PulserSettings['BitDepth'] = bitdepth
#
#     def SetVoltage(self, vol = 200.):
#
#         volset = [50, 100, 150, 200, 250, 300]
#
#         vol = ClosestValue(volset,vol)
#
#         self.PulserSettings['Voltage'] = vol
#
#     def ValidateGain(self, dB = 25.):
#
#         Gain = dB
#
#         if Gain > 70:
#
#             Gain = 70
#
#         Gain = int(round(Gain/0.25))
#
#         return Gain
#
#     def GetConventioalData(self, Channels = (1,1), Gate = (0.,10.),Average = 16, Voltage = 200, ReceiverGain = 25.):
#
#         self.Socket.send(('Num 1\r').encode())
#
#         self.Socket.send(('PSV ' + str(Channels[0]) + ' ' + str(self.PulserSettings['Voltage']) + '\r').encode())
#
#         self.Socket.send(('TXN 1' + ' ' + str(Channels[0]) + '\r').encode())
#
#         self.Socket.send(('RXN 1' + ' ' + str(Channels[0]) + '\r').encode())
#
#         self.Socket.send(('PDW ' + str(Channels[0]) + ' ' + str(0) + ' ' + str(self.PulserSettings['Pulsewidth']) + '\r').encode())
#
#         self.Socket.send(('GAN  1' + ' ' + str(ValidateGain(ReceiverGain)) + '\r').encode())
#
#         self.Socket.send(('AWF 1 1\r').encode())
#
#         self.Socket.send(('GAT 1' + ' ' + str(Gate[0]*self.PulserSettings['SamplingFrequency']) + ' ' + str(Gate[1]*self.PulserSettings['SamplingFrequency']) + '\r').encode())
#
#         self.Socket.send(('DLY 1 0\r').encode())
#
#         self.Socket.send(('AMP 1 3\r').encode())
#
#         self.Socket.send(('ETM 1 0\r').encode())
#
#         PRF = 1.5e6/(Gate[1] - Gate[0])
#
#         self.Socket.send(('PRF ' + str(self.ValidatePRF(PRF)) + '\r').encode())
#
#         self.Socket.send(('CAL 1\r').encode())
#
#         scnlngth = Gate[1] - Gate[0]
#
#         if self.PulserSettings['BitDepth']==16:
#
#             mlngth = 2*scnlngth
#
#         elif self.PulserSettings['BitDepth']==8:
#
#             mlngth = scnlngth
#
#         o = zeros(scnlngth)
#
#         m = ReadExactly(self.Socket,mlngth+8)
#
#         o = BytesToFloat(m[8::],self.PulserSettings['BitDepth'])
#
#         self.AScans.append(o)
#
#     def ClearAScans(self):
#
#         self.AScans = []
#
#     def SaveScan(self,flname,Reversed=False):
#
#         out = self.PulserSettings.copy()
#
#         if Reversed:
#
#             out['AScans'] = self.AScans[::-1]
#
#         else:
#
#             out['AScans'] = self.AScans
#
#
#         pickle.dump(out,open(flname,'wb'))
#
#     def __del__(self):
#
#         self.Socket.close()
#
#
#



#     def GetFMCData(self,gate=(0.,10.),dB=0,Voltage=200,Averages=0):
#
#         self.Socket.send(('PAV 1 '+str(self.NumberOfElements)+' '+str(int(self.ValidPAVoltage(Voltage)))+'\r').encode())
#
#         Gate = (int(gate[0]*self.PulserSettings['SamplingFrequency']),int(gate[1]*self.PulserSettings['SamplingFrequency']))
#
#         self.SetPRF(1.5e6/(Gate[1]-Gate[0]))
#
#         for tr in range(1,self.NumberOfElements+1):
#
#             self.Socket.send(('TXF '+str(tr)+' 0 -1\r').encode())
#
#             self.Socket.send(('TXF '+str(tr)+' '+str(tr)+' 0\r').encode())
#             self.Socket.send(('TXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())
#
#             self.Socket.send(('RXF '+str(tr)+' 0 -1 0\r').encode())
#
#             for rc in range(1,self.NumberOfElements+1):
#
#                 self.Socket.send(('RXF '+str(tr)+' '+str(rc)+' 0 0\r').encode())
#
#             self.Socket.send(('RXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())
#
#
#         self.Socket.send(('SWP 1 '+str(256)+' - '+str(256+self.NumberOfElements-1)+'\r').encode())
#
#         self.Socket.send(('GANS 1 '+str(int(self.ValidGain(dB)))+'\r').encode())
#         self.Socket.send(('GATS 1 '+str(Gate[0])+' '+str(Gate[1])+'\r').encode())
#         self.Socket.send(('AMPS 1 13 '+str(int(self.ValidAverage(Averages)))+' 0\r').encode())
#
#         self.Socket.send(('AWFS 1 1\r').encode())
#         self.Socket.send(('CALS 1\r').encode())
#
#         scnlngth = Gate[1] - Gate[0]
#
#         if self.PulserSettings['BitDepth']==16:
#
#             mlngth = 2*scnlngth
#
#         elif self.PulserSettings['BitDepth']==8:
#
#             mlngth = scnlngth
#
#         o = zeros((self.NumberOfElements,self.NumberOfElements,scnlngth))
#
#         for tr in range(self.NumberOfElements):
#
#             for rc in range(self.NumberOfElements):
#
#                 m = ReadExactly(self.Socket,mlngth+8)
#
#                 o[tr,rc,:] = BytesToFloat(m[8::],self.PulserSettings['BitDepth'])
#
#         self.AScans.append(o)
#
#     def ClearAScans(self):
#
#         self.AScans = []
#
#     def SaveScan(self,flname,Reversed=False):
#
#         out = self.PulserSettings.copy()
#
#         if Reversed:
#
#             out['AScans'] = self.AScans[::-1]
#
#         else:
#
#             out['AScans'] = self.AScans
#
#
#         _pickle.dump(out,open(flname,'wb'))
#
#     def __del__(self):
#
#         self.Socket.close()
#
# #     def SetPhasedArrayInfo(self,Probe = {'NumberofElements': 16., 'Pitch': 0.6}, Wedge = {'WedgeAngle': 36., 'Height': 14.34, 'Velocity': 2330}, BeamSet = {'StartElement': 1., 'ApertureElements': 16.}, PieceVelocity = {'Compression': 5900, 'Shear': 3240}):
# #
# #         self.ProbeInfo = Probe
# #         self.WedgeInfo = Wedge
# #         self.BeamSet = Beamset
# #         self.PieceInfo = PieceVelocity
# #
# #     def SetContactPlaneWaveDelays(self,PlaneWaveAngle = 0.):
# #
# #         delays = []
# #
# #         d = self.ProbeInfo['Pitch']
# #
# #         c = self.PieceInfo['Compression']/1000
# #
# #         delayIncrement = (d/c)*sin(PlaneWaveAngle*pi/180)*1000
# #
# #         #elements = range(self.BeamSetInfo['StartElement'], self.BeamSetInfo['StartElement'] + self.BeamSetInfo['ApertureElements'])
# #
# #         for i in range(0,len(elements)):
# #
# #             #delays.append((elements[i],i*delayIncrement))
# #             delays.append(i*delayIncrement)
# #
# #         if PlaneWaveAngle >= 0:
# #
# #             return delays
# #
# #         else:
# #
# #             return delays.reverse()
# #
# #     def SetWedgePlaneWaveDelays(self,PlaneWaveAngle = 60.):
# #
# #         cw = self.WedgeInfo['Velocity']
# #         phiw = self.WedgeInfo['WedgeAngle'] *pi/180
# #         d = self.ProbeInfo['Pitch']
# #         cs = self.PieceInfo['Shear']
# #
# #         #elements = range(self.BeamSetInfo['StartElement'], self.BeamSetInfo['StartElement'] + self.BeamSetInfo['ApertureElements'])
# #
# #         delays = []
# #
# #         phii = arcsin((cw/cs)*sin(PlaneWaveAngle*pi/180))
# #
# #         delayIncrement = (d/cw)*sin(abs(phiw - phii))
# #
# #         for i in range(0,len(elements)):
# #
# #             #delays.append((elements[i],i*delayIncrement))
# #             delays.append(i*delayIncrement)
# #
# #         if phii >= phiw:
# #
# #             return delays
# #
# #         else:
# #
# #             return delays.reverse()
# #
# #     def SetSectorialBeamSet(self, Contact = False, MinAngle = 30., MaxAngle = 70., AngleIncrement = 1.):
# #
# #         angles = range(MinAngle,MaxAngle + 1,AngleIncrement)
# #
# #         focallaw = {}
# #
# #         if Contact:
# #
# #             for i in range(0,len(angles)):
# #
# #                 focallaw[str(angles[i])] = self.SetContactPlaneWaveDelays(angles[i])
# #         else:
# #
# #             for i in range(0,len(angles)):
# #
# #                 focallaw[str(angles[i])] = self.SetWedgePlaneWaveDelays(angles[i])
# #
# #         self.FocalLaws = focallaw
# #         self.Angles = angles
# #
# #     def GetPhasedArrayData(self):
# #
# #         elements = range(self.BeamSetInfo['StartElement'], self.BeamSetInfo['StartElement'] + self.BeamSetInfo['ApertureElements'])
# #         testNo = 256
# #
# #         for i in range(len(self.FocalLaws)):
# #
# #             self.Socket.send(('TXF ' + str(i+1) + ' 0 -1\r').encode())
# #
# #             for j in range(len(elements)):
# #
# #                 self.Socket.send(('TXF ' + str(i+1) + ' ' + str(elements[j]) + ' ' + str(self.FocalLaws[str(self.Angles[i])][j]) + '\r').encode())
# #
# #             self.Socket.send(('TXN ' + str(testNo) + ' ' + str(i+1) +'\r').encode())
# #
# #             self.Socket.send(('RXF ' + str(i+1) + ' 0 -1\r').encode())
# #
# #             for j in range(len(elements)):
# #
# #                 self.Socket.send(('RXF ' + str(i+1) + ' ' + str(elements[j]) + ' ' + str(self.FocalLaws[str(self.Angles[i])][j]) + '\r').encode())
# #
# #             self.Socket.send(('RXN ' + str(testNo) + ' ' + str(i+1) +'\r').encode())
# #
# #             testNo = testNo + 1
# #
# #         self.Socket.send(('SWP 1 256 - ' + str(testNo-1) + '\r').encode())
# #
# #         self.Socket.send(('GANS 1 ' + str(int(self.ValidGain(dB))) + '\r').encode())
# #
# #         self.Socket.send(('AMPS 1 3 ' + str(int(self.ValidAverage(Averages))) + '\r').encode())
# #
# #         self.Socket.send(('CALS 1\r').encode())
# #
# #         scnlngth = Gate[1] - Gate[0]
# #
# #         if self.PulserSettings['BitDepth']==16:
# #
# #             mlngth = 2*scnlngth
# #
# #         elif self.PulserSettings['BitDepth']==8:
# #
# #             mlngth = scnlngth
# #
# #         o = zeros((len(self.FocalLaws),scnlngth))
# #
# #         for i in range(len(self.FocalLaws)):
# #
# #             m = ReadExactly(self.Socket,mlngth+8)
# #
# #             o[i,:] = BytesToFloat(m[8::],self.PulserSettings['BitDepth'])
# #
# #         self.AScans.append(o)
# #
# # class ConventionalUT:
# #
# #     def __init__(self, ip = '10.10.1.2', port = 1067, fsamp = 25, pwidth = 1/5., bitdepth = 16., voltage = 200.):
# #
# #         self.IP =  ip
# #         self.Port = port
# #
# #         self.Socket = socket.socket()
# #         self.Socket.connect((self.IP,self.Port))
# #
# #         self.PulserSettings = {}
# #         self.SetSamplingFrequency(fsamp)
# #         self.SetPulseWidth(pwidth)
# #         self.SetBitDepth(bitdepth)
# #         self.SetVoltage(voltage)
# #
# #         self.AScan = []
# #
# #     def SetSamplingFrequency(self, fs = 25.):
# #
# #         fset = [10,25,40,50,80,100]
# #
# #         fs = int(ClosestValue(fset,fs))
# #
# #         self.PulserSettings['SamplingFrequency'] = fs
# #
# #         self.Socket.send(('SRST ' + str(fs) + '\r').encode())
# #
# #     def ValidatePRF(self,prf):
# #
# #         if prf<1:
# #             PRF = 1
# #         elif 1<=prf<=20000:
# #             PRF = int(prf)
# #
# #         elif prf>20000:
# #             PRF = 20000
# #
# #         else:
# #             PRF = 1000
# #
# #         return PRF
# #
# #     def SetPulsewidth(self, pw = 1/5.):
# #
# #         pwset = list(range(16,1010,2))
# #
# #         pw = ClosestValue(pwset,pw)
# #
# #         self.PulserSettings['PulseWidth'] = pw
# #
# #     def SetBitDepth(self, bitdepth = 16.):
# #
# #         if bitdepth == 8:
# #
# #             self.Socket.send(('DOF 0\r').encode())
# #
# #         else:
# #
# #             self.Socket.send(('DOF 4\r').encode())
# #
# #         self.PulserSettings['BitDepth'] = bitdepth
# #
# #     def SetVoltage(self, vol = 200.):
# #
# #         volset = [50, 100, 150, 200, 250, 300]
# #
# #         vol = ClosestValue(volset,vol)
# #
# #         self.PulserSettings['Voltage'] = vol
# #
# #     def ValidateGain(self, dB = 25.):
# #
# #         Gain = dB
# #
# #         if Gain > 70:
# #
# #             Gain = 70
# #
# #         Gain = int(round(Gain/0.25))
# #
# #         return Gain
# #
# #     def GetConventioalData(self, Channels = (1,1), Gate = (0.,10.),Average = 16, Voltage = 200, ReceiverGain = 25.):
# #
# #         self.Socket.send(('Num 1\r').encode())
# #
# #         self.Socket.send(('PSV ' + str(Channels[0]) + ' ' + str(self.PulserSettings['Voltage']) + '\r').encode())
# #
# #         self.Socket.send(('TXN 1' + ' ' + str(Channels[0]) + '\r').encode())
# #
# #         self.Socket.send(('RXN 1' + ' ' + str(Channels[0]) + '\r').encode())
# #
# #         self.Socket.send(('PDW ' + str(Channels[0]) + ' ' + str(0) + ' ' + str(self.PulserSettings['Pulsewidth']) + '\r').encode())
# #
# #         self.Socket.send(('GAN  1' + ' ' + str(ValidateGain(ReceiverGain)) + '\r').encode())
# #
# #         self.Socket.send(('AWF 1 1\r').encode())
# #
# #         self.Socket.send(('GAT 1' + ' ' + str(Gate[0]*self.PulserSettings['SamplingFrequency']) + ' ' + str(Gate[1]*self.PulserSettings['SamplingFrequency']) + '\r').encode())
# #
# #         self.Socket.send(('DLY 1 0\r').encode())
# #
# #         self.Socket.send(('AMP 1 3\r').encode())
# #
# #         self.Socket.send(('ETM 1 0\r').encode())
# #
# #         PRF = 1.5e6/(Gate[1] - Gate[0])
# #
# #         self.Socket.send(('PRF ' + str(self.ValidatePRF(PRF)) + '\r').encode())
# #
# #         self.Socket.send(('CAL 1\r').encode())
# #
# #         scnlngth = Gate[1] - Gate[0]
# #
# #         if self.PulserSettings['BitDepth']==16:
# #
# #             mlngth = 2*scnlngth
# #
# #         elif self.PulserSettings['BitDepth']==8:
# #
# #             mlngth = scnlngth
# #
# #         o = zeros(scnlngth)
# #
# #         m = ReadExactly(self.Socket,mlngth+8)
# #
# #         o = BytesToFloat(m[8::],self.PulserSettings['BitDepth'])
# #
# #         self.AScans.append(o)
# #
# #     def ClearAScans(self):
# #
# #         self.AScans = []
# #
# #     def SaveScan(self,flname,Reversed=False):
# #
# #         out = self.PulserSettings.copy()
# #
# #         if Reversed:
# #
# #             out['AScans'] = self.AScans[::-1]
# #
# #         else:
# #
# #             out['AScans'] = self.AScans
# #
# #
# #         pickle.dump(out,open(flname,'wb'))
# #
# #     def __del__(self):
# #
# #         self.Socket.close()
