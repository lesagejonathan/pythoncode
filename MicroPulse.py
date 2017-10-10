import socket
import struct
from numpy import array,zeros,log2
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



def BytesToFloat(x,depth):

    converter = {}

    converter['8'] = lambda x : array([xx-2**7 for xx in x]).astype(float)

    converter['16'] = lambda x : array([x[i]+x[i+1]*256 - 2**15 for i in range(0,len(x),2)]).astype(float)


    return converter[str(depth)](x)

def ReadExactly(sock,size):

    buff = bytearray()

    while len(buff) < size:

        data = sock.recv(size-len(buff))

        if not data:

            break

        buff.extend(data)

    return buff

def ReadBuffer(sock,buff,size,stopcapture):

    while not(stopcapture.is_set()):

        data = sock.recv(size)

        if len(data)>0:
            buff.extend(data)

    return



class PhasedArray:

    def __init__(self,ip='10.10.1.2',port=1067,fsamp=25,pwidth=1/10.,bitdepth=16):


        self.IP = ip
        self.Port = port

        self.Socket = socket.socket()
        self.Socket.connect((ip,port))

        self.PulserSettings = {}
        self.SetSamplingFrequency(fsamp)

        self.SetPulseWidth(pwidth)
        self.SetBitDepth(bitdepth)

        self.ClearScans()



    def SetSamplingFrequency(self,fs=25):

        fset = [10,25,40,50,80,100]

        fs = int(ClosestValue(fset,fs))

        self.PulserSettings['SamplingFrequency'] = int(fs)

        self.Socket.send(('SRST '+str(fs)+'\r').encode())

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

        gain = int(dB/0.25)

        if gain<0:

            Gain = 0

        elif 0<=gain<=80:

            Gain = gain

        elif gain>80:

            Gain = 80

        else:

            Gain = 24

        return Gain

    def ValidPAVoltage(self,voltage):

        vset = list(range(50,205,5))

        return int(ClosestValue(vset,voltage))

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

    def SetPulseWidth(self,width):

        """

            Sets pulse widths on all phased array channels to value specified
            by argument width (or closest valid value), floating point in
            nanoseconds

        """

        wdthset = list(range(20,502,2))

        self.PulserSettings['PulseWidth'] = int(ClosestValue(wdthset,width*1e3))

        self.Socket.send(('PAW 1 128 '+str(self.PulserSettings['PulseWidth'])+'\r').encode())

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

    def SetFMCCapture(self, Elements, Gate, Voltage=100., Gain=20., Averages=0):

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
                    (adjested to closest allowed value)

        Todo:

            * Allow Gate, Gain, Voltage and Averages to be set separately for
              each element

        """

        self.Socket.send(('STX 1\r').encode())


        if type(Elements) is int:

            Elements = (range(1,Elements+1), range(1,Elements+1))


        self.Socket.send(('PAV 1 '+str(len(Elements[0]))+' '+str(int(self.ValidPAVoltage(Voltage)))+'\r').encode())

        gate = (int(Gate[0]*self.PulserSettings['SamplingFrequency']),int(Gate[1]*self.PulserSettings['SamplingFrequency']))

        self.ScanLength = gate[1]-gate[0]

        ReadLength = self.ScanLength*(int(self.PulserSettings['BitDepth'])/8) + 8

        self.SetPRF(1.5e6/(Gate[1]-Gate[0]))

        for tr in Elements[0]:

            self.Socket.send(('TXF '+str(tr)+' 0 -1\r').encode())

            self.Socket.send(('TXF '+str(tr)+' '+str(tr)+' 0\r').encode())
            self.Socket.send(('TXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())

            self.Socket.send(('RXF '+str(tr)+' 0 -1 0\r').encode())

            for rc in Elements[1]:

                self.Socket.send(('RXF '+str(tr)+' '+str(rc)+' 0 0\r').encode())

            self.Socket.send(('RXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())


        self.Socket.send(('SWP 1 '+str(256)+' - '+str(256+len(Elements[0])-1)+'\r').encode())

        self.Socket.send(('GANS 1 '+str(int(self.ValidGain(Gain)))+'\r').encode())
        self.Socket.send(('GATS 1 '+str(gate[0])+' '+str(gate[1])+'\r').encode())
        self.Socket.send(('AMPS 1 13 '+str(int(self.ValidAverage(Averages)))+' 0\r').encode())

        self.Socket.send(('AWFS 1 1\r').encode())

        self.CaptureSettings = {'CatpureType': 'FMC', 'Elements': Elements,
                                'Gate':Gate,'Voltage': Voltage, 'Gain': Gain,
                                'Averages': Averages}


        self.StopCapture = threading.Event()
        self.BufferThread = threading.Thread(target = ReadBuffer, args = (self.Socket, self.Buffer, ReadLength, self.StopCapture))
        self.BufferThread.start()


    def ExecuteCapture(self, NExecutions=1, TimeBetweenCaptures = None):


        """
            Executes capture previously set on MicroPulse the number of times
            specified by NExecutions, waiting TimeBetweenCaptures seconds
            between them
        """

        for n in range(NExecutions):

            self.Socket.send(('CALS 1\r').encode())

            if TimeBetweenCaptures is not None:

                time.sleep(TimeBetweenCaptures)

            self.ScanCount += 1


    def ReadBuffer(self):

        """
            Reads data from the buffer - currently only working
            for FMC captures

            TODO:

            * Add functionality to read scans from buffer and store them for
              sectorial scans, electronic scans, etc.

        """

        if self.CaptureSettings['CatpureType'] == 'FMC':

            Nt = self.ScanLength*(int(self.PulserSettings['BitDepth'])/8)
            Ntr = len(self.CaptureSettings['Elements'][0])
            Nrc = len(self.CaptureSettings['Elements'][1])

            totalscanbytes = self.ScanCount*(Nt+8)*Ntr*Nrc

            while len(self.Buffer)<totalscanbytes:

                time.sleep(0.1)

            self.StopCapture.set()

            indstart = 0
            indstop = 0

            for s in self.ScanCount:

                A = zeros((Ntr, Nrc, self.ScanLength))

                for tr in Ntr:
                    for rc in Nrc:

                        indstart = s*Ntr*Nrc*Nt+tr*Nrc*Nt+rc*Nt+8
                        indstop = indstart + Nt + 1

                        A[tr,rc,:] = BytesToFloat(self.Buffer[indstart:indstop],'16')

                self.AScans.append(A)

            self.ScanCount = 0
            self.Buffer = bytearray()

    def ClearScans(self):

        """

            Removes all scans stored in AScans, zeros ScanCount and stops
            all UT tests in progress + clears MicroPulse data buffer and
            local Buffer variable

        """

        self.AScans = []
        self.ScanCount = 0

        self.Socket.send(('STX 1\r').encode())

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

        self.Socket.send(('STX 1\r').encode())
        self.Socket.close()


# class Conventional:
#
#     def __init__(self, ip = '10.10.1.2', port = 1067, fsamp = 25, pwidth = 1/5., bitdepth = 16., voltage = 200.):
#
#         self.IP =  ip
#         self.Port = port
#
#         self.Socket = socket.socket()
#         self.Socket.connect((self.IP,self.Port))
#
#         self.PulserSettings = {}
#         self.SetSamplingFrequency(fsamp)
#         self.SetPulseWidth(pwidth)
#         self.SetBitDepth(bitdepth)
#         self.SetVoltage(voltage)
#
#         self.AScan = []
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
#     def ValidatePRF(self,prf):
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
#         return PRF
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
#
#
#
# #     def GetFMCData(self,gate=(0.,10.),dB=0,Voltage=200,Averages=0):
# #
# #         self.Socket.send(('PAV 1 '+str(self.NumberOfElements)+' '+str(int(self.ValidPAVoltage(Voltage)))+'\r').encode())
# #
# #         Gate = (int(gate[0]*self.PulserSettings['SamplingFrequency']),int(gate[1]*self.PulserSettings['SamplingFrequency']))
# #
# #         self.SetPRF(1.5e6/(Gate[1]-Gate[0]))
# #
# #         for tr in range(1,self.NumberOfElements+1):
# #
# #             self.Socket.send(('TXF '+str(tr)+' 0 -1\r').encode())
# #
# #             self.Socket.send(('TXF '+str(tr)+' '+str(tr)+' 0\r').encode())
# #             self.Socket.send(('TXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())
# #
# #             self.Socket.send(('RXF '+str(tr)+' 0 -1 0\r').encode())
# #
# #             for rc in range(1,self.NumberOfElements+1):
# #
# #                 self.Socket.send(('RXF '+str(tr)+' '+str(rc)+' 0 0\r').encode())
# #
# #             self.Socket.send(('RXN '+str(tr+256-1)+' '+str(tr)+'\r').encode())
# #
# #
# #         self.Socket.send(('SWP 1 '+str(256)+' - '+str(256+self.NumberOfElements-1)+'\r').encode())
# #
# #         self.Socket.send(('GANS 1 '+str(int(self.ValidGain(dB)))+'\r').encode())
# #         self.Socket.send(('GATS 1 '+str(Gate[0])+' '+str(Gate[1])+'\r').encode())
# #         self.Socket.send(('AMPS 1 13 '+str(int(self.ValidAverage(Averages)))+' 0\r').encode())
# #
# #         self.Socket.send(('AWFS 1 1\r').encode())
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
# #         o = zeros((self.NumberOfElements,self.NumberOfElements,scnlngth))
# #
# #         for tr in range(self.NumberOfElements):
# #
# #             for rc in range(self.NumberOfElements):
# #
# #                 m = ReadExactly(self.Socket,mlngth+8)
# #
# #                 o[tr,rc,:] = BytesToFloat(m[8::],self.PulserSettings['BitDepth'])
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
# #         _pickle.dump(out,open(flname,'wb'))
# #
# #     def __del__(self):
# #
# #         self.Socket.close()
# #
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
