import gclib

class Controller:

    def __init__(self,instrument='ImmersionTank'):


        if instrument is 'ImmersionTank':

            self.AxisKeys = {'X':'A','Y':'B','Z':'C','Rotation':'D'}

            self.StepsPerMeasure = {'X':1000.,'Y':801.721016,'Z':1000.,'Rotation':160.*1000./360.}

            ip = '10.10.1.12'

        elif instrument is 'ZMC4':

            self.AxisKeys = {'Rotation':'A', 'Index':'B'}

            self.StepsPerMeasure = {'Index':131345. , 'Rotation':79.6}

            ip = '169.254.157.1'


        self.Galil = gclib.py()

        self.Galil.GOpen(ip+' --direct -s ALL')

        self.CurrentPosition = {}


        for k in list(self.AxisKeys.keys()):

            self.ZeroEncoder(k)


    def MoveRelative(self,Axis,Position,Speed):
        # set Speed

        self.Galil.GCommand('SH'+self.AxisKeys[Axis])

        self.Galil.GCommand('SP'+self.AxisKeys[Axis]+'='+str(round(Speed*self.StepsPerMeasure[Axis])))

        # set Position

        self.Galil.GCommand('PR'+self.AxisKeys[Axis]+'='+str(round(Position*self.StepsPerMeasure[Axis])))

        # execute Move

        self.Galil.GCommand('BG'+self.AxisKeys[Axis])


        # Reset Global Position

        self.CurrentPosition[Axis] += Position

    def MoveAbsolute(self,Axis,Position,Speed):

        # set Speed

        self.Galil.GCommand('SH'+self.AxisKeys[Axis])

        self.Galil.GCommand('SP'+self.AxisKeys[Axis]+'='+str(round(Speed*self.StepsPerMeasure[Axis])))

        # set Position

        self.Galil.GCommand('PA'+self.AxisKeys[Axis]+'='+str(round(Position*self.StepsPerMeasure[Axis])))

        # execute Move

        self.Galil.GCommand('BG'+self.AxisKeys[Axis])

        # Reset Global Position

        self.CurrentPosition[Axis] = Position

    def ZeroEncoder(self,Axis):

        self.Galil.GCommand('DP'+self.AxisKeys[Axis]+'=0')

        self.CurrentPosition[Axis] = 0.0

    def __del__(self):

        self.Galil.GClose()
