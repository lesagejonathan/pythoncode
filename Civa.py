import xml.etree.ElementTree as ET
import os
from numpy import *
import linecache

def LoadAScansFromTxt(fl):

    nt = int(linecache.getline(fl,2).split(' : ')[1])

    x = genfromtxt(fl,delimiter=';',skip_header=11)

    nch = int((x.shape[1]-3)/2)

    x = x[:,-nch-1:-1]

    x = [x[i:i+nt,:].transpose() for i in range(0,nt*nch,nt)]

    x = vstack(x)


    x = x.reshape((nch,nch,nt))

    return x

def ConvertAScanToBinary(infl,outfl,bitdepth='16'):

    x = LoadAScansFromTxt(infl)


    n = x.shape

    x = x.reshape((n[0]*n[1],n[2]))

    converter = {'8': lambda x: (256*x/(amax(x)-amin(x))).astype(int8), '16': lambda x: (65536*x/(amax(x)-amin(x))).astype(int16)}

    x = converter[bitdepth](x)

    f = open(outfl,'wb')

    x.tofile(f)

    f.close()

def ConvertAScansToBinary(indir,outdir,bitdepth='16'):

    D = os.listdir(indir)

    print(D)

    if not os.path.isdir(outdir):

        os.mkdir(outdir)

    for d in D:

        ConvertAScanToBinary(indir+d,outdir+d,bitdepth)




class CivaModel:

    def __init__(self,ModelDir):

        self.ModelXML = ModelDir + '/proc0/model.xml'
        self.Geometry = []
        self.Defect = []

        self.tree=ET.parse(self.ModelXML)
        self.root=self.tree.getroot()


    def DrawLine(self,Boundary,pt1,pt2,LineType="Straight"):

        b = {'Front':'ROUGE', 'BackWall': 'VERT', 'Side': 'BLEU', 'Interface': 'JAUNE'}
        l = {'Straight': 'SEG_DROITE'}

        self.Geometry.append(b[Boundary]+' '+l[LineType]+' '+str(pt1[0])+' '+str(pt1[1])+' '+str(0)+' '+str(pt2[0])+' '+str(pt2[1])+' '+str(0)+' ')

    def UpdateXMLGeometry(self):

        self.root[0][0].attrib['descriptionCgef']='GRAPHICINTERFACE: '+ str(len(self.Geometry)) + ' ' + ''.join(self.Geometry)

    def DrawDefect(self,Boundary,pt1,pt2,LineType="Straight"):

        b = {'Extrude':'MAGENTA','Front':'ROUGE'}
        l = {'Straight': 'SEG_DROITE'}

        self.Defect.append(b[Boundary]+' '+l[LineType]+' '+str(pt1[0])+' '+str(pt1[1])+' '+str(0)+' '+str(pt2[0])+' '+str(pt2[1])+' '+str(0)+' ')

    def UpdateXMLDefect(self):

        self.root[5][0][1][0].attrib['descriptionCgef']= 'GRAPHECHINTERFACE: '+ str(len(self.Defect))+ ' ' +''.join(self.Defect)

    def SetDefectcentercoordinates(self,x,y,z):

        self.root[5][0][0][1][0].attrib['x'] = str(x)
        self.root[5][0][0][1][0].attrib['y'] = str(y)
        self.root[5][0][0][1][0].attrib['z'] = str(z)

    def WriteUpdatedXML(self):

        self.tree.write(self.ModelXML,encoding="ISO-8859-1",xml_declaration=True)
        f = open(self.ModelXML,'r')
        L = f.readlines()
        f.close()
        os.remove(self.ModelXML)
        H = '<!DOCTYPE ChampSons PUBLIC "-//fr.cea//DTD champsons.resources.dtd.ChampSons//FR" "ChampSons.dtd" >'
        K = [L[0],H]
        KK = K + L[1::]
        g = open(self.ModelXML,'w')
        g.writelines(KK)
        g.close()



class LSample:

    def __init__(self,Model):

        self.Model = Model

    def DrawGeometry(self,Thickness,WedgeLength,WeldVerticalFusionLength,WeldHorizontalFusionLength):

        #self.Model.DrawLine('Side',(0,0),(0,Thickness))
        #self.Model.DrawLine('Front',(0,Thickness),(WedgeLength,Thickness))
        #self.Model.DrawLine('BackWall',(WedgeLength,Thickness),(WedgeLength + WeldVerticalFusionLength,Thickness + WeldHorizontalFusionLength))
        #self.Model.DrawLine('Side',(WedgeLength + WeldVerticalFusionLength,Thickness + WeldHorizontalFusionLength),(WedgeLength + WeldVerticalFusionLength,Thickness))
        #self.Model.DrawLine('BackWall',(WedgeLength + WeldVerticalFusionLength,Thickness),(WedgeLength + WeldVerticalFusionLength,0))
        #self.Model.DrawLine('BackWall',(WedgeLength + WeldVerticalFusionLength,0),(0,0))

        self.Model.DrawLine('Side',(0,0),(0,-Thickness))
        self.Model.DrawLine('BackWall',(0,-Thickness),(WedgeLength + WeldVerticalFusionLength,-Thickness))
        self.Model.DrawLine('BackWall',(WedgeLength + WeldVerticalFusionLength,-Thickness),(WedgeLength + WeldVerticalFusionLength,0))
        self.Model.DrawLine('Side',(WedgeLength + WeldVerticalFusionLength,0),(WedgeLength + WeldVerticalFusionLength,WeldHorizontalFusionLength))
        self.Model.DrawLine('BackWall',(WedgeLength + WeldVerticalFusionLength,WeldHorizontalFusionLength),(WedgeLength,0))
        self.Model.DrawLine('Front',(WedgeLength,0),(0,0))
        self.Model.UpdateXMLGeometry()

    def DrawDefect(self,DisbondLength,DefectExtrusionLength,RootGapVerticalLength,RootGapHorizontalLength):

        self.Model.DrawDefect('Extrude',(0,0),(DefectExtrusionLength,0))
        self.Model.DrawDefect('Front',(0,0),(-RootGapVerticalLength,0))
        self.Model.DrawDefect('Front',(-RootGapVerticalLength,0),(0,RootGapHorizontalLength))
        if DisbondLength > 0:
            self.Model.DrawDefect('Front',(0,RootGapHorizontalLength),(0,RootGapHorizontalLength + DisbondLength))
        self.Model.UpdateXMLDefect()

    def LocateDefect(self,WedgeLength,WeldVerticalFusionLength,DisbondLength,RootGapVerticalLength,RootGapHorizontalLength):

        self.Model.SetDefectcentercoordinates(WedgeLength + WeldVerticalFusionLength - 0.5*RootGapVerticalLength,25,-0.5*(RootGapHorizontalLength + DisbondLength))
