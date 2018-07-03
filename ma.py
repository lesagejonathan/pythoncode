def getdata(flname,Nc,Navg,Nt,Nch=-1):
	from numpy import loadtxt,concatenate,tile
	data = loadtxt(flname)
	for ii in range(data.shape[0]):
		xx=data[ii,:].reshape((Nc,Navg,Nt,Nch)).transpose((2,0,3,1))[:,:,0,:]
		xx=xx.reshape((Nt,Nc,1,Navg))
		yy=data[ii,:].reshape((Nc,Navg,Nt,Nch)).transpose((2,0,3,1))[:,:,1::,:]
		xx=tile(xx,(1,1,yy.shape[2],1))
		if ii==0:
			x=xx
			y=yy
		else:
			x=concatenate((x,xx),axis=2)
			y=concatenate((y,yy),axis=2)
		
	return x,y


def H1(x,y,dt):
	from numpy import average,linspace,concatenate
	from numpy.fft import rfft
	X=rfft(x,None,0)
	Y=rfft(y,None,0)
	H1=average((X.conj()*Y)/(X.conj()*X),3)
#	H1=concatenate((H1,H1[:,:,0,:].reshape((H1.shape[0],H1.shape[1],1,H1.shape[3]))),axis=2)
	f=linspace(0.,1./(2*dt),H1.shape[0])
	return f,H1
	
	
def nspec(H,N):
	from numpy import trapz,linspace,pi,cos,sin,zeros
	from numpy.matlib import repmat
	th=linspace(0.,2*pi,H.shape[1])
	H=H.transpose((0,2,1))
	Hnc=zeros((H.shape[0],H.shape[1],N),complex)
	Hns=zeros((H.shape[0],H.shape[1],N),complex)
	for n in range(N):
		Hnc[:,:,n]=(1/(2*pi))*trapz(cos(n*th)*H,dx=th[1],axis=2)
		Hns[:,:,n]=(1/(2*pi))*trapz(sin(n*th)*H,dx=th[1],axis=2)
	return Hnc,Hns
		
	
#def getdata(flname,dim):
#	from numpy import loadtxt
#	data = loadtxt(flname)
#	data = data.reshape(dim).transpose((4,3,0,1,2))
#	x=data[0,:,:,:,:]
#	y=data[1::,:,:,:]
#	return x,y	

#def H1(x,y,dt):
#	from numpy import average,linspace,concatenate
#	from numpy.fft import rfft
#	X=rfft(x,None,0)
#	Y=rfft(y,None,1)
#	H1=average((X.conj()*Y)/(X.conj()*X),4)
#	H1=concatenate((H1,H1[:,:,0,:].reshape((H1.shape[0],H1.shape[1],1,H1.shape[3]))),axis=2)
#	f=linspace(0.,1./(2*dt),H1.shape[1])
#	return f,H1


	
#def nmspec(X):
#	from numpy.fft import fft
#	Xnm=fft(fft(X,None,1),None,2)
#	return Xnm
#	
#def pltms(f,X,frange,numfreqs,zloc=0,R=1.):
#	from numpy import linspace,pi,imag
#	from spr import pkfind
#	from matplotlib.pyplot import subplot,plot,grid,xlabel,ylabel,title,polar,show,figure,close,legend
#	close('all')
#	indmax,fres,Xres=pkfind(f[(f>=frange[0]) & (f<=frange[1])],imag(X[(f>=frange[0]) & (f<=frange[1]),0,zloc]),numfreqs)
#	th=linspace(0,2*pi,X.shape[1])
#	fresst=[]
#	for i in range(len(indmax)):
#		figure(0)
#		polar(th,imag(X[indmax[i],:,zloc])*(0.2*R/abs(imag(X[indmax[i],:,zloc])).max())+R)
#		title('Circumferential Mode Shapes')
#		fresst.append(str(fres[i])+' Hz')
#	legend(fresst)


#class modal:
#	def __init__(self,dbnamedbfile='modal.dat',pth=None):
#		import os.chdir
#		if pth==None:
#			if os.path.exists('/Users/jlesage'):
#				self.pth='/Users/jlesage/Dropbox/experimental_data'
#			elif os.path.exists('/home/jonathan/Dropbox'):
#				self.pth='/home/jonathan/Dropbox/experimental_data'
#			elif os.path.exists('/home/jlesage'):
#				self.pth='/home/jlesage/Dropbox/experimental_data'
#		else:
#			self.pth=pth
#		self.dbname=dbname

#	def save(self,flname,param):
#		import 
#		
#		
#		
