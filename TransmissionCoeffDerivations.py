import sympy as sp
# import numpy as np
# import matplotlib.pylab as plt

# csL = 5.9
# csT = 3.24
# cw = 2.33
#
# rhow = 1.0
# rhos = 7.8

th,cw,csL,csT,rhow,rhos,w,x,y,A,R,TL,TT = sp.symbols('th,cw,csL,csT,rhow,rhos,w,x,y,A,R,TL,TT')
#
#
#
# pw = A*sp.exp(1j*(w/cw)*(sp.sin(th)*x + sp.cos(th)*y)) + R*sp.exp(1j*(w/cw)*(sp.sin(th)*x - sp.cos(th)*y))
#
# ps = TL*sp.exp(1j*w*(sp.sin(th)*x/cw + sp.sqrt((1/csL)**2 - (sp.sin(th)/cw)**2)*y))
#
Ps = TT*sp.exp(1j*w*(sp.sin(th)*x/cw + sp.sqrt((1/csT)**2 - (sp.sin(th)/cw)**2)*y))

uw = [sp.diff(pw,x), sp.diff(pw,y)]


sw = (rhow*cw**2)*sp.diff(uw[1],y)

ss = [rhos*(csL**2-2*csT**2)*sp.diff(us[0],x) + rhos*(csL**2)*sp.diff(us[1],y), rhos*(csT**2)*(sp.diff(us[0],y) + sp.diff(us[1],x))]



V = [TL,TT,R]

E = [((uw[1]-us[1])/w).subs(y,0), ((ss[0]+sw)/w**2).subs(y,0), (ss[1]/w**2).subs(y,0)]


C = [[sp.simplify(sp.exp(-1j*w*sp.sin(th)*x/cw)*sp.diff(EE,VV)) for VV in V] for EE in E]

b = [-sp.simplify(sp.exp(-1j*w*sp.sin(th)*x/cw)*sp.diff(EE,A)) for EE in E]

print(C)

print(b)
