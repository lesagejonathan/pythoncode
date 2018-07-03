import sympy as sp

kx,w,rho,cL,cT,A,B,z,x,sinc = sp.symbols('kx,w,rho,cL,cT,A,B,z,x,sinc')

p = A*sp.exp(1j*z*sp.sqrt((w/cL)**2 - kx**2))*exp(1j*kx*x) + B*sp.exp(-1j*z*sp.sqrt((w/cL)**2 - kx**2))*exp(1j*kx*x)

v = [x,z]

u = [sp.simplify(sp.diff(vv)) for vv in v]

M = rho*cL**2
G = rho*cT**2
L = rho*(cL**2 - 2*cT**2)

# s = [M*sp.diff(u[1],z) + L*sp.diff(u[0],x), G*(sp.diff(u[1],x) + sp.diff(u[0],z))]

s = M*sp.diff(u[1],z) + L*sp.diff(u[0],x)

s0 = sp.subs(s,z,0.)

bc = s0 - 
