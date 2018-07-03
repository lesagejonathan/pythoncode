from sympy import *
from sympy.matrices import *
from sympy.assumptions.assume import global_assumptions

x,y,xx,yy,kxtr,kytr,kxrc,kyrc,h,A,B,C,D,ZwL,ZwT,cwL,cwT,csL,csT,ZsL,ZsT,th,r,cphi,sphi,tphi,ksL,ksT,n,y0,JnT,JnL,dJnT,dJnL,d2JnT,d2JnL,cth,sth = symbols('x y xx yy kxtr kytr,kxrc,kyrc h A B C D ZwL ZwT cwL cwT csL csT ZsL ZsT th r cphi sphi tphi ksL ksT n y0 JnT JnL dJnT dJnL d2JnT d2JnL cth sth')

Ptr = A*exp(1j*(-kytr*y + kxtr*x))/(-1j*kytr)
Prc = B*exp(1j*(kyrc*y + kxrc*x))/(1j*kyrc)
#
#
Uw = Matrix([[diff(Ptr,x)+diff(Prc,x)],[diff(Ptr,y)+diff(Prc,y)]])

Rphi = Matrix([[cphi,-sphi],[sphi,cphi]])

Mw = cwL*ZwL
lw = cwL*ZwL - 2 *cwT*ZwT

Ew = [diff(Uw[0,0],x),diff(Uw[1,0],y),diff(Uw[0,0],y)+diff(Uw[1,0],x)]

Sw = Matrix([[ Mw*Ew[0] + lw*Ew[1], cwT*ZwT*Ew[2]],[cwT*ZwT*Ew[2], lw*Ew[0] + Mw*Ew[1]]])

Swn = Rphi*Sw*Rphi.transpose()

Uwn = Rphi*Uw

Uwn = Uwn.subs([(x,cphi*xx - sphi*(yy-h)),(y, sphi*xx + cphi*(yy-h))])
Uwn = simplify(Uwn.subs(yy,0))

Swn = Swn.subs([(x,cphi*xx - sphi*(yy-h)),(y, sphi*xx + cphi*(yy-h))])
Swn = simplify(Swn.subs(yy,0))

# b = [ Uwn[1,0], Swn[1,1], 0]

b = [Uwn[1,0], Swn[1,1]]

#
#
# sw = Matrix(((Mw*diff(Uw[0],x) + lw*diff(Uw[1],y)),(lw*diff(Uw[0],x)+Mw*diff(Uw[1],y)), (cwT*ZwT*(diff(Uw[0],y) + diff(Uw[1],x)))))
#
# Rv = Matrix(((cphi,-sphi),(sphi,cphi)))
# Rt = Matrix(((cphi**2,sphi**2,2*sphi*cphi),(sphi**2,cphi**2,-2*sphi*cphi),(-sphi*cphi,sphi*cphi,(cphi**2-sphi**2))))
#
# Uw = Rv*Uw
# Uw = Uw.subs(x,cphi*(xx+H*tphi) - sphi*(yy+H))
# Uw = Uw.subs(y,sphi*(xx+H*tphi)+cphi*(yy+H))
# Uw = Uw.subs(yy,0)
#
# sw = Rt*sw
# sw = sw.subs(x,cphi*(xx+H*tphi) - sphi*(yy+H))
# sw = sw.subs(y,sphi*(xx+H*tphi)+cphi*(yy+H))
# sw = sw.subs(yy,0)
#
# Sw = Matrix(((sw[0],sw[2]),(sw[2],sw[1])))
#
# nw = Matrix(((0),(1)))
#
# Uwn = nw.transpose()*Uw
#
# Swn = Sw*nw

# PDs = C*JnL*exp(1j*n*th)
# PSs = D*JnT*exp(1j*n*th)

PDs = C*besselj(n,ksL*r)*exp(1j*n*th)
PSs = D*besselj(n,ksT*r)*exp(1j*n*th)

# Us = Matrix([[PDs.subs(JnL,dJnL) + diff(PSs,th)/r],[diff(PDs,th)/r - PSs.subs(JnT,dJnT)]])

Us = Matrix([ [diff(PDs,r) + diff(PSs,th)/r ], [diff(PDs,th)/r - diff(PSs,r)] ])

Ms = csL*ZsL
ls = csL*ZsL - 2*csT*ZsT

Rth = Matrix([[cth,-sth],[sth,cth]])


# F = r - y0/sin(th)
#
# n = Matrix(((diff(F,r)), (diff(F,th)/r)))
# n = n/sqrt(n[0,0]**2 + n[1,0]**2)

# U = [Us[0,0], Us[1,0]]
#
# E = [Us[0,0].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)]), (1/r)*(Us[0,0] + diff(Us[1,0],th)), diff(Us[0,0],th)/r + Us[1,0].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)]) - Us[1,0]/r]

E = [diff(Us[0,0],r), (1/r)*(Us[0,0] + diff(Us[1,0],th)), diff(Us[0,0],th)/r + diff(Us[1,0],r) - Us[1,0]/r]


Es = Matrix([[E[0],E[2]],[E[2],E[1]]])

Ss = Matrix([[ Ms*E[0] + ls*E[1], csT*ZsT*E[2] ], [csT*ZsT*E[2], ls*E[0] + Ms*E[1]]])

Us = Rth*Us

Ssn = Rth*Ss*Rth.transpose()

e = [Us[1,0].subs(r,y0/sth),Ssn[1,1].subs(r,y0/sth),Ssn[0,1].subs(r,y0/sth)]

Us = Us.subs([(n*besselj(n,ksL*r)/r, (ksL/2)*(besselj(n-1,ksL*r)+besselj(n+1,ksL*r))), (n*besselj(n,ksL*r)/r, (ksT/2)*(besselj(n-1,ksT*r)+besselj(n+1,ksT*r)))])

Us = simplify(Us)

Us = [Us[0,0],Us[1,0]]
# Ss = Matrix([[Ms*(Us[0].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)])+(ls/r)*(Us[0]+diff(Us[1],th),(csT*ZsT/r)*(diff(Us[0],th) + (Us[1].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)]) - Us[1]/r)],[(csT*ZsT/r)*(diff(Us[0],th) + Us[1].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)]) - Us[1]/r),ls*diff(Us[0].subs([(JnL,dJnL),(JnT,dJnT),(dJnL,d2JnL),(dJnT,d2JnT)])+(Ms/r)*(Us[0]+diff(Us[1],th)]])

# Us = Us.subs(r,y0/sin(th))

# Usn = (n.transpose())*Us
#
# Usn = Usn.subs(r,y0/sin(th))
#
# Ssn = Ss*n
#
# Ssn = Ssn.subs(r,y0/sin(th))
#
# v = [C,D]
#
# # e = [Uwn[0]-Usn[0], Swn[1,0] - Ssn[1,0]]
#
# e = [Usn[0], Ssn[1,0],Ssn[0,0]]
#
# b = [simplify(Uwn[0]), simplify(Swn[1,0]),0]
#
#
# # e = [Uwn-Usn, Swn[1,0] - Ssn[1,0], Ssn[0,0]]
#
v = [C,D]

a = [[simplify(diff(ee,vv)) for vv in v] for ee in e]

# b = simplify((Matrix(e) - a*Matrix(v)).subs(xx,x))

# a = simplify((a.subs(xx,x)))
