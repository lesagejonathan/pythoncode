from sympy import *

# x,X,Y,l0,l1,l2,l3,m,p,cphi,sphi,c,h,cw = symbols('x,X,Y,l0,l1,l2,l3,m,p,cphi,sphi,c,h,cw')

x,X,Y,h,m,p,cphi,sphi,c,cw = symbols('x,X,Y,h,m,p,cphi,sphi,c,cw')

# T = sqrt((m*p))


# T = sqrt((x))


# x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,X0,X1,l0,l1,l2,l3,m,n,p,cphi,sphi,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,h,cw = symbols('x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,X0,X1,l0,l1,l2,l3,m,n,p,cphi,sphi,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,h,cw')

# x0,x1,x2,X,Y,l0,m,p,cphi,sphi,c0,c1,c2,h,cw = symbols('x0,x1,x2,X,Y,l0,m,p,cphi,sphi,c0,c1,c2,h,cw')

# x0,x1,x2,l0,l1,l2,l3,h,c0,c1,c2,cw,m,p,sphi,cphi,X,Y = symbols('x0,x1,x2,l0,l1,l2,l3,h,c0,c1,c2,cw,m,p,sphi,cphi,X,Y')

# T = sqrt((x0-m*p*cphi)**2 + (h+m*p*sphi)**2)/cw + sqrt((x1-x0)**2 + l0**2)/c0 + sqrt((l1-x1)**2 + (l0-x2)**2)/c1 + sqrt((l1-x3)**2 + x2**2)/c2 + sqrt((x3-n*p*cphi)**2 + (h+n*p*sphi)**2)/cw

# T = []

# T2 = T**2

# y3 = -(l3/l2)*x3 + l3*(l1-l2)/l2

# y2 = -(l3/l2)*x2 + l3*(l1-l2)/l2
#
# y4 = -(l3/l2)*x4 + l3*(l1-l2)/l2
#
# y3 = -(l3/l2)*x4 + l3*(l1-l2)/l2
#
#

# y4 = -(l3/l2)*x4 + l3*(l1-l2)/l2
# y6 = -(l3/l2)*x6 + l3*(l1-l2)/l2
#
# T = []
#

T = []

T.append(sqrt((x-m*p*cphi)**2 + (h+m*p*sphi)**2)/cw)

T.append(sqrt((X-x)**2 + Y**2)/c)

T = sum(T)


#
# T.append(sqrt((x2-x1)**2 + l0**2)/c1)
#
# T.append(sqrt((l1-x2)**2 + x3**2)/c2)
#
# T.append(sqrt((x4-l1)**2 + (l0-x3)**2)/c3)
#
# T.append(sqrt((x5-x4)**2 + l0**2)/c4)
#
# T.append(sqrt((x5-n*p*cphi)**2 + (h+n*p*sphi)**2)/cw)
#
#
#


# T.append(sqrt((x3-x2)**2 + l0**2)/c2)
#
# T.append(sqrt((x4-x3)**2 + (y4-l0)**2)/c3)
#
# T.append(sqrt((l1-x4)**2 + (x5-y4)**2)/c4)
#
# T.append(sqrt((x6-l1)**2 + (y6-x5)**2)/c5)
#
# T.append(sqrt((x7-x6)**2 + (l0-y6)**2)/c6)
#
# T.append(sqrt((x8-x7)**2 + l0**2)/c7)
#
# T.append(sqrt((x9-x8)**2 + l0**2)/c8)
#
# T.append(sqrt((X0-x9)**2 + l0**2)/c9)
#
# T.append(sqrt((X1*cphi -X0)**2 + (h+X1*sphi)**2)/cw)

# T.append(sqrt((x2-x1)**2 + (y2-l0)**2)/c1)
#
# T.append(sqrt((l1-x2)**2 + (x3-y2)**2)/c2)
#
# T.append(sqrt((x4-l1)**2 + (y4-x3)**2)/c3)
#
# T.append(sqrt((x5-x4)**2 + (l0-y4)**2)/c4)
#
# T.append(sqrt((x6-x5)**2 +l0**2)/c5)
#
# T.append(sqrt((x7-x6)**2 +l0**2)/c6)
#
# T.append(sqrt((x8-x7)**2 + l0**2)/c7)
#
# T.append(sqrt((x9*cphi-x8)**2 + (h+x9*sphi)**2)/cw)


# T.append(sqrt((X-x2)**2 + Y**2)/c2)
#
# T.append(sqrt((x2-x1)**2 + (y2-l0)**2)/c2)
#
# # T.append(sqrt((X-x2)**2 + (Y-y2)**2)/c2)
#
# # T.append(sqrt((x2-x1)**2 + (y2-l0)**2)/c1)
# #
# T.append(sqrt((l1-x2)**2 + (y2-x3)**2)/c2)
# #
# T.append(sqrt((l1-x4)**2 + (y3-x3)**2)/c3)
#
# T.append(sqrt((x5-x4)**2 + (l0-y3)**2)/c4)
#
# T.append(sqrt((x5-x6)**2 + l0**2)/c5)
# #
# T.append(sqrt((x7*cphi-x6)**2 + (h+x7*sphi)**2)/cw)
#
# # T = 2*sum(T)
#
# T = sum(T)

T = simplify(T)

J = simplify(diff(T,x))
#
#
# x = [x0,x1,x2,x3,x4,x5]

# x = [x0,x1,x2,x3]

# x = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,X0,X1]

# J = [diff(T2,xx) for xx in x]
#
# H = [[diff(JJ,xx) for xx in x] for JJ in J]
#
# print(simplify(T2))

# J = [simplify(diff(T,xx)) for xx in x]

# H = [[diff(JJ,xx) for xx in x] for JJ in J]

# print(simplify(T))

print(T)

print(J)

# print(H)
