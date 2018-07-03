import sympy

x,X,Y,c,cw,sphi,cphi,n,p,h = sympy.symbols('x X Y c cw sphi cphi n p h')

e1 = (X-x)*(cw*sympy.sqrt((h + n*p*sphi)**2 + (cphi*n*p - x)**2))
e1 = e1**2
e2 = -(cphi*n*p - x)*(c*sympy.sqrt(Y**2 + (X - x)**2))
e2 = e2**2

expr = e1 - e2

expr = sympy.expand(expr)

d = sympy.collect(expr,x,evaluate=False)

print('P[0]='+str(sympy.simplify(d[x**4])))
print('P[1]='+str(sympy.simplify(d[x**3])))
print('P[2]='+str(sympy.simplify(d[x**2])))
print('P[3]='+str(sympy.simplify(d[x**1])))
print('P[4]='+str(sympy.simplify(d[1])))



# print(sympy.simplify(d[1]))
# print(sympy.simplify(d[x**3]))
# print(sympy.simplify(d[x**2]))
# print(sympy.simplify(d[x**1]))
# print(sympy.simplify(d[1]))




# T = sympy.sqrt((h+n*p*sphi)**2 + (x-n*p*cphi)**2)/cw + sympy.sqrt((X-x)**2 + Y**2)/c
#
# dTdx = sympy.simplify(sympy.diff(T,x))
# print(dTdx)
