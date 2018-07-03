from FreeCAD import Base
import Part
import random

l = [1.7*2,25.2,22.5,20.3,7.85/2,10.7/2,2.69/2,2.7]

# l[0] = Shell thickness
# l[1] = Probe depth
# l[2] = Probe width
# l[3] = Probe height
# l[4] = Side hole radius
# l[5] = Top hole radius
# l[6] = Fasterner hole radius
# l[7] = Distance from edge of shell to fasterner hole centre


S1 = Part.makePolygon([Base.Vector(0,0,0), Base.Vector(l[0],0,0), Base.Vector(l[0],l[1],0), Base.Vector(l[2]+l[0],l[1],0), Base.Vector(l[2]+l[0],0,0), Base.Vector(l[2]+2*l[0],0,0), Base.Vector(l[2]+2*l[0],l[1]+l[0],0), Base.Vector(0,l[1]+l[0],0), Base.Vector(0,0,0)])
S1 = Part.Face(S1)

# S1 = S1.extrude(Base.Vector(0,0,l[3]))
#
# C1 = Part.makeCircle(l[4],Base.Vector(0,l[1]/2,l[3]/2),Base.Vector(1,0,0),0,360)
#
# C1 = Part.Face(Part.Wire(C1))
#
# C2 = Part.makeCircle(l[4],Base.Vector(2*l[0]+l[2],l[1]/2,l[3]/2),Base.Vector(1,0,0),0,360)
#
# C2 = Part.Face(Part.Wire(C2))
#
#
# H1 = C1.extrude(Base.Vector(l[0]/2,0,0))
#
# H2 = C2.extrude(Base.Vector(-l[0]/2,0,0))
#
#
# # H1 = Part.Face(Part.makeCircle(l[4],Base.Vector(0,l[1]/2,l[3]/2),Base.Vector(0,1,0),0,360)).extrude(Base.Vector(0,l[0]/2,0))
# #
# S1 = S1.cut(H1)
#
# S1 = S1.cut(H2)
#
# L = []
#
# L.append(Part.Line(Base.Vector(0,0,l[3]),Base.Vector(0,l[1]+l[0],l[3])))
# L.append(Part.Line(Base.Vector(0,l[1]+l[0],l[3]), Base.Vector(2*l[0]+l[2],l[1]+l[0],l[3])))
# L.append(Part.Line(Base.Vector(2*l[0]+l[2],l[1]+l[0],l[3]),Base.Vector(2*l[0]+l[2],0,l[3])))
# L.append(Part.Line(Base.Vector(2*l[0]+l[2],0,l[3]),Base.Vector(0.5*(2*l[0]+l[2])+l[5],0,l[3])))
# L.append(Part.Line(Base.Vector(0.5*(2*l[0]+l[2])+l[5],0,l[3]),Base.Vector(0.5*(2*l[0]+l[2])+l[5],0.5*l[1],l[3])))
#
#
# L.append(Part.Line(Base.Vector(0.5*(2*l[0]+l[2])+l[5],0.5*l[1],l[3]), Base.Vector(0.5*(2*l[0]+l[2])-l[5],0.5*l[1],l[3])))
# L.append(Part.Line(Base.Vector(0.5*(2*l[0]+l[2])-l[5],0.5*l[1],l[3]),Base.Vector(0.5*(2*l[0]+l[2])-l[5],0,l[3])))
# L.append(Part.Line(Base.Vector(0.5*(2*l[0]+l[2])-l[5],0,l[3]), Base.Vector(0,0,l[3])))
#
# S2 = Part.Shape(L)
#
# S2 = Part.Wire(S2.Edges)
#
# S2 = Part.Face(S2)
#
# S2 = S2.extrude(Base.Vector(0,0,l[0]))
#
#
#
#
#
#
# # A = Part.Arc(Base.Vector(),Base.Vector(0.5*(2*l[0]+l[2])+l[5],0.5*(l[1]+l[0]),l[3]),Base.Vector(0.5*(2*l[0]+l[2])-l[5],0.5*(l[1]+l[0]),l[3]))
#
# H3 = Part.makeCircle(l[5],Base.Vector(0.5*(2*l[0]+l[2]),l[1]/2,l[3]),Base.Vector(0,0,1),0,360)
# H3 = Part.Face(Part.Wire(H3))
#
# H3 = H3.extrude(Base.Vector(0,0,l[0]))
#
# H4 = Part.makeCircle(l[6],Base.Vector(l[7]+l[0],l[1]/2,l[3]),Base.Vector(0,0,1),0,360)
# H4 = Part.Face(Part.Wire(H4))
#
# H4 = H4.extrude(Base.Vector(0,0,l[0]))
#
#
# H5 = Part.makeCircle(l[6],Base.Vector(l[0]+l[2]-l[7],l[1]/2,l[3]),Base.Vector(0,0,1),0,360)
# H5 = Part.Face(Part.Wire(H5))
#
# H5 = H5.extrude(Base.Vector(0,0,l[0]))
#
#
# S = S1.fuse(S2)
#
# S = S.cut(H3)
#
# S = S.cut(H4)
#
# S = S.cut(H5)
#
# Part.show(S)
#
# # L = Part.makePolygon([Base.Vector(0,0,0),Base.Vector(17,0,0),Base.Vector(17,0,-15),Base.Vector(22,0,-15),Base.Vector(22,0,5),Base.Vector(0,0,5),Base.Vector(0,0,0)])
# #
# # L = Part.Face(L)
# #
# # # L = Part.extrude(Base.Vector(0,10,0))
# #
# # l = 10
# #
# # dl = 0.5
# #
# # L = L.extrude(Base.Vector(0,l,0))
# #
# #
# # h = 5.0
# # v = 7.0
# #
# # W=Part.makePolygon([Base.Vector(10,0,0),Base.Vector(17,0,-5),Base.Vector(17,0,0),Base.Vector(10,0,0)])
# # W=Part.Face(W)
# # W=W.extrude(Base.Vector(0,dl,0))
# #
# # y = 0
# #
# # for i in range(int(l/dl)-1):
# #
# #     hh = h+random.uniform(-1,1)
# #     vv = v+random.uniform(-1,1)
# #
# #     y += dl
# #
# #     w = Part.makePolygon([Base.Vector(17-vv,y,0),Base.Vector(17,y,-hh),Base.Vector(17,y,0),Base.Vector(17-vv,y,0)])
# #
# #     w = Part.Face(w)
# #
# #     w = w.extrude(Base.Vector(0,dl,0))
# #
# #     W = W.fuse(w)
# #
# # WL = W.fuse(L)
# #
# # WL.exportStep("/Users/jlesage/Dropbox/Eclipse/VariableCap.step")
