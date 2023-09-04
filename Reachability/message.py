#!/usr/bin/env python

class Position:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
class Oreintation:
    def __init__(self,r,p,y):
        self.r = r
        self.p = p
        self.y = y
        
class Pose:
    def __init__(self,x,y,z,r,p,yy):
        self.position = Position(x,y,z)
        self.oreintation = Oreintation(r,p,yy)

class GraspPose:
    def __init__(self,obj_id=None,ff=None,mf=None,rf=None,th=None,obj=None):
        self.obj_id = obj_id
        # type:float[] xyz_rpy 
        self.ff  = ff
        self.mf  = mf
        self.rf  = rf
        self.th  = th
        self.obj = obj
        


class WSData:
    def __init__(self):
        self.meshes  = MeshData()
        self.flags = WSFlags()
        

class WSFlags:
    def __init__(self):
        self.ff  = False
        self.mf  = False
        self.rf  = False
        self.th  = False
        
class MeshData:
    def __init__(self,ff=None,mf=None,rf=None,th=None,obj=None):
        # type : Mesh 
        self.ff  = ff
        self.mf  = mf
        self.rf  = rf
        self.th  = th
        self.obj = obj
        
        

        
class  Mesh:
    def __init__(self,triangles = None,vertices= None):
        # type: MeshTriangle 
       self.triangles = triangles 
        # type: Point 
       self.vertices  = vertices
    
    
class Point:
    def __init__(self,x=None,y=None,z=None):
        self.x =x
        self.y =y
        self.z =z
        
class MeshTriangle:
    def __init__(self,vertex_indices=None):
        # type: uint32[3] 
        self.vertex_indices = vertex_indices
    
    
 