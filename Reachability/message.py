#!/usr/bin/env python
import numpy as np

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
        self.low_resolution_meshes  = MeshData()
        self.high_resolution_meshes  = MeshData()
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
 
class FingerTips:
    def __init__(self,ff_point=None,mf_point=None,rf_point=None,th_point=None):
        # type :Point 
        self.ff = ff_point 
        self.mf = mf_point 
        self.rf = rf_point 
        self.th = th_point 
        
        
    def get_dic(self):
        dic = {
            "ff":self.ff.get_numpy(),
            "mf":self.mf.get_numpy(),
            "rf":self.rf.get_numpy(),
            "th":self.th.get_numpy()
        }
        return dic
    
    
class ReachabilityGoals:
    def __init__(self,ff_goals,mf_goals,rf_goals,th_goals):
        # list of Point 
        self.ff = ff_goals
        # list of Point
        self.mf = mf_goals
        # list of Point
        self.rf = rf_goals
        # list of Point
        self.th = th_goals
        
    @classmethod
    def from_dic(cls,goals_dic):
        ff = [Point.from_list(point.tolist()) for point in goals_dic["ff"]]
        mf = [Point.from_list(point.tolist()) for point in goals_dic["mf"]]
        rf = [Point.from_list(point.tolist()) for point in goals_dic["rf"]]
        th = [Point.from_list(point.tolist()) for point in goals_dic["th"]]
        
        return cls(ff,mf,rf,th)
         
    
        
class ReachabilityMap:
    def __init__(self,ff_mesh=None,mf_mesh=None,rf_mesh=None,th_mesh=None):
        # type : Mesh 
        self.ff = ff_mesh
        self.mf = mf_mesh
        self.rf = rf_mesh
        self.th = th_mesh
    def get_dic(self):
        print("ReachabilityMap::get_dic")
        dic = {
            "ff":self.ff.get_dic(),
            "mf":self.mf.get_dic(),
            "rf":self.rf.get_dic(),
            "th":self.th.get_dic()
        }
        return dic
    
class PreGrasp:
    def __init__(self,radius=0.05,fingertips=None,reachabilityMaps=None):
        # type : FingerTips 
        self.fingertips = fingertips
        # type : ReachabilityMap
        self.reachabilityMaps = reachabilityMaps 
        self.radius =radius

        
class  Mesh:
    def __init__(self,triangles = None,vertices= None,normals=None):
        # type: MeshTriangle 
       self.triangles = triangles 
        # type: Point 
       self.vertices  = vertices
       #type:Point
       self.normals = normals
       
    @classmethod
    def from_dic(cls,dic):
        triangles = dic["face_indexs"]
        vertices  = [Point.from_list(point_list) for point_list in dic["points"]]
        normals   = [Point.from_list(point_list) for point_list in dic["normals"]]
        
        return cls(triangles,vertices,normals)
    
    def get_dic(self):
        dic = {
            "face_indexs":None,
            "points":None,
            "normals":None,
        }
        dic["face_indexs"]=self.triangles
        dic["points"] = np.array([point.get_numpy()  for point in self.vertices])
        dic["points"] = np.array([normal.get_numpy() for normal in self.normals])
        
        return dic
    
    
    
class Point:
    def __init__(self,x=None,y=None,z=None):
        self.x =x
        self.y =y
        self.z =z
        
        
    @classmethod
    def from_list(cls, point_list):
        x, y, z = point_list[0],point_list[1],point_list[2]
        return cls(x, y, z)
       
    @classmethod
    def from_tuple(cls, point_tuple):
        x, y, z = point_tuple
        return cls(x, y, z)

    @classmethod
    def from_dict(cls, point_dict):
        return cls(point_dict.get('x'), point_dict.get('y'), point_dict.get('z'))
    
    
    def get_numpy(self):
        return np.array([self.x,self.y,self.z])

    
        
class MeshTriangle:
    def __init__(self,vertex_indices=None):
        # type: uint32[3] 
        self.vertex_indices = vertex_indices
    
    
 