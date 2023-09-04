#!/usr/bin/env python


from  .message import WSData,GraspPose

import pybullet as p
from pkg_resources import resource_string,resource_filename
from .mamad_util import JointInfo

import pyvista as pv
import pymesh
import trimesh

import numpy as np
import random
import os
import logging 
from pynput import keyboard


from .com import Client


fingers = ["palm","cube"]
current_finger = "cube"


class Keyboard_control():
  def __init__(self):
    self.pos = {
      "palm":[0]*6,
    
      "cube":[0]*6
    }

    self.keyboard_layout = {
        # Z axis 
        "z+":keyboard.KeyCode.from_char('o'),
        "z-":keyboard.KeyCode.from_char('l'),
         # X axis 
         "x+":keyboard.KeyCode.from_char(";"),
         "x-":keyboard.KeyCode.from_char('k'),
         # Y axis 
         "y+":keyboard.KeyCode.from_char('x'),
         "y-":keyboard.KeyCode.from_char('z'),
        #  Z Rotation 
        "zr+":keyboard.KeyCode.from_char('u'),
        "zr-":keyboard.KeyCode.from_char('i'),
        #  X Rotation 
        "xr+":keyboard.KeyCode.from_char('y'),
        "xr-":keyboard.KeyCode.from_char('h'),
        #  Y Rotation 
        "yr+":keyboard.KeyCode.from_char('e'),
        "yr-":keyboard.KeyCode.from_char('d')
    }

    self.nav_keys = self.keyboard_layout.values()

    self.dx = 0.01
    self.dz = 0.01
    self.dy = 0.01

    self.dxr = 0.01
    self.dzr = 0.01
    self.dyr = 0.01

  def run(self):
    listener = keyboard.Listener(
    on_press=self.process_keyboard)
   
    listener.start()
    
  def process_palm(self,key):

    # print("palm::process_palm")


    dx = self.dx
    dz = self.dz
    dy = self.dy

    dxr = self.dxr
    dzr = self.dzr
    dyr = self.dyr

    # *************X*************
    if key == self.keyboard_layout["x+"]:
        self.pos["palm"][0] +=dx
    if key == self.keyboard_layout["x-"]:
        self.pos["palm"][0] -=dx
    # *************Y*************
    if key   == self.keyboard_layout["y+"]:
       self.pos["palm"][1] +=dy
    if key   == self.keyboard_layout["y-"]:
       self.pos["palm"][1] -=dy
    # *************Z**************
    if key == self.keyboard_layout["z+"]:
        self.pos["palm"][2] +=dz
    if key == self.keyboard_layout["z-"]:
        self.pos["palm"][2] -=dz
    

    # *************XR*************
    print("self.keyboard_layout[xr+]:: ",self.keyboard_layout["xr+"])
    print("self.keyboard_layout[xr-]:: ",self.keyboard_layout["xr-"])
    if key == self.keyboard_layout["xr+"]:
        self.pos["palm"][3] +=dxr
        print("xRotation+")
    if key == self.keyboard_layout["xr-"]:
        self.pos["palm"][3] -=dxr
        print("xRotation-")
    # self.adjust_for_angle_limits(3)
  
    # *************YR*************
    if key   == self.keyboard_layout["yr+"]:
       self.pos["palm"][4] +=dyr
       print("yRotation+")
    if key   == self.keyboard_layout["yr-"]:
       self.pos["palm"][4] -=dyr
       print("yRotation-")
    # self.adjust_for_angle_limits(4)
    # *************ZR**************
    if key == self.keyboard_layout["zr+"]:
        self.pos["palm"][5] +=dzr
        print("zRotation+")
    if key == self.keyboard_layout["zr-"]:
        self.pos["palm"][5] -=dzr
        print("zRotation-")
    # self.adjust_for_angle_limits(5)


  def process_cube(self,key):
    dx = self.dx
    dz = self.dz
    dy = self.dy

    
    # *************Z**************
    if key == self.keyboard_layout["z+"]:
        self.pos["cube"][2] +=self.dz
        # print("up")
    if key == self.keyboard_layout["z-"]:
        self.pos["cube"][2] -=self.dz
        # print("down")
    # *************X*************
    if key == self.keyboard_layout["x+"]:
        self.pos["cube"][0] +=self.dx
    if key == self.keyboard_layout["x-"]:
        self.pos["cube"][0] -=self.dx
    # *************Y*************
    if key   == self.keyboard_layout["y+"]:
       self.pos["cube"][1] +=self.dy
    if key   == self.keyboard_layout["y-"]:
       self.pos["cube"][1] -=self.dy


    # *************XR*************
    print("self.keyboard_layout[xr+]:: ",self.keyboard_layout["xr+"])
    print("self.keyboard_layout[xr-]:: ",self.keyboard_layout["xr-"])
    if key == self.keyboard_layout["xr+"]:
        self.pos["cube"][3] +=self.dxr
        print("xRotation+")
    if key == self.keyboard_layout["xr-"]:
        self.pos["cube"][3] -=self.dxr
        print("xRotation-")
    # self.adjust_for_angle_limits(3)
  
    # *************YR*************
    if key   == self.keyboard_layout["yr+"]:
       self.pos["cube"][4] +=self.dyr
       print("yRotation+")
    if key   == self.keyboard_layout["yr-"]:
       self.pos["cube"][4] -=self.dyr
       print("yRotation-")
    # self.adjust_for_angle_limits(4)
    # *************ZR**************
    if key == self.keyboard_layout["zr+"]:
        self.pos["cube"][5] +=self.dzr
        print("zRotation+")
    if key == self.keyboard_layout["zr-"]:
        self.pos["cube"][5] -=self.dzr
        print("zRotation-")


  def process_keyboard(self,key):
     global current_finger
     is_nav_key = key in self.nav_keys

    #  print("current_finger:: ",current_finger)
    #  print("key:: ",key)
    #  print("key::type ",type(key))
    #  print("is_nav_key:: ",is_nav_key)
     if key   ==keyboard.KeyCode.from_char('1'):
         current_finger = "palm"

     elif key ==keyboard.KeyCode.from_char('2'):
         current_finger = "cube"

     elif is_nav_key:
        if current_finger   == "palm":
           self.process_palm(key)
        elif current_finger == "cube":
           self.process_cube(key)
      
  def get_state(self):
    return self.pos

  # utility function 
  def adjust_for_angle_limits(self,index):
    index_belong_to_rotation = index>2
    if index_belong_to_rotation:

        if self.palm[index] >3.14:
            self.palm[index] =3.14

        if self.palm[index] <0:
            self.palm[index] =0

class ReachabilityMapRequester(Client):
    def __init__(self):
        super().__init__()
        self.mesh_data = {
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None
        }
        self.ws_flags = {
            "ff":False,
            "mf":False,
            "rf":False,
            "th":False
        }
        
    def send_ws_poses_and_wait_for_result(self,graspPose):
        
        data = self.send(graspPose)
        self.mesh_data,self.ws_flags =self.convert_rosWs_to_ws(data)
        
    
    def get_update(self):
        return self.mesh_data,self.ws_flags 
    
    
    
    # utility function
    def convert_rosWs_to_ws(self,rosWs):
        # getting mesh data 
        ws_meshData = {
     
            "th":None,
            "ff":None,
            "mf":None,
            "rf":None
        }

        ros_ws_dic  ={
     
            "th":rosWs.meshes.th,
            "ff":rosWs.meshes.ff,
            "mf":rosWs.meshes.mf,
            "rf":rosWs.meshes.rf
        }
        for finger in ["th","ff","mf","rf"]:
            ws_meshData[finger]=self.convert_rosMesh_to_MeshData(ros_ws_dic[finger])

        # getting flags
       
        flags_dic ={
                "th":rosWs.flags.th,
                "ff":rosWs.flags.ff,
                "mf":rosWs.flags.mf,
                "rf":rosWs.flags.rf
            } 
 

        return ws_meshData,flags_dic
    
    def convert_rosMesh_to_MeshData(self,rosMesh):
        MeshData = {
            "face_indexs":None,
            "points":None
        }
        points = rosMesh.vertices
        triangles = rosMesh.triangles

        MeshData["points"]      = self.convert_point_to_list(points)
        MeshData["face_indexs"] = self.convert_mesh_trangle_to_list(triangles)

        # print("MeshData::points:: ",MeshData["points"])
        # print("MeshData::face_indexs::type:: ",type(MeshData["face_indexs"]))

        return MeshData
        
    def convert_point_to_list(self,rosPoints):
        point_list = []

        for ros_p in rosPoints:
            point  = [ros_p.x,ros_p.y,ros_p.z]
            point_list.append(point)

        return point_list
        
    def convert_mesh_trangle_to_list(self,rosTriangles):
        face_indexes = []

        for tri in rosTriangles:
            face_indexes.append(list(tri.vertex_indices))

        return np.array(face_indexes).flatten().tolist()
       

class ObjDatabaseHandler():

    def __init__(self,physics_engine) -> None:
        self._p = physics_engine
        self.path = "../../../../GraspObjCollection/models/"
        self.obj_name_range = [00,84]
        self.obj_name = "0"+str(self.obj_name_range[0])
        self.candidate_obj = self.obj_name
        self.meshScale = [1,1,1]

        self.maximum_num_pose_in_storage = 15
        self.angle_similarity_threshold = 0.0523599   # three degrees

    def get_candidate(self):
        candiadate = random.randint(self.obj_name_range[0],self.obj_name_range[-1])
        # print("candiadate::obj:: ",candiadate)
        candiadate = 3
        self.obj_name = self.create_obj_name(candiadate)

        return self.obj_name

    def load_candidate(self,pose):
        # print("load_candidate::pose:: ",pose)
        """
        visual_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH,
                                    fileName=self.path+self.obj_name+"/textured.obj",
                                    meshScale=self.meshScale
                                    )



        model_id = self._p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=visual_id,
                      basePosition=pose,
                      useMaximalCoordinates=0
                      )
        """




        model_id = pybulletPLYLoader(self._p).load(self.path+self.obj_name+"/nontextured__sf.obj",pose)

        path = self.path+self.obj_name

        return model_id , path

    def store_acceptable_obj_spawn_pose(self,pose):
      """
      This is a pose that make object intersect with thumb and one of finger workspaces.
      In addition the object can not be in collision with the hand
      This points will be sampled at random later on to save time 
      """
      path = self.create_file_if_does_not_exist() 
      if self.has_maximum_num_storage_has_been_reached() or self.is_similar_pose_exist(pose):
        # TODO: Raise a warning 
        return 
      with open(path, 'a') as f:
        f.write(str(pose)+"\n")

    def get_pose_from_storage(self):
      path = self.path+self.obj_name+"/"+self.obj_name+"_pose.txt"
      there_is_a_file = os.path.exists(path)
      if there_is_a_file:
        num_poses = self.get_num_poses_in_file()
        candidate = random.randint(0,num_poses)
        return self.get_pose_from_file(candidate)
      else:
        # TODO: raise an error 
        print("there is no sotrage")

    def remove_model(self,model_id):
        self._p.removeBody(model_id)

    def there_is_enough_poses_in_storage(self):
      if self.get_num_poses_in_file()+1 >= self.maximum_num_pose_in_storage:
        return True
      else:
        return False
        
    # utility functions
    def create_file_if_does_not_exist(self):
      path = self.path+self.obj_name+"/"+self.obj_name+"_pose.txt"
    #   print("path:: ",path)
      there_is_a_file = os.path.exists(path)

      if not there_is_a_file:
        f = open(path, "w")

      
      return path
      
    def create_obj_name(self,num):
      """
      there is two cases when integet has
      signle element 0 or double element 00
      """
      str_num = str(num)
      length = len(str_num)
      name =""

      if length ==1:
        name = "00"+str_num
      elif length ==2:
        name = "0"+str_num
      else:
        # TODO: Raise an error 
        print("The range is not correct is shoud be of format 000")

      return name

    def get_num_poses_in_file(self):
      """
      this is equal to number of lines
      """
      count=0
      path = self.create_file_if_does_not_exist()
      with open(path, 'r') as fp:
        for count, line in enumerate(fp):
          pass
      print('Total Lines', count + 1)
      return count

    def get_pose_from_file(self,line_num):
      pose_str = None
      path = self.path+self.obj_name+"/"+self.obj_name+"_pose.txt"
      with open(path, 'r') as fp:
        for count, line in enumerate(fp):
          if count ==line_num:
            pose_str = line

      print("pose_str:: ",pose_str)
      pose_str = pose_str.replace("["," ")
      pose_str = pose_str.replace("]"," ")
      print("pose_str:: ",pose_str)
      pose_str = pose_str.split(",")
      # remove brackets

      print("pose_str:: ",pose_str)
      pose_str = [float(i) for i in pose_str]
      print("pose_str::",pose_str)

      return pose_str

    def has_maximum_num_storage_has_been_reached(self):
      num_poses = self.get_num_poses_in_file()+1

      if  num_poses >= self.maximum_num_pose_in_storage:
        return True
      return False

    def is_similar_pose_exist(self,pose):
      num_poses = self.get_num_poses_in_file()
      for line in range(num_poses):
        read_pose = self.get_pose_from_file(line)
        if self.are_poses_similar(read_pose,pose):
          return True

      return False
        
    def are_poses_similar(self,pose_one,pose_two):
      """
      The positions will be similar so we do not take them into considereation
      The Orientation is what we compare The difference should be more than three degree
      """

      orn_euler_one =  p.getEulerFromQuaternion(pose_one[3:])
      orn_euler_two =  p.getEulerFromQuaternion(pose_two[3:])

      for i in range(3):
        if abs(orn_euler_one[i]-orn_euler_two[i]) > self.angle_similarity_threshold:
          return True
      
      return False

class pybulletPLYLoader():
    
    def __init__(self,physicsEngine):
        self._p = physicsEngine

    def load(self,path,pose):
        
        object_path   = resource_filename(__name__ , path)
        pymesh_mesh = pymesh.load_mesh(object_path)
        pyvista_mesh = self.convert_pymesh_to_pyvista(pymesh_mesh)
        mesh =  MeshDic(pyvista_mesh).get_MeshData()

        meshData={
                        "face_indexs": mesh["face_indexs"].flatten(),
                        "points"     : mesh["points"]
            }  

        model_id = self.create_mesh_visual(meshData,pose)

        return model_id

    # utility functions 

    def convert_pymesh_to_pyvista(self,pymesh):
        tmesh = trimesh.Trimesh(pymesh.vertices, faces=pymesh.faces, process=False)
        mesh = pv.wrap(tmesh)

        return mesh 

    def create_mesh_visual(self,MeshData,pose,color=[0,0,0,0.2]):
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMesh.py
        # print("create_mesh_visual:: ",pose)
        # print("pybulletPLYLoader::create_mesh_visual::MeshData:: ",MeshData)
        
        visual_id  = self._p.createVisualShape(self._p.GEOM_MESH, vertices = MeshData["points"], indices = MeshData["face_indexs"],rgbaColor=color)
        
        model_id = self._p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=visual_id,
                      basePosition=pose,
                      useMaximalCoordinates=0
                      )

        return model_id

class MeshDic:
    def __init__(self,mesh):
       
        self.current_pos =[0]*6

        self.model =  mesh

        self.MeshData = {
            "face_indexs":None,
            "face_normals":None,
            "points":None,

        }
    
    def move_model(self,pose):
        dx = pose[0] -self.current_pos[0] 
        dy = pose[1] -self.current_pos[1] 
        dz = pose[2] -self.current_pos[2] 

        dxr = pose[3] -self.current_pos[3] 
        dyr = pose[4] -self.current_pos[4] 
        dzr = pose[5] -self.current_pos[5] 

        # translation 
        self.model.translate((dx, dy, dz), inplace=True)

        # rotation 
        delta_degree_x = math.degrees(dxr)
        delta_degree_y = math.degrees(dyr)
        delta_degree_z = math.degrees(dzr)

        self.model.rotate_x(delta_degree_x, point=pose[:3], inplace=True)
        self.model.rotate_y(delta_degree_y, point=pose[:3], inplace=True)
        self.model.rotate_z(delta_degree_z, point=pose[:3], inplace=True)

        # update current pose to new pose        
        if dx !=0 or dy !=0 or dz !=0 or dxr !=0 or dyr !=0 or dzr !=0:
            self.current_pos =pose

    def get_MeshData(self):
        faces    = self.model.faces
        reshaped_faces = faces.reshape(-1,4)
        cleaned_faces = reshaped_faces[:,1:]
        # print("faces::type ",type(faces))
        # print("faces:: ",faces)
        # print("faces::reshape ",reshaped_faces)
        # print("faces::cleaned_faces ",cleaned_faces)
        # print("faces::cleaned_faces::shape ",cleaned_faces.shape)

        n_face   = self.model.n_faces
        # print("n_face:: ",n_face)
        n_points = self.model.n_points
        points   = self.model.points
        # print("points::shape:: ",points.shape)
        # print("points:: ",points)

        face_normals = self.model.face_normals
        # print("face_normals:: ",face_normals)

        self.MeshData["face_indexs"]  = cleaned_faces
        self.MeshData["face_normals"] = face_normals
        self.MeshData["points"]       = points


        return self.MeshData
      
class ReachabilityMaper:
    
    def __init__(self,renders=True,timeStep=100):
        self._p = p
        self._render = renders
        self._timeStep = 1/timeStep

        self.objDsHandler = ObjDatabaseHandler(self._p)
        self.graspObjId = None


        #connecting to a physic server
        if self._render:
          cid = self._p.connect(self._p.SHARED_MEMORY)
          if (cid<0):
             id = self._p.connect(self._p.GUI)
          self._p.resetDebugVisualizerCamera(cameraDistance=0.8,cameraYaw=135,cameraPitch=-20,cameraTargetPosition=[0,0,0])
        else:
          self._p.connect(self._p.DIRECT)
        
        ###### scene ######
        self.assets  =[]
        self.color = {
            "red"   :[0.8,0.0,0.2,1],
            "gray"  :[0.7,0.7,0.8,1],
            "blue"  :[0.3,0.2,1.0,1],
            "yellow":[1.0,1.0,0.3,1]
        }
        self.ws_color = {
            "ff":self.color["red"],
            "mf":self.color["gray"],
            "rf":self.color["blue"],
            "th":self.color["yellow"],
        }

        self.outer_boundary = {
            "visual_id":None,
            "shapeType":p.GEOM_SPHERE,
            "rgbaColor":[0.5, 0.8, 0.8, 0.0],
            "specularColor":[0.4, .4, 0],
            "radius":0.14,

            "model_id":None,
            "baseMass":0,
            "basePosition":None,
            "useMaximalCoordinates":0
            
        }
        self.inner_boundary = {
            "visual_id":None,
            "shapeType":p.GEOM_SPHERE,
            "rgbaColor":[0.8, 0.8, 0.8, 0.0],
            "specularColor":[0.4, .4, 0],
            "radius":0.02,

            "model_id":None,
            "baseMass":0,
            "basePosition":None,
            "useMaximalCoordinates":0
            
        }

        self.cube = {
            "path":None,
            "path_texture":None,
            "texture_id":None,
            "model_id":None,
            "pose":{
                "xyz":[0,0,0],
                "rpy":(0,0,0,1)
                },
            "rgbaColor" :[1,1,1,1]

        }
        self.robot = {
            "path":None,
            "model":None,
            "model_id":None,
            "pose":{
                "xyz":[0,0,0.5],
                "rpy":(0,0,0,1)
                },
        }

        self.client_response_models = {
            "id":{
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None,
                "obj":None
            }
        }
        print("Loading scene")
        self.load_scene()


        
     
        ######setting sim parameters#####
        
        print("TEST MESSAGE: Loading scene")
        self._p.setTimeStep(self._timeStep)
        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0,0,0)

        #jointInfo
        self.jointInfo = JointInfo()
        self.jointInfo.get_infoForAll_joints(self.robot["model"])
        ######Communication#####
        self.client = ReachabilityMapRequester()
        self.rmaps = None
        self.ws_flags = None
        self.there_has_been_an_update = False
        self.random_val = 0.000001

        print("self._p",self._p)

    def reset(self):
        """
        remove the old object and randomly load a new object 
        """
        logging.debug("ReachabilityMaper::reset::called")
        self._p.resetSimulation()
        self.load_scene()
        if self.cube["model_id"]:
            self.objDsHandler.remove_model(self.cube["model_id"])
        
        pose = None
        self.graspObjId = self.objDsHandler.get_candidate()
        if self.objDsHandler.there_is_enough_poses_in_storage():
            pose = self.objDsHandler.get_pose_from_storage()
            print("getting pose from storage")
        else:
            print("Generating new pose")
            orn = self.generate_rpy()
            pos = self.generate_xyz()
            pose = list(pos)+list(orn)

        self.cube["model_id"],path = self.objDsHandler.load_candidate(pose)
        path +="/nontextured_simplified.ply" 
      
    def step(self,kb_state):
        
        state_obj  = kb_state["cube"]
        state_hand = kb_state["palm"]
     
        pos_obj = state_obj[:3]
        orn_obj = state_obj[3:]
        orn_obj = self._p.getQuaternionFromEuler(orn_obj)

        pos_hand = state_hand[:3]
        orn_hand = state_hand[3:]
        orn_hand = self._p.getQuaternionFromEuler(orn_hand)

        self._p.resetBasePositionAndOrientation(self.robot["model_id"],pos_hand,orn_hand)
        self._p.resetBasePositionAndOrientation(self.cube["model_id"],pos_obj,orn_obj)
       
        self._p.stepSimulation()


        # if self.are_in_collition():
        #     return self.step()
        
        wses_obs = self.get_ws_obs()
        cube_obs = self.get_obj_obs()



        self.rmaps,self.ws_flags = self.client.get_update()
       
        self.load_client_reponse_into_scene()
        # print("ReachabilityMaper::step::self.rmaps::obj:: ",self.rmaps["obj"])

        if self.there_has_been_an_update:
            pose_msg = self.produce_msg(cube_obs,wses_obs)
            self.client.send_ws_poses_and_wait_for_result(pose_msg)
            self.there_has_been_an_update = False
                 
    def get_obj_obs(self):
        state = self._p.getBasePositionAndOrientation(self.cube["model_id"])
        pos = state[0]
        Quat_orn = state[1]
        orn = self._p.getEulerFromQuaternion(Quat_orn)

        return pos+orn

    def get_ws_obs(self):
        wses_name = ["FF","MF","RF","TH"]
        obs_dic = {
            "FF":None,
            "MF":None,
            "RF":None,
            "TH":None,
            "Palm":None
        }
        for ws_name in wses_name:
            index = self.get_link_index("ws_ee_"+ws_name)
            state  =self._p.getLinkState(self.robot["model_id"],index)
            pos = state[0]
            orn = self._p.getEulerFromQuaternion(state[1])
            state = pos+orn
            obs_dic[ws_name] = state

        # palm ee 
        index  = self.get_link_index("palm_fake")
        state = self._p.getLinkState(self.robot["model_id"],index) 
        pos = state[0]
        orn = self._p.getEulerFromQuaternion(state[1])
        state = pos+orn
        obs_dic["Palm"] = state

        return obs_dic
    
    # utility functions 
    def create_mesh_visual(self,MeshData,pose,color=[0,0,0,0.3]):
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMesh.py
        # print("create_mesh_visual:: ",pose)
        # print("create_mesh_visual::MeshData::points::",MeshData["points"])
        # print("create_mesh_visual::MeshData::face_indexs::",MeshData["face_indexs"])
        
        visual_id  = self._p.createVisualShape(self._p.GEOM_MESH, vertices = MeshData["points"], indices = MeshData["face_indexs"],rgbaColor=color)
        
        model_id = self._p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=visual_id,
                      basePosition=pose,
                      useMaximalCoordinates=0
                      )

        return model_id

    def produce_msg(self,cube_obs,wses_obs):
        graspPose = GraspPose()

        graspPose.obj_id =  self.graspObjId
       
        graspPose.obj    = cube_obs
        graspPose.ff     = wses_obs["FF"]
        graspPose.mf     = wses_obs["MF"]
        graspPose.rf     = wses_obs["RF"]
        graspPose.th     = wses_obs["TH"]
 
        return graspPose
 
    def get_link_index(self,ws_name):
        
        link_name =ws_name

        link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
        palm_link_jointInfo = self.jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]

        palm_link_Index  = palm_link_jointInfo["jointIndex"]

        return palm_link_Index

    def load_scene(self):
        ###### load operational boundry of the hand- the boundry is made of a sphere #####
                        ####### outer boundry #########
        # creating visual
        self.outer_boundary["visual_id"]  = self._p.createVisualShape(shapeType=self.outer_boundary["shapeType"],
                                                                rgbaColor=self.outer_boundary["rgbaColor"],
                                                                specularColor=self.outer_boundary["specularColor"],
                                                                radius = self.outer_boundary["radius"]
                                    )
        # adding visual to the scene 
        self.outer_boundary["model_id"]=self._p.createMultiBody(baseMass=self.outer_boundary["baseMass"],
                      baseVisualShapeIndex=self.outer_boundary["visual_id"],
                      basePosition=self.outer_boundary["basePosition"],
                      useMaximalCoordinates=self.outer_boundary["useMaximalCoordinates"])

        self.assets.append(self.outer_boundary["model_id"])

                        ####### inner boundry #########
        # creating visual
        self.inner_boundary["visual_id"]  = self._p.createVisualShape(shapeType=self.inner_boundary["shapeType"],
                                                                rgbaColor=self.inner_boundary["rgbaColor"],
                                                                specularColor=self.inner_boundary["specularColor"],
                                                                radius = self.inner_boundary["radius"]
                                    )
        # adding visual to the scene 
        self.inner_boundary["model_id"]=self._p.createMultiBody(baseMass=self.inner_boundary["baseMass"],
                      baseVisualShapeIndex=self.inner_boundary["visual_id"],
                      basePosition=self.inner_boundary["basePosition"],
                      useMaximalCoordinates=self.inner_boundary["useMaximalCoordinates"])

        self.assets.append(self.inner_boundary["model_id"])
   
        ##### load the hand inside the boundry #######

        self.robot["path"] = resource_filename(__name__,"/model/model_full.sdf")
        self.robot["model"] = self._p.loadSDF( self.robot["path"])
        self.robot["model_id"]= self.robot["model"][0]
        self._p.resetBasePositionAndOrientation(self.robot["model_id"],self.robot["pose"]["xyz"],self.robot["pose"]["rpy"])

    def spawn_pos_inside_boundry(self):
        """
        This function propose a spawn point that is between innter boundry and outer boundey
        This condition should be checked for parm and base link
        """
        x = self.genrate_a_random_number_between_innter_and_outer_radius()
        y = self.genrate_a_random_number_between_innter_and_outer_radius()
        z = self.genrate_a_random_number_between_innter_and_outer_radius()

        pos =(x,y,z)

        # return pos

        if self.is_outside_inner_boundry(pos) and self.is_insdie_outer_boundry(pos):
            return pos
        else:
            return self.spawn_pos_inside_boundry()

    def is_outside_inner_boundry(self,pos):
        """
        x^2+y^2+z^2 > r_inner
        """
        x,y,z = pos
        result = x**2+y**2+z**2 > self.inner_boundary["radius"]**2
        # print("is_outside_inner_boundry::result:: ",result)
        return result

    def is_insdie_outer_boundry(self,pos):
        """
        x^2+y^2+z^2 < r_outer
        """
        x,y,z = pos
        result = x**2+y**2+z**2 < self.outer_boundary["radius"]**2
        # print("is_insdie_outer_boundry::result:: ",result)
        return result

    def genrate_a_random_number_between_innter_and_outer_radius(self):
        """
        1. find a number according to  abs(num)>r_innter and abs(num)<r_outer
        2. assign a sign at random
        """
        sign = 1
        number = random.uniform(self.inner_boundary["radius"], self.outer_boundary["radius"])
    
        if random.randint(0, 1)==0:
            sign = -1
        
        number  *=sign

        return number

    def generate_rpy(self):
        """
        generate a number between [-3.14, 3.14] radian  equivelant to [-180,180] degree

        """
        orn = {
            "r":None,
            "p":None,
            "y":None,
        }
        for key in orn.keys():
            orn[key] = random.uniform(-3.14,3.14)
        
        quat = self._p.getQuaternionFromEuler((orn["r"],orn["p"],orn["y"]))

        # print("orn:: ",orn)
        # print("quat:: ",quat)

        return quat

    def generate_xyz(self,limit = 0.05):  
        x= random.uniform(-limit,limit)
        y= random.uniform(-limit,limit)
        z= random.uniform(-limit,limit)
        return (x,y,z)
    
    def are_in_collition(self):
        contact = self._p.getContactPoints(self.robot["model_id"],self.cube["model_id"])
        print("contact:: ",contact)

    def load_client_reponse_into_scene(self):
        if self.rmaps["ff"]:
            print("load_client_reponse_into_scene::self.rmaps::ff::keys ",self.rmaps["ff"].keys())
            print("load_client_reponse_into_scene::self.rmaps::ff::points ",self.rmaps["ff"]["points"])
            print("load_client_reponse_into_scene::self.rmaps::ff::face_indexs ",self.rmaps["ff"]["face_indexs"])
            
        print("load_client_reponse_into_scene::self.rmaps:: ",self.rmaps)
        
        if not self.is_rmaps_valid():
            return 
        
        pose = [0]*3+[0,0,0,1]

        for finger in ["ff","mf","rf","th"]:
            if self.ws_flags[finger]:        
                self.client_response_models["id"][finger] = self.create_mesh_visual(self.rmaps[finger],pose,color=self.ws_color[finger])
    
    def remove_client_reponse_from_scene(self):
        for key in self.client_response_models["id"].keys():
            if self.client_response_models["id"][key]:
                self._p.removeBody(self.client_response_models["id"][key])
                self.client_response_models["id"][key] = None
        
        self._p.stepSimulation()

    def there_is_a_need_for_update(self):
        self.there_has_been_an_update = True   
    
    def is_rmaps_valid(self):
        print("is_rmaps_is_valid:: self.ws_flags:: ", self.ws_flags)

        if self.ws_flags: 
            ws_flags = [self.ws_flags["ff"],self.ws_flags["mf"],self.ws_flags["rf"],self.ws_flags["th"]]
            print("ws_flags:: ",ws_flags)
            wees_flag_count = self.indexs(ws_flags,True)
            count = len(wees_flag_count)
            if count > 0:
                return True
            
        return False

    def indexs(self,seq,item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item,start_at+1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs
      
    def create_mesh_visual(self,MeshData,pose,color=[0,0,0,0.3]):
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMesh.py
        # print("create_mesh_visual:: ",pose)
        # print("create_mesh_visual::MeshData::points::",MeshData["points"])
        # print("create_mesh_visual::MeshData::face_indexs::",MeshData["face_indexs"])
        
        visual_id  = self._p.createVisualShape(self._p.GEOM_MESH, vertices = MeshData["points"], indices = MeshData["face_indexs"],rgbaColor=color)
        
        model_id = self._p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=visual_id,
                      basePosition=pose,
                      useMaximalCoordinates=0
                      )

        return model_id


rendered_reachabilityMap = True

if __name__ == '__main__':
    Rmapper = ReachabilityMaper()
    kb = Keyboard_control()
    kb.run()
    Rmapper.reset()

    space_key =32

    while(1):
		
        key_pressed = p.getKeyboardEvents()
        kb_state = kb.get_state()
        # print("kb_state:: ",kb_state)
        Rmapper.step(kb_state)

        if len(key_pressed.keys()) >0:
            # print("keys.keys():: ",key_pressed.keys())
            # print("len(keys.keys()):: ",len(key_pressed.keys()))

            key = list(key_pressed.keys())[0]
            if key==space_key:
                Rmapper.there_is_a_need_for_update()
        
        
       
       
 
            


        
     
