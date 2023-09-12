#!/usr/bin/env python

from scipy.spatial import cKDTree
import random
import os

import numpy as np
import pybullet as p
import pyvista as pv
import pymesh
import trimesh



class ObjDatabaseHandler():

    def __init__(self) -> None:
    
     
        self.path = "/home/mamad/FingersFamily/GraspObjCollection/models/"
        self.obj_name_range = [00,84]
        self.obj_name = "0"+str(self.obj_name_range[0])
        self.candidate_obj = self.obj_name
        self.meshScale = [1,1,1]

        self.maximum_num_pose_in_storage = 15
        self.angle_similarity_threshold = 0.0523599   # three degrees

    def set_candidate(self,obj_id):
      self.obj_name = obj_id
    
    def get_candidate(self):
        candiadate = random.randint(self.obj_name_range[0],self.obj_name_range[-1])
        # print("candiadate::obj:: ",candiadate)
        # candiadate = 3
        self.obj_name = self.create_obj_name(candiadate)

        return self.obj_name

    def get_candidate_path(self):
        
        path = self.path+self.obj_name +"/nontextured_simplified.ply"

        return  path

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
      print("path:: ",path)
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


class MeshCorrespondence:
    def __init__(self, low_res_mesh, high_res_mesh):
        self.low_res_mesh = low_res_mesh
        self.high_res_mesh = high_res_mesh
        self.low_res_rep_points = self.compute_representative_points(low_res_mesh)
        self.high_res_rep_points = self.compute_representative_points(high_res_mesh)
        self.kdtree = cKDTree(self.high_res_rep_points)

    def compute_representative_points(self, mesh):
        # Compute representative points (e.g., centroids) for each face in the mesh
        return np.mean(mesh.points[mesh.faces], axis=1)

    def find_corresponding_faces(self):
        # Find the nearest neighbors for each representative point in the low-res mesh
        _, indices = self.kdtree.query(self.low_res_rep_points, k=1)

        # 'indices' contains the index of the nearest neighbor in the high-res mesh
        return indices


class MeshClustering:
    def __init__(self,low_res,high_res):
        self.low_res = low_res
        self.high_res = high_res
    
    
def convert_pymesh_to_pyvista(pymesh):
        tmesh = trimesh.Trimesh(pymesh.vertices, faces=pymesh.faces, process=False)
        mesh = pv.wrap(tmesh)

        return mesh   
    
if __name__ == '__main__':
    
    loader = ObjDatabaseHandler()
    loader.set_candidate("001")
    loader.get_candidate()
    path = loader.get_candidate_path()
    high_res = convert_pymesh_to_pyvista(pymesh.load_mesh(path))
    path = "./reachability_ff.ply"
    low_res  = convert_pymesh_to_pyvista(pymesh.load_mesh(path))
    
    print("high_res::",high_res)
    print("low_res::",low_res)
    
    print("low_res::faces::",low_res.faces)
    print("low_res::n_points::",low_res.n_points)
    print("low_res::points::",low_res.points)
    

     

    
          