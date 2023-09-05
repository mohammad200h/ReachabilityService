#!/usr/bin/env python

import pybullet as p
import os


from pkg_resources import resource_string,resource_filename
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np

import math
import random

import pymesh
import trimesh
import copy

import plotly.graph_objects as go
import time


from threading import Thread
from .com import Server

from .message import WSData,GraspPose,Mesh,Point,MeshTriangle
import cupy as cp

import sys

# adding axis:
# https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_axes.html


# https://stackoverflow.com/questions/69099877/update-pyvista-plotter-with-new-scalars
# https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.update_scalars.html
# https://stackoverflow.com/questions/63429806/plotting-multiple-meshes-on-1-figure-with


# https://docs.pyvista.org/api/utilities/_autosummary/pyvista.utilities.axis_rotation.html
# https://answers.ros.org/question/362734/how-to-get-the-transformation-matrix-from-translation-and-rotation/
# https://www.programcreek.com/python/example/12960/vtk.vtkTransform
# https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.html


import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper


class ClusteringUsingAnglesBetweenNormals:
    def __init__(self,mesh):
        self.mesh = mesh
        self.clusters ={
            "n_unique_normals":None,
            "unique_normals_list":[],
            "unique_normals":{},
            "clusters":{}
        }
    @timing_decorator  
    def get_clusterGPU(self):
        normals = cp.asarray(self.mesh["face_normals"])
        working_normals =  cp.asarray(self.mesh["face_normals"])
        
        max_size = 10000
        
        counter = 0
        n_unique_normals= 0
        cluster_bound = cp.full((max_size,1),-1,dtype=cp.int64)
        unique_normals_list= cp.full((max_size,3),666, dtype=cp.float64) # upper bound is  n_unique_normals | max storage is max_size | nonsense value is 666 |
        unique_normals= cp.full((max_size,3),666, dtype=cp.float64)   # upper bound is  counter | max storage is max_size | nonsense value is 666 |
        clusters= cp.full((max_size,max_size),-1, dtype=cp.int64) # upper bound is  counter | max storage is max_size | nonsense value is -1 |
        
        # print("normals::type::",type(normals))
        normals_list_length = normals.shape[0]
        all_indices = [i for i in  range(0,normals_list_length)]
        
        visited_indeces = cp.empty(0, dtype=cp.float64)
        
        size_of_working_normals = 0
        while(working_normals.shape[0]!=0  ):
            # print("visited_indeces:: ",visited_indeces)
            # print("Clustering::get_cluster::while(working_normals.shape[0]!=0  )")
            current_normal = working_normals[0]
            # print("current_normal:: ",current_normal)
           

            dot_product   = np.dot(normals,current_normal)
            normals_pow   = np.power(normals,2)
            normals_sum   = np.sum(normals_pow,axis=1)
            normals_sqrt  = np.sqrt(normals_sum)

            # print("dot_product:: ",dot_product)
            # print("normals[0]:: ",normals[0])
            # print("normals_pow[0]:: ",normals_pow[0])
            # print("normals_sum[0]:: ",normals_sum[0])
            # print("normals_sqrt[0]:: ",normals_sqrt[0])

            current_normal_pow  = np.power(current_normal,2)
            current_normal_sum  = np.sum(current_normal_pow)
            current_normal_sqrt = np.sqrt(current_normal_sum)

              
            # print("current_normal_pow:: ",current_normal_pow)
            # print("current_normal_sum:: ",current_normal_sum)
            # print("current_normal_sqrt:: ",current_normal_sqrt)
            
            threshold_degree_in_radian =0.0001745329
            nenomerator = dot_product
            denomenator = normals_sqrt*current_normal_sqrt
            cos_theta   = nenomerator/denomenator
            cos_theta = np.where(cos_theta >1,1,cos_theta) # dealing with nan situation
            cos_theta = np.where(cos_theta <-1,-1,cos_theta) # dealing with nan situation
            theta = np.arccos(cos_theta)
            indexes  = np.where(theta<threshold_degree_in_radian)[0]

            # print("nenomerator:: ",nenomerator)
            # print("denomenator:: ",denomenator)
            # print("cos_theta:: ",cos_theta)
            # print("np.min(cos_theta):: ",np.min(cos_theta))
            # print("np.max(cos_theta):: ",np.max(cos_theta))


            # print("theta:: ",theta)
            # print("cos_theta[0]:: ",cos_theta[0])
            # print("theta[0]:: ",theta[0])
            # print("cos_theta[1]:: ",cos_theta[1])
            # print("theta[1]:: ",theta[1])
            # print("indexes:: ",indexes)
            # break
            have_indices_been_visted = np.isin(cp.asarray(indexes),cp.asarray(visited_indeces))
            is_there_a_repeated_element = True in have_indices_been_visted 
            
           
            # print("indexes:: ",indexes)
            # print("visited_indeces:: ",visited_indeces)
            
            visited_indeces = np.concatenate((visited_indeces, indexes))
            clusters[counter][:indexes.shape[0]] = indexes
            cluster_bound[counter]=indexes.shape[0]
            
            # print("have_indices_been_visted:: ",have_indices_been_visted)
            # print("is_there_a_repeated_element:: ",is_there_a_repeated_element)
            
            
            # working_normals = cp.delete(normals,cp.asarray(visited_indeces),axis=0)
            working_normals = normals[cp.logical_not(cp.isin(cp.arange(normals.shape[0]), visited_indeces))]
            # print("working_normals:: ",working_normals)
            there_is_no_matching_normal = size_of_working_normals == working_normals.shape[0]
            if there_is_no_matching_normal:
                index_of_current_normal = np.where(normals==current_normal)[0]
                visited_indeces = np.concatenate((index_of_current_normal,visited_indeces))
                continue

            size_of_working_normals = working_normals.shape[0]

            
            unique_normals_list[counter,:] = current_normal
            unique_normals[counter,:]      = current_normal
            counter +=1
            # print(" normals.shape[0]:: ", normals.shape[0])
            # print(" working_normals.shape[0]:: ", working_normals.shape[0])
            
        n_unique_normals = counter
        
        
        #####crop arrays to the only field part#####
        croped_unique_normals_list= unique_normals_list[:counter,:]
        croped_unique_normals     = unique_normals[:counter,:]
        croped_clusters           = clusters[:counter,:]
        croped_cluster_bound      = cluster_bound[:counter]
        
        return croped_unique_normals_list,croped_unique_normals,croped_clusters,croped_cluster_bound
      
    @timing_decorator  
    def get_cluster(self):
        normals = self.mesh["face_normals"]
        working_normals = self.mesh["face_normals"]
        
        # print("normals::type::",type(normals))
        normals_list_lenght = normals.shape[0]
        all_indicies = [i for i in  range(0,normals_list_lenght)]
        # print("normals_list_lenght:: ",normals_list_lenght)
        # print("all_indicies:: ",all_indicies)
        visited_indeces = []
        counter = 0
        size_of_working_normals = 0
        while(working_normals.shape[0]!=0  ):
            # print("visited_indeces:: ",visited_indeces)
            # print("Clustering::get_cluster::while(working_normals.shape[0]!=0  )")
            current_normal = working_normals[0]
            # print("current_normal:: ",current_normal)
           

            dot_product = np.dot(normals,current_normal)
            normals_pow = np.power(normals,2)
            normals_sum = np.sum(normals_pow,axis=1)
            normals_sqrt = np.sqrt(normals_sum)

            # print("dot_product:: ",dot_product)
            # print("normals[0]:: ",normals[0])
            # print("normals_pow[0]:: ",normals_pow[0])
            # print("normals_sum[0]:: ",normals_sum[0])
            # print("normals_sqrt[0]:: ",normals_sqrt[0])

            current_normal_pow  = np.power(current_normal,2)
            current_normal_sum  = np.sum(current_normal_pow)
            current_normal_sqrt = np.sqrt(current_normal_sum)

              
            # print("current_normal_pow:: ",current_normal_pow)
            # print("current_normal_sum:: ",current_normal_sum)
            # print("current_normal_sqrt:: ",current_normal_sqrt)
            
            threshold_degree_in_radian =0.0001745329
            nenomerator = dot_product
            denomenator = normals_sqrt*current_normal_sqrt
            cos_theta   = nenomerator/denomenator
            cos_theta = np.where(cos_theta >1,1,cos_theta) # dealing with nan situation
            cos_theta = np.where(cos_theta <-1,-1,cos_theta) # dealing with nan situation
            theta = np.arccos(cos_theta)
            indexes  = np.where(theta<threshold_degree_in_radian)[0].tolist()

            # print("nenomerator:: ",nenomerator)
            # print("denomenator:: ",denomenator)
            # print("cos_theta:: ",cos_theta)
            # print("np.min(cos_theta):: ",np.min(cos_theta))
            # print("np.max(cos_theta):: ",np.max(cos_theta))


            # print("theta:: ",theta)
            # print("cos_theta[0]:: ",cos_theta[0])
            # print("theta[0]:: ",theta[0])
            # print("cos_theta[1]:: ",cos_theta[1])
            # print("theta[1]:: ",theta[1])
            # print("indexes:: ",indexes)
            # break
            have_indices_been_visted = np.isin(np.array(indexes),np.array(visited_indeces))
            is_there_a_repeated_element = True in have_indices_been_visted 
            
           
            # print("indexes:: ",indexes)
            # print("visited_indeces:: ",visited_indeces)
            visited_indeces += indexes
            self.clusters["clusters"][counter] = indexes
            
            
            # print("have_indices_been_visted:: ",have_indices_been_visted)
            # print("is_there_a_repeated_element:: ",is_there_a_repeated_element)
            
            
            working_normals = np.delete(normals,visited_indeces,axis=0)
            # print("working_normals:: ",working_normals)
            there_is_no_matching_normal = size_of_working_normals == working_normals.shape[0]
            if there_is_no_matching_normal:
                index_of_current_normal = np.where(normals==current_normal)[0].tolist()
                visited_indeces += index_of_current_normal
                continue

            size_of_working_normals = working_normals.shape[0]

            self.clusters["unique_normals_list"].append(current_normal)
            self.clusters["unique_normals"][counter]=current_normal
            counter +=1
            # print(" normals.shape[0]:: ", normals.shape[0])
            # print(" working_normals.shape[0]:: ", working_normals.shape[0])
            
        self.clusters["n_unique_normals"] = counter

        # print(self.clusters)
        return self.clusters
  
class FaceOperations():
  

    def get_shared_normals_between_two_mesh(self,cluster_a,cluster_b):
        A = np.array(cluster_a["unique_normals_list"])
        B = np.array(cluster_b["unique_normals_list"])

        nrows, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
               'formats':ncols * [A.dtype]}

        C = np.intersect1d(A.view(dtype), B.view(dtype),return_indices=True)

        # print("c:: ",C)
        # print("c::1 ",C[1])
        # print("c::2 ",C[2])

        return C[1],C[2]
    
    def get_shared_normals_between_two_mesh_approximate(self,cluster_a,cluster_b):
        return self.find_index_of_normal_that_are_almost_identtical(cluster_a,cluster_b)
      
    def get_shared_normals_between_two_mesh_approximateGPU(self,a_unique_normals_list ,b_unique_normals_list):
        return self.find_index_of_normal_that_are_almost_identticalGPU(a_unique_normals_list ,b_unique_normals_list)
        
    def find_index_of_normal_that_are_almost_identticalGPU(self,a_unique_normals_list ,b_unique_normals_list):
        
          
        
        # print("cube::unique_normals:: ",a_unique_normals_list)
        # (a,b)
        match_list = []
        visited_indeces = cp.empty(0, dtype=cp.int64)

        working_normals = a_unique_normals_list
        # print("working_normals::shape",working_normals.shape)
        normals  = b_unique_normals_list
        # print("normals::shape",normals.shape)
        counter = 0
        while(working_normals.shape[0]!=0):
            current_normal = working_normals[0]

            dot_product = np.dot(normals,current_normal)
            normals_pow = np.power(normals,2)
            normals_sum = np.sum(normals_pow,axis=1)
            normals_sqrt = np.sqrt(normals_sum)

            current_normal_pow  = np.power(current_normal,2)
            current_normal_sum  = np.sum(current_normal_pow)
            current_normal_sqrt = np.sqrt(current_normal_sum)

            threshold_degree_in_radian = 0.0001745329
            nenomerator = dot_product
            denomenator = normals_sqrt*current_normal_sqrt
            cos_theta   = nenomerator/denomenator
            cos_theta = np.where(cos_theta >1,1,cos_theta)
            cos_theta = np.where(cos_theta <-1,-1,cos_theta)
            theta = np.arccos(cos_theta)
            # print("current_normal:: ",current_normal)
            # print("theta:: ",theta)
            indexes  = np.where(theta<threshold_degree_in_radian)[0]
            if indexes.shape[0]>0:
                match_list.append((counter,indexes))

            visited_indeces =np.concatenate((visited_indeces,cp.array([counter])))

            working_normals = a_unique_normals_list[cp.logical_not(cp.isin(cp.arange(a_unique_normals_list.shape[0]), visited_indeces))]
            
            

            counter +=1

        
        # print("match_list:: ",match_list)
        # print("cluster_a[clusters]:: ",cluster_a["clusters"].keys())
        # print("cluster_b[clusters]:: ",cluster_b["clusters"].keys())

     
        A = []
        B = []

        for tup in match_list:
            # print("a::\n")
            # print("\tIndex"+str(tup[0])+"::Normal::",a_unique_normals_list[tup[0]])
            A +=[tup[0]]
            # print("b::\n")
            for index in tup[1]:
                # print("\tIndex"+str(index)+"::Normal::",b_unique_normals_list[index])
                B +=[index]

        # print("A::", A)
        # print("B::", B)
        return A,B
  
    
    def find_index_of_normal_that_are_almost_identtical(self,cluster_a,cluster_b):
        a_unique_normals_list = cluster_a["unique_normals_list"]
        b_unique_normals_list = cluster_b["unique_normals_list"]
        

        # print("cube::unique_normals:: ",a_unique_normals_list)
        # (a,b)
        match_list = []
        visited_indeces = []

        working_normals =np.array( a_unique_normals_list)
        # print("working_normals::shape",working_normals.shape)
        normals  = np.array(b_unique_normals_list)
        # print("normals::shape",normals.shape)
        counter = 0
        while(working_normals.shape[0]!=0):
            current_normal = working_normals[0]

            dot_product = np.dot(normals,current_normal)
            normals_pow = np.power(normals,2)
            normals_sum = np.sum(normals_pow,axis=1)
            normals_sqrt = np.sqrt(normals_sum)

            current_normal_pow  = np.power(current_normal,2)
            current_normal_sum  = np.sum(current_normal_pow)
            current_normal_sqrt = np.sqrt(current_normal_sum)

            threshold_degree_in_radian = 0.0001745329
            nenomerator = dot_product
            denomenator = normals_sqrt*current_normal_sqrt
            cos_theta   = nenomerator/denomenator
            cos_theta = np.where(cos_theta >1,1,cos_theta)
            cos_theta = np.where(cos_theta <-1,-1,cos_theta)
            theta = np.arccos(cos_theta)
            # print("current_normal:: ",current_normal)
            # print("theta:: ",theta)
            indexes  = np.where(theta<threshold_degree_in_radian)[0].tolist()
            if len(indexes)>0:
                match_list.append((counter,indexes))

            visited_indeces += [counter]

            working_normals = np.delete(np.array( a_unique_normals_list),visited_indeces,axis=0)
            

            counter +=1

        
        # print("match_list:: ",match_list)
        # print("cluster_a[clusters]:: ",cluster_a["clusters"].keys())
        # print("cluster_b[clusters]:: ",cluster_b["clusters"].keys())

     
        A = []
        B = []

        for tup in match_list:
            # print("a::\n")
            # print("\tIndex"+str(tup[0])+"::Normal::",a_unique_normals_list[tup[0]])
            A +=[tup[0]]
            # print("b::\n")
            for index in tup[1]:
                # print("\tIndex"+str(index)+"::Normal::",b_unique_normals_list[index])
                B +=[index]

        # print("A::", A)
        # print("B::", B)
        return A,B
  
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
    
    def generate_an_empty_mesh(self):
        return self.MeshData

class PyvistaMesh():
    def __init__(self,meshData):
        self.meshData=  meshData
       

    def create_mesh_with_indices(self,indeces):
      
        print("create_mesh_with_indices::indeces::",indeces)
        
        vertices = self.meshData["points"]
        faces =  self.meshData["face_indexs"]
        selected_faces = self.get_faces_matching_indeces(faces,indeces)
        tmesh = trimesh.Trimesh(vertices, faces=selected_faces, process=False)
        mesh = pv.wrap(tmesh)

        return mesh

    def get_faces_matching_indeces(self,faces,indeces):
        selected_faces = faces[indeces]
        # print("selected_faces:: ",selected_faces)
        return selected_faces

    def save_mesh(self,mesh,name):
        mesh.save(name+".ply")

class ReducingMeshQaulity:
    def __init__(self,path) -> None:
        

        self.mesh = pymesh.load_mesh(path)
        self.mesh =  self.convert_pymesh_to_pyvista(self.mesh)

    def convert_pymesh_to_pyvista(self,pymesh):
        tmesh = trimesh.Trimesh(pymesh.vertices, faces=pymesh.faces, process=False)
        mesh = pv.wrap(tmesh)

        return mesh

    def reduce_resolution(self,target_reduction=0.9,repeat=2):
        # https://docs.pyvista.org/examples/01-filter/decimate.html
        
        low_res_mesh = self.mesh

        for i in range(repeat):
            low_res_mesh = low_res_mesh.decimate(target_reduction)
        return low_res_mesh

class ObjDatabaseHandler():

    def __init__(self) -> None:
    
        # self.path = "../../../../GraspObjCollection/models/"
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

class ReachabilityMaps():
    
    def __init__(self):
        ########## state ##########
        self.actor = {
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None,
          "working_copy":None,
          "inter":{
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None
          },
          "reachability":{
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None
          }
        }
        self.current_pos = {
          "ff":[0]*6,
          "mf":[0]*6,
          "rf":[0]*6,
          "th":[0]*6,
          "palm":[0]*6,
          "obj":[0]*6
        }
        self.new_pos = {
          "ff":[0]*6,
          "mf":[0]*6,
          "rf":[0]*6,
          "th":[0]*6,
          "palm":[0]*6,
          "obj":[0]*6
        }

        self.objDataHandler = ObjDatabaseHandler()
        self.coordinate ={
          "x":None,
          "y":None,
          "z":None
        }
        self.obj= {
          "obj_id":None,
          "untouched":None,
          "working_copy":None
        }
        
        self.ws_mesh = {
            "untouched":{
              "ff":None,
              "mf":None,
              "rf":None,
              "th":None,
            },
            "working_copy":{
              "ff":None,
              "mf":None,
              "rf":None,
              "th":None,
            },
           

            "inter":{
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None
            },
            
            "reachability":{
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None
            },
            
            "cleaned_inter":{
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None
            },
            
            "clean_reachability":{
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None
            },
        }
        
        self.collision_flags = {
          "ff":False,
          "mf":False,
          "rf":False,
          "th":False
        }
        self.reachability_map_calculated=True
        
        self.load_obj()
        self.load_coordinate()

        self.obj_id = None
        ########## visulization ##########
        self.color = {
            "working_copy":"gray",
            "x":"red",
            "y":"green",
            "z":"blue",
            "ff":"white",
            "mf":"white",
            "rf":"white",
            "th":"white"

        }

        
        self.sub_plotter = BackgroundPlotter(window_size=(1000,1000),shape=(2, 2))
        self.sub_plotter.add_axes(line_width=5, labels_off=True)
      
        # self.plotter  = BackgroundPlotter(window_size=(1000,1000))

        # We create a blue point light at the position of the sphere
        self.light = pv.Light((0, 0, 1), (0, 0, 0), 'blue')
        
    def render(self):
        self.sub_plotter.add_light(self.light)
        self.sub_plotter.add_callback(self.update_scene, interval=100)

        self.sub_plotter.subplot(0, 0)
        self.actor["working_copy"] = self.sub_plotter.add_mesh(self.obj["working_copy"] ,color=self.color["working_copy"], label='working_copy')

        self.sub_plotter.add_mesh(self.coordinate["x"] ,color=self.color["x"], label='x')
        self.sub_plotter.add_mesh(self.coordinate["y"] ,color=self.color["y"], label='y')
        self.sub_plotter.add_mesh(self.coordinate["z"] ,color=self.color["z"], label='z')


        self.actor["ff"] = self.sub_plotter.add_mesh(self.ws_mesh["working_copy"]["ff"] ,color=self.color["ff"], label='ff')
        self.actor["mf"] = self.sub_plotter.add_mesh(self.ws_mesh["working_copy"]["mf"] ,color=self.color["mf"], label='mf')
        self.actor["rf"] = self.sub_plotter.add_mesh(self.ws_mesh["working_copy"]["rf"] ,color=self.color["rf"], label='rf')
        self.actor["th"] = self.sub_plotter.add_mesh(self.ws_mesh["working_copy"]["th"] ,color=self.color["th"], label='th')
        
        self.sub_plotter.show()
        self.sub_plotter.app.exec_() 

    def load_obj(self):
      
        obj_path   = resource_filename(__name__,"./nontextured_simplified.ply")

        self.objDataHandler.get_candidate()
        path = self.objDataHandler.get_candidate_path()
        # self.obj["untouched"] = ReducingMeshQaulity(obj_path).reduce_resolution()
        self.obj["untouched"] = ReducingMeshQaulity(path).reduce_resolution()
        self.obj["working_copy"] = self.obj["untouched"].copy()

        thumb_path  = resource_filename(__name__,"/model/meshes/ws/TH/TH_ws_cleaned_ADJUSTED_origin.ply")
        finger_path = resource_filename(__name__,"/model/meshes/ws/FF/FF_ws_cleaned_ADJUSTEDorigine.ply")

        for finger in ["ff","mf","rf"]:
          self.ws_mesh["working_copy"][finger]   = pymesh.load_mesh(finger_path)
          self.ws_mesh["working_copy"][finger]  = self.convert_pymesh_to_pyvista(self.ws_mesh["working_copy"][finger])
          self.ws_mesh["untouched"][finger] = self.ws_mesh["working_copy"][finger].copy()

        self.ws_mesh["working_copy"]["th"]   = pymesh.load_mesh(thumb_path)
        self.ws_mesh["working_copy"]["th"]  = self.convert_pymesh_to_pyvista(self.ws_mesh["working_copy"]["th"])
        self.ws_mesh["untouched"]["th"] = self.ws_mesh["working_copy"]["th"].copy()
        
    def load_coordinate(self):
      path_x = resource_filename(__name__,"/coordinate_frame/x.obj")
      path_y = resource_filename(__name__,"/coordinate_frame/y.obj")
      path_z = resource_filename(__name__,"/coordinate_frame/z.obj")

      self.coordinate["x"] =  pymesh.load_mesh(path_x)
      self.coordinate["y"] =  pymesh.load_mesh(path_y)
      self.coordinate["z"] =  pymesh.load_mesh(path_z)

      self.coordinate["x"] = self.convert_pymesh_to_pyvista(self.coordinate["x"])
      self.coordinate["y"] = self.convert_pymesh_to_pyvista(self.coordinate["y"])
      self.coordinate["z"] = self.convert_pymesh_to_pyvista(self.coordinate["z"])

    def update_pose(self,pose_dic):
        """
        this function is used for reseting the service 
        getting ready for new caclulation
        """
        self.update_pose_of_obj(pose_dic)
        self.update_fingers_ws_pose(pose_dic)
        
    def update_mesh_inplace(self):
        d_dic = {}
        

        dx = self.new_pos["obj"][0] -self.current_pos["obj"][0] 
        dy = self.new_pos["obj"][1] -self.current_pos["obj"][1] 
        dz = self.new_pos["obj"][2] -self.current_pos["obj"][2] 

        dxr = self.new_pos["obj"][3] -self.current_pos["obj"][3] 
        dyr = self.new_pos["obj"][4] -self.current_pos["obj"][4] 
        dzr = self.new_pos["obj"][5] -self.current_pos["obj"][5] 
        
        # print("update_mesh_inplace::obj::dx::",dx)
        # print("update_mesh_inplace::obj::dy::",dy)
        # print("update_mesh_inplace::obj::dz::",dz)
        # print("update_mesh_inplace:: self.new_pos[obj::", self.new_pos["obj"])
        # print("update_mesh_inplace:: self.current_pos[obj]::", self.current_pos["obj"])
               
        # translation 
        self.obj["working_copy"].translate((dx, dy, dz), inplace=True)
            
        delta_degree_x = math.degrees(dxr)
        delta_degree_y = math.degrees(dyr)
        delta_degree_z = math.degrees(dzr)
            
        self.obj["working_copy"].rotate_x(delta_degree_x, self.new_pos["obj"][:3], inplace=True)
        self.obj["working_copy"].rotate_y(delta_degree_y, self.new_pos["obj"][:3], inplace=True)
        self.obj["working_copy"].rotate_z(delta_degree_z, self.new_pos["obj"][:3], inplace=True)
            
        d_dic["working_copy"] = (dx,dy,dz,dxr,dyr,dzr)

        fingers  =["ff","mf","rf","th"]
        for finger in fingers:
        
            dx = self.new_pos[finger][0] -self.current_pos[finger][0] 
            dy = self.new_pos[finger][1] -self.current_pos[finger][1] 
            dz = self.new_pos[finger][2] -self.current_pos[finger][2] 

            dxr = self.new_pos[finger][3] -self.current_pos[finger][3] 
            dyr = self.new_pos[finger][4] -self.current_pos[finger][4] 
            dzr = self.new_pos[finger][5] -self.current_pos[finger][5] 
            
            # translation 
            self.ws_mesh["working_copy"][finger].translate((dx, dy, dz), inplace=True)
            
            delta_degree_x = math.degrees(dxr)
            delta_degree_y = math.degrees(dyr)
            delta_degree_z = math.degrees(dzr)
            
            self.ws_mesh["working_copy"][finger].rotate_x(delta_degree_x, self.new_pos[finger][:3], inplace=True)
            self.ws_mesh["working_copy"][finger].rotate_y(delta_degree_y, self.new_pos[finger][:3], inplace=True)
            self.ws_mesh["working_copy"][finger].rotate_z(delta_degree_z, self.new_pos[finger][:3], inplace=True)

            d_dic[finger] = (dx,dy,dz,dxr,dyr,dzr)


        return d_dic  

    def update(self):
        d_dic=  self.update_mesh_inplace()
        self.perform_intersection_pyvista()
        self.get_reachability_map()

        meshes  =["working_copy","th","ff","mf","rf"]
        for mesh in meshes:
            dx,dy,dz,dxr,dyr,dzr = d_dic[mesh]
            if dx !=0 or dy !=0 or dz !=0 or dxr !=0 or dyr !=0 or dzr !=0:
            
                self.current_pos[mesh] =self.new_pos[mesh][:]
    
    def update_scene(self):

        perfrom_operation = False
        d_dic=  self.update_mesh_inplace()
        for key in d_dic.keys():
          dx,dy,dz,dxr,dyr,dzr = d_dic[key]
          if  dx !=0 or dy !=0 or dz !=0 or dxr !=0 or dyr !=0 or dzr !=0:
            perfrom_operation = True
            if self.actor[key]:
              self.sub_plotter.subplot(0, 0)
              self.sub_plotter.remove_actor( self.actor[key])
              if key =="working_copy":
                self.actor[key] = self.sub_plotter.add_mesh(self.obj[key] ,color=self.color[key], label='working_copy')
              else:
                self.actor[key] = self.sub_plotter.add_mesh(self.ws_mesh["working_copy"][key] ,color=self.color[key], label='ff')
              if key =="working_copy":
                self.current_pos["obj"] =self.new_pos["obj"][:]
              else:
                self.current_pos[key] =self.new_pos[key][:]
      
        if perfrom_operation:
          self.perform_intersection_pyvista()
          self.get_reachability_map()
        self.sub_plotter.update()
        # print("ReachabilityMaps::update_scene::called")

        # self.operation_is_complete = True
      
     

    def get_rechability_map_data(self):

      while(self.wait_for_result()):
        time.sleep(0.1)
        
      reachability_dic = {
            "ff": self.ws_mesh["reachability"]["ff"],
            "mf": self.ws_mesh["reachability"]["mf"],
            "rf": self.ws_mesh["reachability"]["rf"],
            "th": self.ws_mesh["reachability"]["th"],
            "obj": self.obj["working_copy"]
      }
      return reachability_dic
      
    # utility function

    def get_diff(self):
      
      diff_dic = {
        "obj":{
          "dx":None,
          "dy":None,
          "dz":None,
          "dxr":None,
          "dyr":None,
          "dzr":None,
          "flag":False 
        },
        "ff":{
          "dx":None,
          "dy":None,
          "dz":None,
          "dxr":None,
          "dyr":None,
          "dzr":None,
          "flag":False  
        },
        "mf":{
          "dx":None,
          "dy":None,
          "dz":None,
          "dxr":None,
          "dyr":None,
          "dzr":None,
          "flag":False  
        },
        "rf":{
          "dx":None,
          "dy":None,
          "dz":None,
          "dxr":None,
          "dyr":None,
          "dzr":None,
          "flag":False  
        },
        "th":{
          "dx":None,
          "dy":None,
          "dz":None,
          "dxr":None,
          "dyr":None,
          "dzr":None ,
          "flag":False 
        }

      }

      elements  =["obj","ff","mf","rf","th"] 
      
      for element in elements:
        
        
        dx = self.new_pos[element][0] -self.current_pos[element][0] 
        dy = self.new_pos[element][1] -self.current_pos[element][1] 
        dz = self.new_pos[element][2] -self.current_pos[element][2] 
        
        dxr = self.new_pos[element][3] -self.current_pos[element][3] 
        dyr = self.new_pos[element][4] -self.current_pos[element][4] 
        dzr = self.new_pos[element][5] -self.current_pos[element][5] 

        diff_flag = [True if d !=0 else False for d in [dx,dy,dx,dxr,dyr,dxr]]
        diff_flag =  True in diff_flag

        dxr = math.degrees(dxr)
        dyr = math.degrees(dyr)
        dzr = math.degrees(dzr)

        diff_dic[element]["dx"]  = dx
        diff_dic[element]["dy"]  = dy     
        diff_dic[element]["dz"]  = dz

        diff_dic[element]["dxr"]  = dxr
        diff_dic[element]["dyr"]  = dyr     
        diff_dic[element]["dzr"]  = dzr
        diff_dic[element]["flag"] = diff_flag
      print("self.new_pos[obj][0]", self.new_pos["obj"][0])
      print("self.current_pos[obj][0]", self.current_pos["obj"][0])
      print("diff_dic[obj][flag]::", diff_dic["obj"]["flag"])
      return diff_dic 
    
    def wait_for_result(self):
      """
      This will return a flag saying if we should for result to be produced.
      we should wiat till we get reachability map for all meshes in collision
      """
      
      # print("ReachabilityMaps::wait_for_result::.called")

      figners_in_contact = []
      if self.collision_flags["called"]:
        for finger in  self.collision_flags.keys():
          if self.collision_flags[finger]==True:
            figners_in_contact.append(finger)

        if len(figners_in_contact)==0:
          return False
        if  self.reachability_map_calculated:
          return False
   
        

    

      return True
      # return False

    def perform_intersection_pyvista(self):
        print("ReachabilityMaps::perform_intersection_pyvista::called")
        self.sub_plotter.subplot(1, 0)
        for finger in self.actor["inter"].keys():
            
            if self.actor["inter"][finger]:
                self.sub_plotter.remove_actor( self.actor["inter"][finger])

            # clip_surface= self.ws_mesh[finger].copy().scale(1.2,1.2,1.2)
            inter = None
            objects_are_in_collision = self.check_collision(finger)
            self.collision_flags[finger] = objects_are_in_collision
            if objects_are_in_collision:
                inter = self.obj["working_copy"].boolean_intersection(self.ws_mesh["working_copy"][finger])
          
                # inter.save("intersection_"+finger+".ply")

                # print("inter:: ",inter)
                # print("inter::0 ",inter.cell_points(0))
                # print("inter::points ",inter.n_points)
                if inter.n_points>0:
                    self.ws_mesh["inter"][finger] = inter
                    self.actor["inter"][finger] = self.sub_plotter.add_mesh(self.ws_mesh["inter"][finger] ,color=self.color[finger], label=finger)
      
        self.collision_flags["called"]=True
        print("perform_intersection_pyvista::call::ended")
        self.sub_plotter.show()

    def get_reachability_map(self):
        self.sub_plotter.subplot(1, 1)

        print("clean_interscetion_pyvista::called")
  
        for finger in self.actor["reachability"].keys():
            if self.actor["reachability"][finger]:
                self.sub_plotter.remove_actor( self.actor["reachability"][finger])
                
            if self.ws_mesh["inter"][finger]:
                inter = self.get_faces_with_same_normal(self.obj["working_copy"],self.ws_mesh["inter"][finger])

                # inter.save("reachability_"+finger+".ply")

                print("inter:: ",inter)
                print("inter::points ",inter.n_points)
                if inter.n_points>0:
                    self.ws_mesh["reachability"][finger] = inter
                    self.actor["reachability"][finger] = self.sub_plotter.add_mesh(self.ws_mesh["reachability"][finger] ,color=self.color[finger], label=finger)

        
        self.reachability_map_calculated=True
        self.sub_plotter.show()
          
    def reset_ws_mesh(self,obj_id):
      print("reset_ws_mesh::obj_id:: ",obj_id)
      diff_dic = self.get_diff()
      wsees_flags = [diff_dic[finger]["flag"] for finger in ["ff","mf","rf","th"]]
      diff_flag_ws  = True in wsees_flags
      obj_flag = diff_dic["obj"]

      
      
      if diff_flag_ws or (self.obj["obj_id"]==None or self.obj["obj_id"]!=obj_id):
        self.reset_grasp_obj(obj_id)
      if obj_flag:
        self.reset_wses()
      self.collision_flags["called"] = False
      self.reachability_map_calculated=False

      if obj_flag or diff_flag_ws:
        for key in ["inter","cleaned_inter","reachability","clean_reachability"]:
            self.ws_mesh[key] ={
                "ff":None,
                "mf":None,
                "rf":None,
                "th":None
            }
        self.collision_flags = {
          "called":False,
          "ff":False,
          "mf":False,
          "rf":False,
          "th":False
        }

    def get_faces_with_same_normal(self,target_object,ws_int):
        faceOpt =FaceOperations()  
        cube = MeshDic(target_object).get_MeshData()
        cube_cluster = ClusteringUsingAnglesBetweenNormals(cube).get_cluster()
        cube_cluster = ClusteringUsingAnglesBetweenNormals(cube).get_clusterGPU()

        ws_inter = MeshDic(ws_int).get_MeshData()
        ws_inter_cluster = ClusteringUsingAnglesBetweenNormals(ws_inter).get_cluster()
        ws_inter_cluster = ClusteringUsingAnglesBetweenNormals(ws_inter).get_clusterGPU()

        cube_index,ws_inter_index=faceOpt.get_shared_normals_between_two_mesh_approximateGPU(cube_cluster[0],ws_inter_cluster[0])
        # print("cube_index:: ",cube_index)
        # print("ws_inter_index:: ",ws_inter_index)

        ws_inter_faces_to_keep =cp.empty(0, dtype=cp.int64)


        print("get_faces_with_same_normal::ws_inter_cluster::type",type(ws_inter_cluster))
        
        
        
        for index in ws_inter_index:
            unclean_cluster = ws_inter_cluster[2][index]
            cluster_bound_for_this_index= ws_inter_cluster[3][index]
            # bucket = unclean_cluster[index]
            bucket = unclean_cluster[:cluster_bound_for_this_index]
            ws_inter_faces_to_keep =np.concatenate((ws_inter_faces_to_keep,bucket)) 
            # print("bucket:: ",bucket)
  
        PVmesh = PyvistaMesh(ws_inter)

        mesh = PVmesh.create_mesh_with_indices(ws_inter_faces_to_keep.get().tolist())

        return mesh
      

    def check_collision(self,finger):
        self.obj["working_copy"]['collisions'] =  np.zeros(self.obj["working_copy"].n_cells, dtype=bool)
        col, n_contacts = self.obj["working_copy"].collision(self.ws_mesh["working_copy"][finger])
        # print("check_collision::finger::n_contacts ",finger,n_contacts)
        if n_contacts >0:
            return True
        return False

    def convert_pymesh_to_pyvista(self,pymesh):
        tmesh = trimesh.Trimesh(pymesh.vertices, faces=pymesh.faces, process=False)
        mesh = pv.wrap(tmesh)

        return mesh

    def reset_wses(self):
      for finger in ["ff","mf","rf","th"]:
        self.ws_mesh["working_copy"][finger] = self.ws_mesh["untouched"][finger].copy()
    
    def reset_grasp_obj(self,obj_id):
      print("reset_grasp_obj::obj_id::",obj_id)
      print("reset_grasp_obj::self.obj[obj_id]::",self.obj["obj_id"])
      print("reset_grasp_obj::self.obj[obj_id]==obj_id::",self.obj["obj_id"]==obj_id)
      if self.obj["obj_id"]==None or self.obj["obj_id"]!=obj_id:
        self.obj["obj_id"]=obj_id
        self.objDataHandler.set_candidate(obj_id)
        path = self.objDataHandler.get_candidate_path()
        abs_path = os.path.abspath(path)
        print("reset_grasp_obj::abs_path::",abs_path)
        self.obj["untouched"] = ReducingMeshQaulity(path).reduce_resolution()

      self.obj["working_copy"] = self.obj["untouched"].copy()

    def update_pose_of_obj(self,pose_dic):
        
        
        self.current_pos["obj"] =  [0]*6
        self.obj["working_copy"] = self.obj["untouched"].copy()

        self.new_pos["obj"] = pose_dic["obj"]
    
    def update_fingers_ws_pose(self,pose_dic):

      for finger in ["ff","mf","rf","th"]:
        
        
        
        self.current_pos[finger] =  [0]*6
        self.new_pos[finger] = pose_dic[finger]
    
class ReachabilityMapProducer(Server,Thread):

    def __init__(self,redner=True) -> None:
        Thread.__init__(self)
        self.daemon = False       
        super().__init__(task=self.on_goal,sm_bsize=500000)
      
        # state 
        self.hand_pose = None
        self.obj_id =None
        self.start()
        #######ReachabilityMaps####
        self.rm_render = redner
        self.rm = ReachabilityMaps()
        if self.rm_render:
            self.rm.render()
         
    def on_goal(self,goal):
        print("ReachabilityMapProducer::on_goal::called")
        print("ReachabilityMapProducer::goal::",goal)
        ws_data = WSData()
        if self.rm_render:
            ws_data = self.on_goal_proccessing_render(goal)
        else:
            ws_data = self.on_goal_processing(goal)
            
        return ws_data
        
    def get_state(self):
        if self.hand_pose ==None:
            return  self.empty_hand()
        hand = self.hand_pose
        print("get_state::hand:: ",hand)
        return hand

    #utility functions
    def empty_hand(slef):
        hand = GraspPose()
        hand.ff    = [0]*6
        hand.mf    = [0]*6
        hand.rf    = [0]*6
        hand.th    = [0]*6
       

        print("empty_hand::hand:: ",hand)

        return hand

    def on_goal_proccessing_render(self,goal):
        print("ReachabilityMapProducer::on_goal_proccessing_render::called")
        obj_id = goal.obj_id
        self.rm.reset_ws_mesh(obj_id)
      
        pose = {
          "obj" :goal.obj,
          "ff"  :goal.ff,
          "mf"  :goal.mf,
          "rf"  :goal.rf,
          "th"  :goal.th
        }
        
        # print("on_goal_proccessing_render::pose::",pose)

    
        self.rm.update_pose(pose)

        ########## sending data back ##########
        reachability_dic = self.rm.get_rechability_map_data()
        ws_data = WSData()
        mesh_data = {}
      
        
        mesh_flags = {}
       
        for finger in ["th","ff","mf","rf"]:
        
            mesh_data[finger],mesh_flags[finger] = self.get_meshData_msg(reachability_dic[finger],finger)
           

        
        # print("ReachabilityMapProducer::on_goal_proccessing_render::mesh_data[finger]:: " ,mesh_data)
        # print("ReachabilityMapProducer::on_goal_proccessing_render::mesh_flags[finger]:: ",mesh_flags)
 
        ws_data.meshes.th  = mesh_data["th"]
        ws_data.meshes.ff  = mesh_data["ff"]
        ws_data.meshes.mf  = mesh_data["mf"]
        ws_data.meshes.rf  = mesh_data["rf"]
       
        ws_data.flags.th = mesh_flags["th"]
        ws_data.flags.ff = mesh_flags["ff"]
        ws_data.flags.mf = mesh_flags["mf"]
        ws_data.flags.rf = mesh_flags["rf"]
      

        print("on_goal_proccessing_render::ws_data.flags:: ",ws_data.flags)

        print("on_goal_proccessing_render::sending result back")
        return ws_data
        
    def on_goal_processing(self,goal):
        obj_id = goal.obj_id
        self.rm.reset_ws_mesh(obj_id)

        pose = {
          "obj" :goal.obj,
          "ff"  :goal.ff,
          "mf"  :goal.mf,
          "rf"  :goal.rf,
          "th"  :goal.th
        }

        self.rm.update_pose(pose)

        ###########Perform Operation###############
        self.rm.update()
        ###########Send result Back################
        reachability_dic = self.rm.get_rechability_map_data()
        ws_data = WSData()
        mesh_data = {}
        mesh_flags = {}
        
        for finger in ["th","ff","mf","rf"]:
        
            mesh_data[finger],mesh_flags[finger] = self.get_meshData_msg(reachability_dic[finger],finger)
           
           

        # self.ws["th"].save("./th_service_debug.ply")
        ws_data.meshes.th  = mesh_data["th"]
        ws_data.meshes.ff  = mesh_data["ff"]
        ws_data.meshes.mf  = mesh_data["mf"]
        ws_data.meshes.rf  = mesh_data["rf"]
       
        ws_data.flags.th = mesh_flags["th"]
        ws_data.flags.ff = mesh_flags["ff"]
        ws_data.flags.mf = mesh_flags["mf"]
        ws_data.flags.rf = mesh_flags["rf"]
      

        print("on_goal_proccessing_render::ws_data.flags:: ",ws_data.flags)

     
        
        print("on_goal_proccessing_render::sending result back")
        return ws_data

    def get_meshData_msg(self,ws,name):
      mesh = Mesh()

      flag = None
      
      if ws !=None:
         
          # print("mesh:: ",mesh)
          # print("mesh::points:: ",mesh.points)
          mesh_dic        = MeshDic(ws).get_MeshData()
          flag = True
          mesh.vertices  = self.convert_to_point(mesh_dic["points"])
          mesh.triangles = self.convert_to_mesh_triangle( mesh_dic["face_indexs"])
      else:
          mesh_dic        = MeshDic(ws).generate_an_empty_mesh()
          flag = False
          mesh.vertices = []
          mesh.triangles = []


     
      # print("mesh.triangles::",mesh.triangles)
      

      return mesh,flag
  
    def convert_to_point(self,numpy_list):
        point_list = []
        # print("convert_to_point::numpy_list:: ",numpy_list)        
        for p in numpy_list:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            point.z = p[2] 

            point_list.append(point)
        return point_list

    def convert_to_mesh_triangle(self,face_index_list):
        # https://programtalk.com/python-more-examples/shape_msgs.msg.Mesh/
        # http://docs.ros.org/en/lunar/api/shape_msgs/html/msg/MeshTriangle.html
        t_list = []
        
        for triangle in face_index_list:
            meshTriangle = MeshTriangle()
            meshTriangle.vertex_indices = [triangle[0],triangle[1],triangle[2]] 
            t_list.append(meshTriangle)

        return t_list
 

if __name__ == '__main__':
    serv = ReachabilityMapProducer(redner=True)
   

 