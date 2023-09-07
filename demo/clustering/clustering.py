#!/usr/bin/env python

import numpy as np
import cupy as cp

import numba

import time

import multiprocessing

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper
    
# @numba.jit
def get_clusterNumba(face_normals):
        normals = face_normals
        working_normals =  face_normals
        
        max_size = 10000
        max_size_v =10000
        
        counter = 0
        n_unique_normals= 0
        cluster_bound = np.full((max_size_v,1),-1,dtype=np.int64)
        unique_normals_list= np.full((max_size_v,3),666, dtype=np.float64) # upper bound is  n_unique_normals | max storage is max_size | nonsense value is 666 |
        unique_normals= np.full((max_size_v,3),666, dtype=np.float64)   # upper bound is  counter | max storage is max_size | nonsense value is 666 |
        clusters= np.full((max_size_v,max_size),-1, dtype=np.int64) # upper bound is  counter | max storage is max_size | nonsense value is -1 |
        
        # print("normals::type::",type(normals))
        normals_list_length = normals.shape[0]
        all_indices = [i for i in  range(0,normals_list_length)]
        
        visited_indeces = np.empty(0, dtype=np.float64)
        
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
def get_clusterGPU(face_normals):
        normals = cp.asarray(face_normals)
        working_normals =  cp.asarray(face_normals)
        
        max_size = 10000
        max_size_v =10000
        
        counter = 0
        n_unique_normals= 0
        cluster_bound = cp.full((max_size_v,1),-1,dtype=cp.int64)
        unique_normals_list= cp.full((max_size_v,3),666, dtype=cp.float64) # upper bound is  n_unique_normals | max storage is max_size | nonsense value is 666 |
        unique_normals= cp.full((max_size_v,3),666, dtype=cp.float64)   # upper bound is  counter | max storage is max_size | nonsense value is 666 |
        clusters= cp.full((max_size_v,max_size),-1, dtype=cp.int64) # upper bound is  counter | max storage is max_size | nonsense value is -1 |
        
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
def get_cluster(face_normals):
        
        clusters ={
            "n_unique_normals":None,
            "unique_normals_list":[],
            "unique_normals":{},
            "clusters":{}
        }
        normals = face_normals
        working_normals = face_normals
        
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
            clusters["clusters"][counter] = indexes
            
            
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

            clusters["unique_normals_list"].append(current_normal)
            clusters["unique_normals"][counter]=current_normal
            counter +=1
            # print(" normals.shape[0]:: ", normals.shape[0])
            # print(" working_normals.shape[0]:: ", working_normals.shape[0])
            
        clusters["n_unique_normals"] = counter

        # print(clusters)
        return clusters
  
@timing_decorator  
def get_cluster2(face_normals):
        
        clusters ={
            "n_unique_normals":None,
            "unique_normals_list":[],
            "unique_normals":{},
            "clusters":{}
        }
        normals = face_normals
      
        
        # print("normals::type::",type(normals))
        normals_list_lenght = normals.shape[0]
        all_indicies = [i for i in  range(0,normals_list_lenght)]
        # print("normals_list_lenght:: ",normals_list_lenght)
        # print("all_indicies:: ",all_indicies)
        visited_indeces = []
        counter = 0
        size_of_normals = 0
        while(normals.shape[0]!=0  ):
            # print("visited_indeces:: ",visited_indeces)
            # print("Clustering::get_cluster::while(working_normals.shape[0]!=0  )")
            current_normal = normals[0]
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
           
            clusters["clusters"][counter] = indexes
            
            
            # print("have_indices_been_visted:: ",have_indices_been_visted)
            # print("is_there_a_repeated_element:: ",is_there_a_repeated_element)
            
            
            normals = np.delete(normals,indexes,axis=0)
            # print("working_normals:: ",working_normals)
            there_is_no_matching_normal = size_of_normals == normals.shape[0]
            if there_is_no_matching_normal:
                index_of_current_normal = np.where(normals==current_normal)[0].tolist()
                indexes += index_of_current_normal
                normals = np.delete(normals,indexes,axis=0)
                continue

            size_of_normals = normals.shape[0]

            clusters["unique_normals_list"].append(current_normal)
            clusters["unique_normals"][counter]=current_normal
            counter +=1
            # print(" normals.shape[0]:: ", normals.shape[0])
            # print(" working_normals.shape[0]:: ", working_normals.shape[0])
            
        clusters["n_unique_normals"] = counter

        # print(clusters)
        return clusters
  

@timing_decorator  
def get_cluster3(face_normals):
    #Not Complete
    clusters ={
        "n_unique_normals":None,
        "unique_normals_list":[],
        "unique_normals":{},
        "clusters":{}
    }
    
    
    normals = face_normals
    vertical_length = normals.shape[0]
    
    ######## choose 4 unique normals #######
    
    sample_normals,samples_indexes = get_sample_normals(normals)
    
    
    # num_processes = multiprocessing.cpu_count()
    num_processes=4
    pool = multiprocessing.Pool(processes=num_processes)
    
    input_data = [(face_normals[:vertical_length*i],sample_normals[i]) for i in range(num_processes)]

    results = pool.starmap(worker_func, input_data)

    pool.close()
    pool.join()
    
    print("results:: ",results)
    visited_indeces = np.concatenate((results[0][1],results[1][1],results[2][1],results[3][1]))
    print("visited_indeces:: ",visited_indeces)
    visited_indeces = np.concatenate((results[0][1],results[1][1],results[2][1],results[3][1],np.array(samples_indexes)))
    print("visited_indeces:: ",visited_indeces)
    # creating list of visited indexs
    # visited_indeces = []
    # for i in range(num_processes):
    #     indexes = results[i][1].tolist()
    #     visited_indeces += indexes
    
      
def worker_func(face_normals,current_normal):
    
    dot_product   = np.dot(normals,current_normal)
    normals_pow   = np.power(normals,2)
    normals_sum   = np.sum(normals_pow,axis=1)
    normals_sqrt  = np.sqrt(normals_sum)
    
    
    current_normal_pow  = np.power(current_normal,2)
    current_normal_sum  = np.sum(current_normal_pow)
    current_normal_sqrt = np.sqrt(current_normal_sum)
    
    
    threshold_degree_in_radian =0.0001745329
    nenomerator = dot_product
    denomenator = normals_sqrt*current_normal_sqrt
    cos_theta   = nenomerator/denomenator
    cos_theta = np.where(cos_theta >1,1,cos_theta) # dealing with nan situation
    cos_theta = np.where(cos_theta <-1,-1,cos_theta) # dealing with nan situation
    theta = np.arccos(cos_theta)
    indexes  = np.where(theta<threshold_degree_in_radian)[0]
    
    
    return (current_normal,indexes)


#utility function 
def get_sample_normals(normals,num_samples=4):
    while(True):
        vertical_length = normals.shape[0]
        quarter_length = int(vertical_length/4) 
        samples_indexes = [i*quarter_length for i in range(4)]
        sample_normals = np.array([normals[idx] for idx in samples_indexes ])
        
        print("get_sample_normals::sample_normals:: ",sample_normals)
        
        #check normals are unique
        for i in range(num_samples):
            normal = sample_normals[i]
            other_normals = sample_normals[ [j for j in range(num_samples)].remove(i)]
            _,indexes = worker_func(other_normals,normal)
            
            if len(indexes)==0:
                return sample_normals,samples_indexes
            
              
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
  
   
if __name__ == '__main__':
    
 
    path ="./mesh.npy"
    normals=np.load(path)
    
    
    result_one=get_cluster(normals)
    result_two=get_cluster2(normals)
    # get_cluster3(normals)
    # get_clusterGPU(normals)
    
    
    print("get_cluster::",result_one)
    print("get_cluster2::",result_two)
    
    
    
  
    
    