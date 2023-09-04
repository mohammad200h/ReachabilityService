#!/usr/bin/env python
import pickle
import socket
import threading
import uuid
from .message import Pose,GraspPose,Position
"""
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
"""

import zmq

PORT = 2321
MAX_BYTES = 1024*2

"""
class ServerSharedMemoery(threading.Thread):
    def __init__(self,task,sm_bsize=1000):
        threading.Thread.__init__(self)
        
        # publisher and subscriber
        context = zmq.Context()
        #server will publish to client
        self.sevice = context.socket(zmq.REP)
        self.sevice.bind('tcp://127.0.0.1:2000')
            

        # shared memory  
        self.server_shm_obj = None
        self.client_shm_obj = None
        
        self.create_shared_memory(sm_bsize)
    
        # task 
        self.task_func = task
        
    def create_shared_memory(self,nbytes):
        print("Server::create_shared_memory::called")
        self.server_shm_obj = shared_memory.SharedMemory(create=True, size=nbytes,name="server_shm")
        self.client_shm_obj = shared_memory.SharedMemory(create=True, size=nbytes,name="client_shm")
        print("Server::create_shared_memory::ended")
        
    def run(self):
    
        while True:
            print("server runing..")
            ######## waiting for client #######
            print("Server::waiting for client::called")
            client_identifier =self.sevice.recv_string()
            print("Server::client_identifier:: ",client_identifier)
            print("Server::waiting for client::enede")
            ########reading from client shared memory ######
            print("Server::send::read data from shared memory::begining")
            received_bytes= self.client_shm_obj.buf[:]
            recived_obj = pickle.loads(received_bytes)
            print("Server::send::read data from shared memory::recived_obj:: ",recived_obj)
            print("Server::send::read data from shared memory::ended")
            ######## runing task on recived data ########
            print("Server::send::runing task on recived data::begining")
            result_obj = self.task_func(recived_obj)
            print("Server::send::runing task on recived data::end")
            ######## writing to server shared memory #######
            print("Server::write to shared memory::called")
            bytes = pickle.dumps(result_obj)
            print("Server::bytes::length:: ",len(bytes))
            self.server_shm_obj.buf[:len(bytes)] = bytes
            print("Server::write to shared memory::ended")
            # ########## notify client #########
            self.sevice.send_string("client_identifier")

            
            
    def __del__(self):
        print("destuctor called")
       
  
        
    #utility functions
    def generate_new_identifier(self,uuid):
        identifier = self.server_addr+" "+uuid
        return identifier
        
class ClientSharedMemoery():
    def __init__(self,sm_bsize=1000):
       
        # publisher and subscriber
        context = zmq.Context()
        #client will publish to server
        
        self.client = context.socket(zmq.REQ)
        self.client.connect('tcp://127.0.0.1:2000')
      
        # shared memory  
        self.server_shm_obj = None
        self.client_shm_obj = None
        
        self.create_shared_memory(sm_bsize)
        
    def create_shared_memory(self,nbytes):
        print("Client::create_shared_memory::called")
        self.server_shm_obj = shared_memory.SharedMemory(create=False,name="server_shm")
        self.client_shm_obj = shared_memory.SharedMemory(create=False,name="client_shm")
        print("Client::create_shared_memory::ended")
         
    def send(self,obj):
        
        ########### write to client shared memory #####
        print("Client::write to shared memory::called")
        bytes = pickle.dumps(obj)
        self.client_shm_obj.buf[:len(bytes)] = bytes
        print("Client::write to shared memory::ended")      
        ########## notifiy server #############
        print("Client::send::notifiy server::begining")
        uuid = self.get_uuid()
        self.client.send_string(uuid)
        print("Client::send::notifiy server::ended")
        # ###### waiting for server ######
        server_identifier =self.client.recv_string()
        # ######## reading from server shared memory #######
        print("Client::send::read data from shared memory::begining")
        
        if(uuid !=server_identifier):
            # TODO: raise an error 
            print("ERROR: the id of data does not match the id of request")
        
        received_bytes= self.server_shm_obj.buf[:]
        recived_obj = pickle.loads(received_bytes)
        print("Client::send::read data from shared memory::recived_obj:: ",recived_obj)
        
        
        return recived_obj
        
   
    def close_connection(self):
        print("close_connection::called")
       
        
        
    def __del__(self):
        self.close_connection()
         
    # utility functions 
    def get_uuid(self):
        return str(uuid.uuid4())
"""   

class Server(threading.Thread):
    def __init__(self,task,sm_bsize=1000):
        threading.Thread.__init__(self)
        
        # publisher and subscriber
        context = zmq.Context()
        #server will publish to client
        self.sevice = context.socket(zmq.REP)
        self.sevice.bind('tcp://127.0.0.1:2000')
            
        # task 
        self.task_func = task
        
    def run(self):
    
        while True:
            print("server runing..")
            ######## waiting for client #######
            print("Server::waiting for client::called")
            recived_obj =self.sevice.recv_pyobj()
            print("Server::recived_obj:: ",recived_obj)
            print("Server::waiting for client::enede")
            ######## runing task on recived data ########
            print("Server::send::runing task on recived data::begining")
            result_obj = self.task_func(recived_obj)
            print("Server::send::runing task on recived data::end")
            # ########## notify client #########
            self.sevice.send_pyobj(result_obj)

            
            
    def __del__(self):
        print("destuctor called")
       
  
        
    #utility functions
    def generate_new_identifier(self,uuid):
        identifier = self.server_addr+" "+uuid
        return identifier
        
class Client():
    def __init__(self,sm_bsize=1000):
       
        # publisher and subscriber
        context = zmq.Context()
        #client will publish to server
        
        self.client = context.socket(zmq.REQ)
        self.client.connect('tcp://127.0.0.1:2000')
      
         
    def send(self,obj):
        
           
        ########## send reuest to server #############
        print("Client::send::req server::begining")
        self.client.send_pyobj(obj)
        print("Client::send::req server::ended")
        # ###### waiting for server ######
        recived_obj =self.client.recv_pyobj()
        # ######## reading from server shared memory #######
        print("Client::send::read data from shared memory::recived_obj:: ",recived_obj)
        
        
        return recived_obj
        
   
    def close_connection(self):
        print("close_connection::called")
       
        
        
    def __del__(self):
        self.close_connection()
         
    # utility functions 
    def get_uuid(self):
        return str(uuid.uuid4())
    
    
    
 