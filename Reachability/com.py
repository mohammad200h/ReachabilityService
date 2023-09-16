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
    
    
    
 