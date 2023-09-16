#!/usr/bin/env python

import sys


import socket
import threading
import pickle


from Reachability.reachabilityMapRequester import ReachabilityMapRequester
from Reachability.message import WSData,GraspPose 





def produce_msg(cube_obs,wses_obs,graspObjId):
        graspPose = GraspPose()

        graspPose.obj_id = graspObjId
       
        graspPose.obj    = cube_obs
        graspPose.ff     = wses_obs["FF"]
        graspPose.mf     = wses_obs["MF"]
        graspPose.rf     = wses_obs["RF"]
        graspPose.th     = wses_obs["TH"]
 
        return graspPose

def main():
    

    
    reachabilityMapClient = ReachabilityMapRequester()
    
    wses_obs = {'FF': (0.0439932955302149, 0.0449614105091795, 0.8724625624788338, 1.3307146962631295, 1.436828098780302, 2.3224380865433822), 'MF': (0.041562713477913876, 0.05104292786958588, 0.8510861423509894, 1.3307146962631295, 1.436828098780302, 2.3224380865433822), 'RF': (0.03998025995994747, 0.04925495282906968, 0.8288568169874528, 1.3307146962631295, 1.436828098780302, 2.3224380865433822), 'TH': (0.04652346391506017, 0.0006404486846511709, 0.8944408685075758, 1.3307146962631295, 1.436828098780302, 2.3224380865433822), 'Palm': (0.0014214867151891818, -0.019621673959122517, 0.8470495805251732, 0.6325829954654003, 1.436828098780302, 2.322438086543385)}
    cube_obs = (0.05, 0.02, 0.83, -0.0, -0.0, 1.57)


    # loading visual object to scene 
   
  
    graspObjId = "001"

    print("wses_obs:: ",wses_obs)
    print("cube_obs:: ",cube_obs)
    
    # getting reachability map   and loading it to scene   
    graspPose_msg =  produce_msg(cube_obs,wses_obs,graspObjId)
    reachabilityMapClient.send_ws_poses_and_wait_for_result(graspPose_msg)
    
    rmaps,ws_flags = reachabilityMapClient.get_update()
    print("ws_flags::",ws_flags)
    print("rmaps::low_res::ff::face_indexs::len::",len(rmaps["low_res"]["ff"]["face_indexs"]))
    print("rmaps::low_res::ff::normals::len::"    ,len(rmaps["low_res"]["ff"]["normals"]))
    
    print("rmaps::high_res::ff::face_indexs::len::",len(rmaps["high_res"]["ff"]["face_indexs"]))
    print("rmaps::high_res::ff::normals::len::"    ,len(rmaps["high_res"]["ff"]["normals"]))
    
    obj_to_dump = rmaps["high_res"]
    
    file_path = "high_res_reachabilityMap.pkl"
    
    # Serialize (dump) the dictionary to a file using pickle
    with open(file_path, "wb") as pickle_file:
        pickle.dump(obj_to_dump, pickle_file)
if __name__ == '__main__':
    main()   
    
    
    
    
    
    
     
    
    