#!/usr/bin/env python

from Reachability.goalsRequester import GoalsRequester
from Reachability.message import PreGrasp,ReachabilityMap,FingerTips,Point,Mesh

import pickle
import numpy as np

def main():
    goalsRequester = GoalsRequester() 
    
    # hand obs 
    hand_obs={
    

        'ff': [ 0.215     ,  1.0000001 ,  0.43799993, -0.0109999 ,  0.01942151,
                -0.02398635, -0.00105893,  0.        ], 
        'mf': [ 6.5253327e-09,  1.0580001e+00,  3.0599999e-01,  1.0373532e-07,
                1.0457150e-02, -3.5109375e-02, -8.4124664e-03,  1.0000000e+00],
        'rf': [ 2.56000042e-01,  9.17999923e-01,  2.07000017e-01,  1.04893054e-07,
               -2.88026524e-03, -3.83125506e-02, -4.48236102e-03,  2.00000000e+00],
        'th': [ 3.94968873e-05  ,4.63830265e-05,  1.22194600e+00 ,-5.95090670e-01,
                4.37354664e-02,  3.97627893e-02, -1.02510106e-02]


    }
    
    #reachabilityMaps
    file_path = "high_res_reachabilityMap.pkl"
    # Deserialize (load) the dictionary from the pickle file
    with open(file_path, "rb") as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    
    
    hand_pos = {}

    for finger in hand_obs.keys():
        hand_pos[finger] = hand_obs[finger][:3]
    
    fingerTips = FingerTips(
        Point.from_list(hand_pos['ff']),
        Point.from_list(hand_pos['mf']),
        Point.from_list(hand_pos['rf']),
        Point.from_list(hand_pos['th']),
    )
    
    print("fingerTips::ff::x ",fingerTips.ff.x)
    
    reachabilityMap = ReachabilityMap(
        
        Mesh.from_dic(loaded_dict["ff"]),
        Mesh.from_dic(loaded_dict["mf"]),
        Mesh.from_dic(loaded_dict["rf"]),
        Mesh.from_dic(loaded_dict["th"])
    )
    
    print("reachabilityMap::ff::vertices::[0]::x ",reachabilityMap.ff.vertices[0].x)
    
    
    preGrasp = PreGrasp()
    
    preGrasp.fingertips = fingerTips
    preGrasp.reachabilityMaps = reachabilityMap
    
    
    print("preGrasp::reachabilityMaps",preGrasp.reachabilityMaps)
    print("preGrasp::reachabilityMap::ff::vertices::[0]::x ",preGrasp.reachabilityMaps.ff.vertices[0].x)
    
    goalsRequester.send_pregrasp_and_wait_for_result(preGrasp)
    
    goals = goalsRequester.get_update()
    
    print("goals::ff::0::x",goals.ff[0].x)
    

if __name__ == '__main__':
    main()   