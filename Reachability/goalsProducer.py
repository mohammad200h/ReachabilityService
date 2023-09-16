#!/usr/bin/env python

from .com import Server
from threading import Thread
from Reachability.message import ReachabilityGoals

import numpy as np

class GoalGenerator:
    def __init__(self):
        pass
  
    def generate_goals(self,finger_tips_pose,reachabilityMaps,radius):
        goals = {
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None
        }
        
        
        # print("GoalGenerator::finger_tips_pose:: ",finger_tips_pose)
        # print("GoalGenerator::reachabilityMaps:: ",reachabilityMaps)
        # print("GoalGenerator::radius:: ",radius)
        
        #generate goals
        for finger in goals.keys():
            closet_point_to_fingertips = self.closest_3d_point(np.array(finger_tips_pose[finger]),reachabilityMaps[finger]["points"])
            points_within_radius = self.closet_3d_points_with_radius(np.array(closet_point_to_fingertips),np.array(reachabilityMaps[finger]["points"]),radius)
            goals[finger] = points_within_radius
            
        return goals
            
    # utility functions
    def closest_3d_point(self,reference_point, points):
        if len(points) == 0:
            return None

        print("GoalGenerator::closest_3d_point::reference_point::",reference_point)
        # Calculate the Euclidean distance between the reference point and all points
        distances = np.linalg.norm(points - reference_point, axis=1)

        # Find the index of the point with the smallest distance
        closest_index = np.argmin(distances)

        # The closest point is the one at the closest_index
        closest_point = points[closest_index]

        return closest_point

    def closet_3d_points_with_radius(self,reference_point, points,radius):
        if len(points) == 0:
            return None
        
        print("closet_3d_points_with_radius::reference_point::type::",type(reference_point))
        print("closet_3d_points_with_radius::points::type::",type(points))
        # Calculate the distances to all points from the closest point
        distances_to_point = np.linalg.norm(points -reference_point, axis=1)

        # Find the indices of points within the specified radius
        within_radius_indices = np.where(distances_to_point <= radius)[0]

        # Get the points that are within the specified radius
        points_within_radius = points[within_radius_indices]


        return points_within_radius


class GoalsProducer(Server,Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = False       
        super().__init__(task=self.onRequest,sm_bsize=500000)
        
        self.goalGenerator= GoalGenerator()
        
        self.start()
    
    def onRequest(self,pregrasp):
        finger_tips_pose = pregrasp.fingertips.get_dic()
        reachabilityMaps = pregrasp.reachabilityMaps.get_dic()
        radius = pregrasp.radius
        
        data = self.goalGenerator.generate_goals(finger_tips_pose,reachabilityMaps,radius)
        
        reachabilityGoals = ReachabilityGoals.from_dic(data)
        
        return reachabilityGoals
        
        
        
        
    