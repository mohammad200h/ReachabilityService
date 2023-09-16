import pickle
import numpy as np

"""_
1. find closet points to fingertip when the finger distal is almost parallel to object axis
2. select multiple point within a specific threshold to specific point 
3. choose one of them at random
"""


def closest_3d_point(reference_point, points):
    if len(points) == 0:
        return None
    
   
    
    # Calculate the Euclidean distance between the reference point and all points
    distances = np.linalg.norm(points - reference_point, axis=1)
    
    # Find the index of the point with the smallest distance
    closest_index = np.argmin(distances)
    
    # The closest point is the one at the closest_index
    closest_point = points[closest_index]
    
    return closest_point


def closet_3d_points_with_radius(reference_point, points,radius):
    print("closet_3d_points_with_radius::reference_point::type::",type(reference_point))
    print("closet_3d_points_with_radius::points::type::",type(points))
    # Calculate the distances to all points from the closest point
    distances_to_point = np.linalg.norm(points -reference_point, axis=1)
    
    # Find the indices of points within the specified radius
    within_radius_indices = np.where(distances_to_point <= radius)[0]

    # Get the points that are within the specified radius
    points_within_radius = points[within_radius_indices]
    
    
    return points_within_radius



file_path = "high_res_reachabilityMap.pkl"


# Deserialize (load) the dictionary from the pickle file
with open(file_path, "rb") as pickle_file:
    loaded_dict = pickle.load(pickle_file)

# Now, 'loaded_dict' contains the dictionary loaded from the file
# print(loaded_dict)


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



hand_pos = {}

for finger in hand_obs.keys():
    hand_pos[finger] = hand_obs[finger][:3]
    
    
closet_point_to_fingertips = {}
  
# Get closest point on reachability map to fingertip 
for finger in hand_obs.keys():
    closet_point_to_fingertips[finger] = closest_3d_point(np.array(hand_pos[finger]),loaded_dict[finger]["points"])
    print("Closest point to",finger, hand_pos[finger], "is", closet_point_to_fingertips[finger])
    
# Get Points around closet point on reachabiltyMap with radius of r=0.05

r=0.05
for finger in hand_obs.keys():
    points_within_radius = closet_3d_points_with_radius(np.array(closet_point_to_fingertips[finger]),np.array(loaded_dict[finger]["points"]),r)
    print(" point in raius 0.05",finger,closet_point_to_fingertips[finger], "are", points_within_radius)






