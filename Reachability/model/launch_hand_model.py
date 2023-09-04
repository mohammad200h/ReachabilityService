#!/usr/bin/env python

import pybullet as p


from modelGenerator import DomainRandomization

finger = "full"
grasp_axis_offset=[-0.02,-0.02,0.06]

dr = DomainRandomization(load_ws=True,load_ws_pcd = False)
dr.save_setting()
dr.generate_model_sdf(grasp_axis_offset=grasp_axis_offset,control_mode=finger)

if __name__ == '__main__':
	p.connect(p.GUI)
	robot = p.loadSDF("./model_"+finger+".sdf")
	# ws    = p.loadSDF("./meshes/ws/FF/model.sdf") 
	while(1):
		
		# shadowHand = robot[0]
		p.setRealTimeSimulation(1)