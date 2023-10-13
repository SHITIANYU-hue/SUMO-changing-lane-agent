import gym
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
from agents.controller import IDMController,GippsController
import os

# check lane change and speed mode params: https://github.com/flow-project/flow/blob/master/flow/core/params.py

def angle_between(p1, p2, rl_angle):
	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	angle = degrees(atan2(yDiff, xDiff))
	# Adding the rotation angle of the agent
	angle += rl_angle
	angle = angle % 360
	return angle



def change_lane(follower_name,leader_name,ego_name,threshold_dis=3):
	print('follower_name',follower_name)
	print('leader name',leader_name)
	if len(follower_name)==0 and len(leader_name)==0:
		change_lane=True
		return change_lane
	

	
	change_lane_lead=True
	change_lane_follow=True

	if len(follower_name)!=0:
		distance_rel=follower_name[0][1]
		speed_follow=traci.vehicle.getSpeed(follower_name[0][0])
		speed_ego=traci.vehicle.getSpeed(ego_name)
		print('follower rel',distance_rel)
		print('follower speed',speed_follow,'ego speed',speed_ego)

		if (distance_rel>threshold_dis) and speed_ego>=speed_follow:
			change_lane_follow=True
		else:
			change_lane_follow=False

	if len(leader_name)!=0:
		distance_rel=leader_name[0][1]
		print('leader rel',distance_rel)
		if (abs(distance_rel)>threshold_dis):
			change_lane_lead=True
		else:
			change_lane_lead=False
	
	change_lane = change_lane_lead and change_lane_follow



	return change_lane


def get_distance(a, b):
	return distance.euclidean(a, b)

def map_action(value,clamp=3):
	output_value = (value + clamp) / clamp
	return round(output_value)



class SumoEnv(gym.Env):
	def __init__(self):
		self.name = 'rlagent'
		self.step_length = 0.4
		self.acc_history = deque([0, 0], maxlen=2)
		self.grid_state_dim = 3
		self.state_dim = (4*self.grid_state_dim*self.grid_state_dim)+1 # 5 info for the agent, 4 for everybody else
		self.pos = (0, 0)
		self.curr_lane = ''
		self.curr_sublane = -1
		self.target_speed = 0
		self.speed = 0
		self.lat_speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.numVehicles = 0
		self.vType = 0
		self.lane_ids = []
		self.max_steps = 4000
		self.curr_step = 0
		self.collision = False
		self.done = False


	def start(self, gui=False, numVehicles=10, vType='human', network_conf="networks/highway/sumoconfig.sumo.cfg", network_xml='networks/highway/highway.net.xml'):
		self.gui = gui
		self.numVehicles = numVehicles
		self.vType = vType
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.curr_step = 0
		self.collision = False
		self.done = False
		self.lane_change_model = 0b00100000000 ## disable lane change
		self.speed_mode=32 ## disable all check for speed

		# Starting sumo
		home = os.getenv("HOME")

		if self.gui:
			sumoBinary = home + "/code/sumo/bin/sumo-gui"
		else:
			sumoBinary = home + "/code/sumo/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		traci.start(sumoCmd)

		self.lane_ids = traci.lane.getIDList()

		# Populating the highway
		for i in range(self.numVehicles):
			veh_name = 'vehicle_' + str(i)
			traci.vehicle.add(veh_name, routeID='route_0', typeID=self.vType, departLane='random',departSpeed='10')
			if i ==(self.numVehicles-3):
				# print('add rl')
				traci.vehicle.add(self.name, routeID='route_0', typeID='rl', departLane='random',departSpeed='10')

			# Lane change model comes from bit set 100010101010
			# Go here to find out what does it mean
			# https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#lane_change_mode_0xb6
			#lane_change_model = np.int('100010001010', 2)
			traci.vehicle.setLaneChangeMode(veh_name, self.lane_change_model)
		# traci.vehicle.add(self.name, routeID='route_0', typeID='rl')
		traci.vehicle.setSpeedMode(self.name, self.speed_mode)

		# Do some random step to distribute the vehicles
		for step in range(self.numVehicles*4):
			traci.simulationStep()

		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance
		traci.vehicle.setLaneChangeMode(self.name, 0)
		# Setting up useful parameters
		self.update_params()

	def update_params(self):
		# initialize params
		self.pos = traci.vehicle.getPosition(self.name)
		self.curr_lane = traci.vehicle.getLaneID(self.name)
		# if self.curr_lane == '':
		# 	'''
		# 	if we had collission, the agent is being teleported somewhere else. 
		# 	Therefore I will do simulation step until he get teleported back
		# 	'''
		# 	assert self.collision
		# 	while self.name in traci.simulation.getStartingTeleportIDList() or traci.vehicle.getLaneID(self.name) == '':
		# 		traci.simulationStep()
		# 	self.curr_lane = traci.vehicle.getLaneID(self.name)
		self.curr_sublane = int(self.curr_lane.split("_")[1])

		self.target_speed = traci.vehicle.getAllowedSpeed(self.name)
		self.speed = traci.vehicle.getSpeed(self.name)
		self.lat_speed = traci.vehicle.getLateralSpeed(self.name)
		self.acc = traci.vehicle.getAcceleration(self.name)
		self.acc_history.append(self.acc)
		self.angle = traci.vehicle.getAngle(self.name)


	# Get grid like state
	def get_grid_state(self, threshold_distance=20):
		'''
		Observation is a grid occupancy grid
		'''
		agent_lane = self.curr_lane
		agent_pos = self.pos
		edge = self.curr_lane.split("_")[0]
		agent_lane_index = self.curr_sublane
		lanes = [lane for lane in self.lane_ids if edge in lane]
		state = np.zeros([self.grid_state_dim, self.grid_state_dim])
		# Putting agent
		agent_x, agent_y = 1, agent_lane_index
		state[agent_x, agent_y] = -1
		# Put other vehicles
		for lane in lanes:
			# Get vehicles in the lane
			vehicles = traci.lane.getLastStepVehicleIDs(lane)
			veh_lane = int(lane.split("_")[-1])
			for vehicle in vehicles:
				if vehicle == self.name:
					continue
				# Get angle wrt rlagent
				veh_pos = traci.vehicle.getPosition(vehicle)
				# If too far, continue
				if get_distance(agent_pos, veh_pos) > threshold_distance:
					continue
				rl_angle = traci.vehicle.getAngle(self.name)
				veh_id = vehicle.split("_")[1]
				angle = angle_between(agent_pos, veh_pos, rl_angle)
				# Putting on the right
				if angle > 337.5 or angle < 22.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the right north
				if angle >= 22.5 and angle < 67.5:
					state[agent_x-1,veh_lane] = veh_id
				# Putting on north
				if angle >= 67.5 and angle < 112.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left north
				if angle >= 112.5 and angle < 157.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left
				if angle >= 157.5 and angle < 202.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the left south
				if angle >= 202.5 and angle < 237.5:
					state[agent_x+1, veh_lane] = veh_id
				if angle >= 237.5 and angle < 292.5:
					# Putting on the south
					state[agent_x+1, veh_lane] = veh_id
				# Putting on the right south
				if angle >= 292.5 and angle < 337.5:
					state[agent_x+1, veh_lane] = veh_id
		# Since the 0 lane is the right most one, flip 
		state = np.fliplr(state)
		return state
		
	def compute_jerk(self):
		return (self.acc_history[1] - self.acc_history[0])/self.step_length

	def detect_collision(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if self.name in collisions:
			self.collision = True
			return True
		self.collision = False
		return False
	
	def get_state(self):
		'''
		Define a state as a vector of vehicles information
		'''
		state = np.zeros(self.state_dim)
		before = 0
		grid_state = self.get_grid_state().flatten()
		for num, vehicle in enumerate(grid_state):
			if vehicle == 0:
				continue
			if vehicle == -1:
				vehicle_name = self.name
				before = 1
			else:
				vehicle_name = 'vehicle_'+(str(int(vehicle)))
			veh_info = self.get_vehicle_info(vehicle_name)
			idx_init = num*4
			if before and vehicle != -1:
				idx_init += 1
			idx_fin = idx_init + veh_info.shape[0]
			state[idx_init:idx_fin] = veh_info
		state = np.squeeze(state)
		return state
	
	
	def get_vehicle_info(self, vehicle_name):
		'''
			Method to populate the vector information of a vehicle
		'''
		if vehicle_name == self.name:
			return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
		else:
			lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
			long_speed = traci.vehicle.getSpeed(vehicle_name)
			acc = traci.vehicle.getAcceleration(vehicle_name)
			dist = get_distance(self.pos, (lat_pos, long_pos))
			lane=traci.vehicle.getLaneIndex(vehicle_name)
			return np.array([dist, long_speed, acc, lat_pos])
		
		
	def compute_reward(self, collision, action,reward_type='secrm'):
		'''
			Reward function is made of three elements:
			 - Comfort 
			 - Efficiency
			 - Safety
			 Taken from Ye et al.
		'''
		# Rewards Parameters
		if reward_type=='ye':
			alpha_comf = 0.1
			w_speed = 5
			w_change = 1
			w_eff = 1
			
			# Comfort reward 
			jerk = self.compute_jerk()
			R_comf = -alpha_comf*jerk**2
			action[0]=map_action(action[0])
			#Efficiency reward
			# Speed
			R_speed = -np.abs(self.speed - self.target_speed)
			# Penalty for changing lane
			if action[0]!=0:
				R_change = -1
			else:
				R_change = 0
			# Eff
			# R_eff = w_eff*(w_speed*R_speed + w_change*R_change) ## i didn't add R_lane rihgt not since it is not mandatory lane change
			R_eff = R_speed
			# Safety Reward
			# Just penalize collision for now
			if collision:
				R_safe = -10
			else:
				R_safe = +1
			
			# total reward
			R_tot = R_comf + R_eff + R_safe

		if reward_type=='secrm':
			alpha_comf = 0.1
			w_speed = 1
			w_change = 0
			w_eff = 1
			w_safe=0.1
			
			this_vel,lead_vel,lead_info,headway,target_speed=self.get_ego_veh_info(self.name)
			if this_vel<-5000:
				this_vel=0
			info=[target_speed,lead_vel,headway,this_vel]
			controller=GippsController()
			target_speed= controller.get_speed(info)
			R_speed = -np.abs(this_vel - 20)
			# print('R speed',R_speed)
			if action[0]!=0:
				R_change = -1
			else:
				R_change = 0
			# Eff
			R_eff = w_eff*(w_speed*R_speed + w_change*R_change) ## i didn't add R_lane rihgt not since it is not mandatory lane change

			if collision:
				R_safe = -10
			else:
				R_safe = +1

			R_safe=w_safe*R_safe
			jerk = self.compute_jerk()
			# R_comf = -alpha_comf*jerk**2
			R_comf=jerk
			R_tot = R_comf + R_eff + R_safe
			# print('R total',R_tot)

		return [R_tot, R_comf, R_eff, R_safe]
		

	def apply_acceleration(self, vid, acc, smooth=True):
		"""See parent class."""
		# to handle the case of a single vehicle
		
		this_vel = traci.vehicle.getSpeed(vid)
		next_vel = max([this_vel + acc * 0.1, 0])
		if smooth:
			traci.vehicle.slowDown(vid, next_vel, 1e-3)
		else:
			traci.vehicle.setSpeed(vid, next_vel)

	def get_ego_veh_info(self,name):
		
		lead_info = traci.vehicle.getLeader(name)
		trail_info = traci.vehicle.getFollower(name)
		this_vel=traci.vehicle.getSpeed(name)
		target_speed=traci.vehicle.getAllowedSpeed(name)

		if lead_info is None or lead_info == '' or lead_info[1]>300:  # no car ahead??
			s_star=0
			headway=999999
			lead_vel=99999
		else:
			lead_id=traci.vehicle.getLeader(name)[0]
			headway = traci.vehicle.getLeader(name)[1]
			lead_vel=traci.vehicle.getSpeed(lead_id)

		return this_vel,lead_vel,lead_info,headway,target_speed

	def get_rela_ego_veh_info(self,name,veh_id):
		
		target_speed=traci.vehicle.getAllowedSpeed(name)

		if veh_id ==0:  # no car ahead
			headway=999999
			lead_vel=target_speed
		else:
			try:
				lead_vel=traci.vehicle.getSpeed(veh_id) ##sometimes sumo can't find leader veh if it is too far?
			except:
				lead_vel=target_speed

		return [target_speed,lead_vel]

	def calculate_distance_veh(self,lead_info,ego_info):
		veh_pos = traci.vehicle.getPosition(lead_info)
		ego_pos = traci.vehicle.getPosition(ego_info)
		headway=get_distance(veh_pos,ego_pos)
		return headway
	

	def step(self, action,max_dec=-3,max_acc=3,stop_and_go=False,sumo_lc=False,sumo_carfollow=False,lane_change='SECRM',car_follow='gipps',gipps_params=[1.5,-1,0.1,4]):
		'''
		This will :
		- send action, namely change lane or stay 
		- do a simulation step
		- compute reward
		- update agent params 
		- compute nextstate
		- return nextstate, reward and done
		'''

		this_vel,lead_vel,lead_info,headway,target_speed=self.get_ego_veh_info(self.name)
		headway_e=headway
		lead_vel_e=lead_vel
		# print('this vel',this_vel)
		# print('headway',headway)
		if headway <10 and this_vel>lead_vel:
			action[1]=-3

		if sumo_lc:
			# 2^0: right neighbors (else: left)
			# 2^1: neighbors ahead (else: behind)
			# 2^2: only neighbors blocking a potential lane change (else: all)
			# right: -1, left: 1. sublane-change within current lane: 0.
			if lane_change=='random':
				traci.vehicle.setLaneChangeMode(self.name,self.lane_change_model)
				# neightbor_vehicle=traci.vehicle.getNeighbors(self.name,2^0)
				if this_vel<lead_vel and lead_info is not None:
					change_right=traci.vehicle.couldChangeLane(self.name,-1)
					if change_right:
						traci.vehicle.changeLane(self.name,1, 0.1)

					change_left=traci.vehicle.couldChangeLane(self.name,1)
					if change_left:
						traci.vehicle.changeLane(self.name,2, 0.1)

			if lane_change=='SECRM':
				gipps_acc,gipps_decel,gipps_tau,gipps_s0=gipps_params
				controller=GippsController()
				lead_left= traci.vehicle.getNeighbors(self.name,'010')
				lead_right=traci.vehicle.getNeighbors(self.name,'011')
				follow_left=traci.vehicle.getNeighbors(self.name,'000')
				follow_right=traci.vehicle.getNeighbors(self.name,'001')

				lead_info=traci.vehicle.getLeader(self.name)
				
				this_vel=traci.vehicle.getSpeed(self.name)
				headway=9999
				
				if lead_left is None or len(lead_left) == 0:  # no car ahead
					lead_id=0
					lead_vel=self.target_speed
					headway_left=headway
				else:
					lead_id=lead_left[0][0]
					lead_vel=traci.vehicle.getSpeed(lead_id)
					headway_left=self.calculate_distance_veh(lead_id,self.name)
				info_n=[self.target_speed,lead_vel,headway_left,this_vel]
				# print('info_n',info_n)
				speed_n= controller.get_speed(info_n)

				if lead_right is None or len(lead_right) == 0 :  # no car ahead
					lead_id=0
					lead_vel=self.target_speed
					headway_right=headway

				else:
					lead_id=lead_right[0][0]
					lead_vel=traci.vehicle.getSpeed(lead_id)
					headway_right=self.calculate_distance_veh(lead_id,self.name)
				info_s=[self.target_speed,lead_vel,headway_right,this_vel]
				speed_s= controller.get_speed(info_s)
				# print('info s',info_s)

				if lead_info is None or lead_info == '' or lead_info[1]>30:  # no car ahead
					lead_id=0
					lead_vel=self.target_speed
					headway_e=headway
				else:
					lead_id=lead_info[0]
					lead_vel=traci.vehicle.getSpeed(lead_id)
					headway_e=self.calculate_distance_veh(lead_id,self.name)
				
				info_e=[self.target_speed,lead_vel,headway_e, this_vel]
				# print('info e',info_e)
				speed_e= controller.get_speed(info_e)
				
				# print('headway',headway,'headwaye',headway_e,'headwayright',headway_right,'headwayleft',headway_left)
				# change_right=traci.vehicle.couldChangeLane(self.name,-1)
				# change_left=traci.vehicle.couldChangeLane(self.name,1)
				change_right=change_lane(follow_right,lead_right, self.name)
				change_left=change_lane(follow_left,lead_left, self.name)

				lane=traci.vehicle.getLaneIndex(self.name)
				lane_id=traci.vehicle.getLaneID(self.name).split("_")[0]
				print('lane id',lane_id)

				try:
					lane_num=traci.edge.getLaneNumber(lane_id)
				except:
					lane_num=0
				if lane==(lane_num-1):
					change_left=False
				# if lane==0 or lane_id=='9778' :
				if lane==0  :
					change_right=False

				# change_left=False ### should disable
				# change_right=True

				print('ego secrm speed',speed_e,'north secrm speed',speed_n,'south secrm speed',speed_s)
				print('ego veh speed:', this_vel)
				print('change right',change_right,'change left',change_left)

				if speed_n>speed_e and speed_n >speed_s and change_left:
					action[0]=2
				if speed_s>speed_e and speed_s > speed_n and change_right:
					action[0]=1
				if abs(speed_n-speed_s)<100 and min(speed_n,speed_s)>speed_e:
					if change_right==True and change_left==False:
						action[0]=1
					if change_left ==True and change_right==False:
						action[0]=2
					if change_left ==True and change_right==True:
						action[0]=2
						
				if change_left==True and headway_left>100 and headway_e<20 and this_vel<target_speed:
					action[0]=2
				if change_right==True and headway_right>100 and headway_e<20 and this_vel<target_speed:
					action[0]=1
				
				if change_left==change_right==True and headway_right>100 and headway_e<20 and this_vel>target_speed:
					action[0]=2

				if stop_and_go and (lane_id=='9775' or lane=='9778' or lane=='9783'):
					action[0]=0
				if lane_id==':e0' or lane_id==':e1' or lane_id==':e2' or lane_id==':e3': ## there is issue with sumo, when it is on the edge, it will not detect correctly the surrounding vehicles
					action[0]=0
				print('lane decision',action[0])
				# road=traci.vehicle.getRoadID('exit_veh')
				# print('road',road)

				if action[0] == 0 :
					traci.vehicle.changeLaneRelative(self.name,0,0.1)
				if action[0] == 1 :
					traci.vehicle.changeLaneRelative(self.name,-1,0.1)
				if action[0] == 2 :
					traci.vehicle.changeLaneRelative(self.name,1,0.1)


		# Action legend : 0 stay, 1 change to right, 2 change to left
		else:
			print('lane',action[0])
			lane=traci.vehicle.getRoadID(self.name)

			lane_num=int(traci.edge.getLaneNumber(lane))
			current_edge=int(traci.vehicle.getLaneID(self.name).split("_")[-1])
			if action[0] == 0 :
				traci.vehicle.changeLaneRelative(self.name,0,0.1)
			if action[0] == 1 :
				traci.vehicle.changeLaneRelative(self.name,-1,0.1)
			if action[0] == 2 :
				traci.vehicle.changeLaneRelative(self.name,1,0.1)

		if sumo_carfollow and action[0]==action[1]==0:

			if car_follow=='IDM':
				controller=IDMController()
				info=[this_vel,target_speed,headway,lead_vel,lead_info]
				acceleration= controller.get_accel(info)

			if car_follow=='Gipps':
				controller=GippsController()
				info=[target_speed,lead_vel,this_vel,speed_e]
				acceleration= controller.get_accel(info)
			print('ego hw',headway_e)
			if stop_and_go and lane_id=='9775':
				acceleration=5
			if stop_and_go and lane_id=='9775' and this_vel<=1 :
				acceleration=0
			# if headway_e<11:
			# 	acceleration=-3
			action[1]=acceleration
			print('acceleration',action[1])
			self.apply_acceleration(self.name,action[1])

		else:
			print('acceleration',action[1])
			self.apply_acceleration(self.name,action[1])
			
		edge = self.curr_lane.split("_")[0]
		# if slow_down:
		lanes = [lane for lane in self.lane_ids if edge in lane]

		if stop_and_go:
			for lane in lanes:
				# Get vehicles in the lane
				vehicles = traci.lane.getLastStepVehicleIDs(lane)
				for vehicle in vehicles:
					if vehicle!=self.name:
						info=self.get_vehicle_info(vehicle)
						road=traci.vehicle.getRoadID(vehicle)
						if road=='gneE6' and info[1]>3 :
							self.apply_acceleration(vehicle,-4)
		# 		if info[0]>0 and info[0]<50: ## travel distance in  between 50 to 150?
		# 			self.apply_acceleration(vehicle,-1)
		# Sim step
		traci.simulationStep()
		# action[0]=map_to_minus_zero_plus(action[0])
		# Check collision
		collision = self.detect_collision()
		# Compute Reward 
		reward = self.compute_reward(collision, action)
		# Update agent params 
		if not collision:
			self.update_params()
		# State 
		next_state = self.get_state()
		# Update curr state
		self.curr_step += 1
		
		# Return
		if self.curr_step <= self.max_steps:
			done = collision
		else:
			done = True
			self.curr_step = 0

		return next_state, reward, done, collision
		
	def render(self, mode='human', close=False):
		pass

	def reset(self, gui=False, numVehicles=10, vType='human'):

		self.start(gui, numVehicles, vType)
		return self.get_state()

	def close(self):
		traci.close(False)
