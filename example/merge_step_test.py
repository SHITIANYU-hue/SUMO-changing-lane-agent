from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import torch
import gym
import numpy as np
import os
import gym_sumo
import random
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo-v7')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=60000, help='number of simulation steps, (default: 6000)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None
    
env = gym.make(args.env_name)
action_dim = 2
state_dim = 37
state_rms = RunningMeanStd(state_dim)






user_lane_input = 0
user_acceleration_input = 0  # 0 for no change, 1 for acceleration, -1 for deceleration

def on_key_press(key):
    global user_lane_input, user_acceleration_input
    try:
        key_char = key.char
        if key_char in ['0', '1', '2']:
            user_lane_input = int(key_char)
        elif key_char == '3':
            user_acceleration_input = 5  # Acceleration
        elif key_char == '4':
            user_acceleration_input = -5  # Deceleration
    except AttributeError:
        pass

def on_key_release(key):
    global user_lane_input, user_acceleration_input
    if key.char in ['0', '1', '2']:
        user_lane_input = 0
    if key.char in ['3', '4']:
        user_acceleration_input = 0

get_vehicle_number = lambda lane_id: traci.lane.getLastStepVehicleNumber(lane_id)
# calculate flow sum up the vehicle speeds on the lane and divide by the length of the lane to obtain flow in veh/s
get_lane_flow = lambda lane_id: (traci.lane.getLastStepMeanSpeed(lane_id) * traci.lane.getLastStepVehicleNumber(lane_id))/traci.lane.getLength(lane_id)



vel_lst = []
jerk_lst = []
avg_scors=[]
stuck=0
pre_step=20
score = 0.0
manual=args.manual
veh_name_='vehicleg_'
veh_name2 = 'vehicleg2_' 
veh_name3 = 'vehicleg3_' 
veh_name4 = 'vehicleg4_' 

flow_rate=10

if manual:
    from pynput import keyboard
    # Create keyboard listener
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.start()
# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(args.epochs):
    state = env.reset(gui=args.render, numVehicles=10,num_rl=3)
    agent_name=env.rl_names
    action={}


    H=30 ## problem is it will stuck in the middle, need to investigate when will it swtich lane
    H2=30
    distance=5
    distance2=7
    departspeed=10


    for t in range(args.horizon):
        print('step',t)
        lane=0
        acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
        veh_name=veh_name_+str(t)
        veh_name2=veh_name2+str(t)
        veh_name3=veh_name3+str(t)
        veh_name4=veh_name4+str(t)

        ## sim step=0.1, demand=10*1*3600=36000veh/h,e.g, flow_rate=10, inflow=3600*3=10800

        if t%flow_rate==0:

            traci.vehicle.add(veh_name, routeID='route_2', typeID='human', departLane='random',departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name, departspeed)

            traci.vehicle.add(veh_name2, routeID='route_2', typeID='human', departLane='random',departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name2, departspeed)

            traci.vehicle.add(veh_name3, routeID='route_1', typeID='human', departLane='random',departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name3, departspeed)

            traci.vehicle.add(veh_name4, routeID='route_1', typeID='human', departLane='random',departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name4, departspeed)


        vehPerin = get_vehicle_number('9832_0') + get_vehicle_number('9832_1') + get_vehicle_number('9832_2')+get_vehicle_number('9813_0')

        vehPermid = get_vehicle_number('9712_0')+get_vehicle_number('9712_1')+get_vehicle_number('9712_2')+get_vehicle_number('9712_3')
        vehPerout = get_vehicle_number('9728_0')+get_vehicle_number('9728_1')+get_vehicle_number('9728_2')
        density_in = (get_vehicle_number('9832_0') + get_vehicle_number('9832_1') + get_vehicle_number('9832_2'))/ traci.lane.getLength('9832_2')
        density_mid = vehPermid/ traci.lane.getLength('9832_2')
        density_out= vehPerout/ traci.lane.getLength('9728_2')
        lane_flow=get_lane_flow('9712_0')+get_lane_flow('9712_1')+get_lane_flow('9712_2')+get_lane_flow('9712_3')
        print('vehPerin',vehPerin,'vehPermid',vehPermid,'vehPerout', vehPerout,'density_in','laneflowmid',lane_flow,density_in,'density_out',density_out,'density_mid',density_mid)

        for i in range(len(agent_name)):
            action[agent_name[i]]=[0,3]
        next_state_, reward_info, done, info = env.step(action,sumo_lc=True,sumo_carfollow=True,stop_and_go=False,car_follow='Gipps',lane_change='SECRM')
        if done:
            print('rl vehicle run out of network!!')
            break


    print('vehPerin',vehPerin,'vehPermid',vehPermid,'vehPerout',vehPerout,'density_in',density_in,'density_out',density_out,'density_mid',density_mid)
    env.close()
