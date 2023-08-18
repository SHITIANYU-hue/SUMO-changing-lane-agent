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

parser.add_argument("--env_name", type=str, default ='gym_sumo-v2')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=6000, help='number of simulation steps, (default: 6000)')
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





    
score_lst = []
state_lst = []
avg_scors=[]
pre_step=20
score = 0.0
# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(args.epochs):
                


    state = env.reset(gui=args.render, numVehicles=0)
    veh_name = 'vehicleg_'
    veh_name2 = 'vehicleg2_' 
    veh_name3 = 'vehcleg3_'
    veh_name4 = 'vehcleg4_'

    H=20 ## problem is it will stuck in the middle, need to investigate when will it swtich lane
    distance=60
    distance2=70
    departspeed=10

    for i in range(pre_step):
        if i %2==0:
            veh_name2_=veh_name2+str(i)
            veh_name_=veh_name3+str(i)
            veh_name4_=veh_name4+str(i)

            # traci.vehicle.add(veh_name4_, routeID='route_1', typeID='human', departPos=i*H+distance2, departLane=0,departSpeed=departspeed)
            # traci.vehicle.setSpeedMode(veh_name4_, departspeed)
            # traci.vehicle.setLaneChangeMode(veh_name4_,0b001000000000)

            traci.vehicle.add(veh_name2_, routeID='route_0', typeID='human', departPos=i*H+distance, departLane=0,departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name2_, departspeed)
            traci.vehicle.setLaneChangeMode(veh_name2_,0b001000000000)

            traci.vehicle.add(veh_name_, routeID='route_0', typeID='human', departPos=i*H+distance, departLane=2,departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name_, departspeed)
            traci.vehicle.setLaneChangeMode(veh_name_,0b001000000000)

        else:
            veh_name3_=veh_name3+str(i)
            traci.vehicle.add(veh_name3_, routeID='route_0', typeID='human', departPos=i*H+distance, departLane=1,departSpeed=departspeed)
            traci.vehicle.setSpeedMode(veh_name3_, departspeed)
            traci.vehicle.setLaneChangeMode(veh_name3_,0b001000000000)


    for t in range(args.horizon):
        print('step',t)
        lane=0
        # if t==102:
        #     lane=2
        next_state_, reward_info, done, info = env.step([lane,0],sumo_lc=True,sumo_carfollow=True,stop_and_go=False,car_follow='Gipps',lane_change='SECRM')
        # print('reward',reward_info)
        distance=traci.vehicle.getDistance('rlagent')
        print(distance)
        # if t %5 ==0:



        # print('state',next_state_)
        # if done:
        #     print('rl vehicle run out of network!!')
        #     break