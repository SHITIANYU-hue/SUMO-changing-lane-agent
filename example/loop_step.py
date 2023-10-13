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

parser.add_argument("--env_name", type=str, default ='gym_sumo-v0')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
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


vel_lst = []
jerk_lst = []
avg_scors=[]
stuck=0
pre_step=20
score = 0.0
manual=args.manual
if manual:
    from pynput import keyboard
    # Create keyboard listener
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.start()
# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(args.epochs):
    state = env.reset(gui=args.render, numVehicles=10)
    for t in range(args.horizon):
        print('step',t)
        lane=0
        acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration

        # if t>110:
        #     lane=1
        # Check if manual control is enabled and if user input keys are pressed
        # Check if manual control is enabled and apply user input
        if manual:
            lane = user_lane_input
            acceleration = user_acceleration_input
        # Apply the lane change (if any) and simulate the next step
        print('user input',[lane,acceleration])
        next_state_, reward_info, done, collision = env.step([lane, acceleration], sumo_lc=True, sumo_carfollow=True, stop_and_go=False, car_follow='Gipps', lane_change='SECRM')



        print('reward',reward_info)
        print('info',collision)
        print('done',done)
        # if t %5 ==0:

        # print('state',next_state_)
        if t>=3000:
            stuck+=1
            break

        if done:
            print('rl vehicle run out of network!!')
            break

        if collision:
            print('rl vehicle collide!!')
            break
    env.close()

