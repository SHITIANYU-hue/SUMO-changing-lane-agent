from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os
import gym_sumo
import time
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo-v0')
parser.add_argument("--algo", type=str, default = 'sac', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
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

env = gym.make(args.env_name)
action_dim = 2
state_dim = 37
state_rms = RunningMeanStd(state_dim)
exp_tag='discrte'

unix_timestamp = int(time.time())

    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'score/{exp_tag}_{args.algo}_{unix_timestamp}')
else:
    writer = None

if args.algo == 'ppo' :
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac' :
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'ddpg' :
    from utils.noise import OUNoise
    noise = OUNoise(action_dim,0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
    
score_lst = []
state_lst = []
avg_scors=[]

if agent_args.on_policy == True:
    score = 0.0
    score_comfort=0
    score_eff=0
    score_safe=0
    # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        state = env.reset(gui=False, numVehicles=15)
        for t in range(agent_args.traj_length):

            if args.render:    
                env.render()
            state_lst.append(state)
            mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu,sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1,keepdim = True)
            next_state_, reward_info, done, info = env.step(action.cpu().numpy())
            reward, R_comf, R_eff, R_safe = reward_info
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,\
                                         action.cpu().numpy(),\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += reward
            score_comfort +=R_comf
            score_eff += R_eff
            score_safe +=R_safe
            if done:
                # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                    writer.add_scalar("score/comfort", score_comfort, n_epi)
                    writer.add_scalar("score/safe", score_safe, n_epi)
                    writer.add_scalar("score/eff", score_eff, n_epi)

                score = 0
                score_comfort=0
                score_eff=0
                score_safe=0
                env.close()
                break

            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            avg_scors.append(sum(score_lst)/len(score_lst))
            print('avg scores',avg_scors)
            np.save(f'score/avgscores_{exp_tag}_{args.algo}_{unix_timestamp}.npy',avg_scors)
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),f'./model_weights/agent_{exp_tag}_{args.algo}'+str(n_epi))
            
else : # off policy 
    score = 0.0
    score_comfort=0
    score_eff=0
    score_safe=0
    for n_epi in range(args.epochs):
        state = env.reset(gui=False, numVehicles=25)
        done = False
        for t in range(agent_args.traj_length):
            if args.render:    
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            if args.algo=='sac': ##not sure why action generate by sac needs to take out
                action=action[0]
            action = action.cpu().detach().numpy()
            next_state_, reward_info, done, info = env.step(action)
            reward, R_comf, R_eff, R_safe = reward_info
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,\
                                         action,\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 
            score += reward
            score_comfort +=R_comf
            score_eff += R_eff
            score_safe +=R_safe
            if done:
                # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                    writer.add_scalar("score/comfort", score_comfort, n_epi)
                    writer.add_scalar("score/safe", score_safe, n_epi)
                    writer.add_scalar("score/eff", score_eff, n_epi)
                score = 0
                score_comfort=0
                score_eff=0
                score_safe=0
                env.close()
                break              

            else:
                state = next_state

            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train_net(agent_args.batch_size, n_epi)


        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            avg_scors.append(sum(score_lst)/len(score_lst))
            print('avg scores',avg_scors)
            np.save(f'score/avgscores_{exp_tag}_{args.algo}_{unix_timestamp}.npy',avg_scors)
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),f'./model_weights/agent_{exp_tag}_{args.algo}'+str(n_epi))