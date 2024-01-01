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
import json

from utils.utils import *
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo-v7')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=180000, help='number of simulation steps, (default: 6000)')
parser.add_argument('--coop', type=float, default=0, help='cooperative factor for human vehicles')
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



def calc_outflow(outID): 
    state = []
    statef = []
    for detector in outID:
        veh_num = traci.inductionloop.getIntervalVehicleNumber(detector)
        state.append(veh_num)
    return np.sum(np.array(state))


get_vehicle_number = lambda lane_id: traci.lane.getLastStepVehicleNumber(lane_id)
# calculate flow sum up the vehicle speeds on the lane and divide by the length of the lane to obtain flow in veh/s
get_lane_flow = lambda lane_id: (traci.lane.getLastStepMeanSpeed(lane_id) * traci.lane.getLastStepVehicleNumber(lane_id))/traci.lane.getLength(lane_id)
get_meanspeed = lambda lane_id: traci.lane.getLastStepMeanSpeed(lane_id)



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
veh_name5 = 'vehicleg5_' 

outID_9728=['9728_0loop','9728_1loop','9728_2loop']
outID_9832=['9832_0loop','9832_1loop','9832_2loop']
outID_9575=['9575_0loop','9575_1loop','9575_2loop']
outID_9813=['9813_0loop']


# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
flow_rate=1
average_flow=[]
average_density=[]

flow_rates=list(range(1, 2))

# for i in range(len(flow_rates)):
state = env.reset(gui=args.render, numVehicles=10,num_rl=1)
agent_name=env.rl_names
action={}


H=30 ## problem is it will stuck in the middle, need to investigate when will it swtich lane
H2=30
distance=5
distance2=7

lane_flow=0
density_mid=0
t=0
outflow=0
rate = 10000  # veh/h


mainline_demand = 3300 ##5500, 4400,3850,3300
merge_lane_demand = 1250
interval = 2000  # interval for calculating average statistics
simdur = args.horizon  # assuming args.horizon represents the total simulation duration
curflow = 0
curflow_9813 = 0
curflow_9832 = 0
curflow_9575 = 0

curdensity = 0
avg_speeds=[]
cos,hcs,noxs,pmxs=[],[],[],[]
inflows = []
inflows_9813 = []
inflows_9832 = []
inflows_9575 = []
time_step=0.1
total_travel_time=0
densities = []
data=[]
warmup=101
t = 0

t=t+warmup

VSLlist=['9712_0','9712_1','9712_2','9712_3']

while t < simdur:
    print('step', t)
    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    veh_name = veh_name_ + str(t)
    veh_name2 = veh_name2 + str(t)
    veh_name3 = veh_name3 + str(t)
    veh_name4 = veh_name4 + str(t)
    departspeed = random.choice([27, 30])
    # Mainline demand
    if 0 <= t <= 54000:
        inflow_rate_mainline = mainline_demand /36000
    elif 54000 < t <= 90000:
        inflow_rate_mainline = max(0, mainline_demand * (1 - (t - 54000) / 36000))/36000
    else:
        inflow_rate_mainline = 0
    # print('inflow_rate_mainline',inflow_rate_mainline)
    
    # Sample from a uniform distribution for mainline
    u_mainline = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for mainline
    if u_mainline < inflow_rate_mainline:
        traci.vehicle.add(veh_name, routeID='route_2', typeID='human', departLane='random', departSpeed=departspeed)
        # traci.vehicle.setSpeed(veh_name, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name, 3)



    # Merge lane demand
    if 0 <= t <= 18000:
        inflow_rate_merge_lane = merge_lane_demand *(t/18000)/36000
    elif 18000 < t <= 54000:
        inflow_rate_merge_lane = merge_lane_demand/36000
    elif 54000 < t <= 72000:
        inflow_rate_merge_lane = max(0, merge_lane_demand * (1 - (t - 54000) / 18000))/36000
    else:
        inflow_rate_merge_lane = 0

    # print('inflow_rate_mergeline',inflow_rate_merge_lane)

    # Sample from a uniform distribution for merge lane
    u_merge_lane = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for merge lane
    if u_merge_lane < inflow_rate_merge_lane:
        traci.vehicle.add(veh_name4, routeID='route_1', typeID='human', departLane='free', departSpeed=departspeed)
        # traci.vehicle.setSpeed(veh_name4, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name4,3)


    vehPermid = get_vehicle_number('9712_1') + get_vehicle_number('9712_2') + get_vehicle_number('9712_3')
    vehPerout = get_vehicle_number('9728_0') + get_vehicle_number('9728_1') + get_vehicle_number('9728_2')

    veh_number = get_vehicle_number('9712_0')+get_vehicle_number('9712_1') + get_vehicle_number('9712_2') + get_vehicle_number('9712_3')
    # veh_number_total=traci.vehicle.getIDCount()+int(len(traci.simulation.getPendingVehicles())) ##i find that it will make it run very slow
    # veh_number_total=traci.vehicle.getIDCount()+len(traci.edge.getPendingVehicles('9813')) ## this is to count waiting vehicle in the merge
    veh_number_total=traci.vehicle.getIDCount()
    total_travel_time = total_travel_time+ (time_step*veh_number_total)/3600

    set_vsl(v=[2,3,4,5],VSLlist=VSLlist)

    curdensity += vehPermid / traci.lane.getLength('9712_1')
    # print('curflow',curflow,'cudensity',curdensity)

    if t % interval == 0:
        # append average flow and density for the last interval
        curflow = curflow + calc_outflow(outID_9728)
        curflow_9813 = curflow_9813 + calc_outflow(outID_9813)
        curflow_9832 = curflow_9832 + calc_outflow(outID_9832)
        curflow_9575 = curflow_9575 + calc_outflow(outID_9575)
        avg_speed=(get_meanspeed('9712_1')+get_meanspeed('9712_2')+get_meanspeed('9712_3')+get_meanspeed('9712_0'))/4 ### this is only bottlneck's speed

        co,hc,nox,pmx,all_avg_speed=calc_emission_speed() ## i consider to calculate all edges emission and avg speed (for whole network)
        cos.append(co),hcs.append(hc),noxs.append(nox),pmxs.append(pmx)
        inflows.append(curflow )
        inflows_9813.append(curflow_9813 )
        inflows_9832.append(curflow_9832 )
        inflows_9575.append(curflow_9575 )
        avg_speeds.append(all_avg_speed)
        # print('flow stas','9728',calc_outflow(outID_9728),'9832',calc_outflow(outID_9832),'9575',calc_outflow(outID_9575))

        densities.append(curdensity / interval)
        # print('average laneflow:', curflow / interval, 'average density', curdensity / interval,'average speed',np.mean(avg_speeds))

        # reset averages
        curflow = 0
        curflow_9813=0
        curflow_9832=0
        curflow_9575=0
        curdensity = 0

    t = t + 1

    for i in range(len(agent_name)):
        action[agent_name[i]] = [2, 3]
    
    try:
        next_state_, reward_info, done, info = env.step(
            action, sumo_lc=False, sumo_carfollow=True, stop_and_go=False, car_follow='Gipps', lane_change='SECRM')
    except:
        print('fail?')
        pass
    
    if done:
        # print('rl vehicle run out of network!!')
        pass

    
    # # # # Save the average values
    # np.save(f'results/9728/Main3300_merge1250average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows)
    # np.save(f'results/9813/Main3300_merge1250average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9813)
    # np.save(f'results/9832/Main3300_merge1250average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9832)
    # np.save(f'results/9575/Main3300_merge1250average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9575)
    # np.save(f'results/Main3300_merge1250average_density2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', densities)
print('total travel time (h):',total_travel_time)
print('average bottleneck speed:',np.mean(avg_speeds))
print('average emission: ','CO:',np.mean(cos),'HC:',np.mean(hcs),'NOX:',np.mean(noxs),'PMX:',np.mean(pmxs))
env.close()
