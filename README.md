# Training RL Agent with SUMO simulator

## Objective

This Project is inspired  from this project (https://github.com/federicovergallo/SUMO-changing-lane-agent) 

The project aims at developing a reinforcement learning application to make an agent drive safely in condition of differnet densitiy of traffic. For doing so, SUMO was used to simulate the behaviour of the ego vehicletogether with a fleet of autonomous vehicles and to train model of RL algorithms.The development flow was the following: creating the highway with NetEdit, get the parameters of thesimulation, create a custom Gym environment and train the networks

Main contributions of this environment:

1. It is easy to configurate. Compared with [FLOW](https://github.com/flow-project/flow), it is easy to directly control SUMO vehicles

2. It is easy to extend to different real road networks, we could use open street map to download the network in [net.xml format](https://wiki.openstreetmap.org/wiki/OSM_XML) and add the [rou.xml file](https://sumo.dlr.de/docs/Tools/Routes.html) to generate the traffic flow.

3. The Autonomous vehicle can be controlled by a. rule-based controller; b. RL controller; c. user's keyboard. 

4. It is possible to extend to control by large language model


## Single agent Example scenario

1. Loop network: this network can support long running steps to gather enough experience for agent to learn. We can also support changing differnet traffic density to evaluate the driving efficiency and safety. In this network, the curvature's effect on speed is disabled.

1.1 Normal Loop network:

![Alt text](figures/loop.gif?raw=true "Loop network Traffic scenario. Red vehicle is the AV agent")


1.2 Loop network with stop and go:

![Alt text](figures/loopstopandgo.gif?raw=true "Loop network Traffic scenario with stop and go. Red vehicle is the AV agent")


2. Winston Churchill eastbound on-ramp QEW road geometry: To further evaluate the effectiveness of the proposed method, we create another network based on real road geometry, i.e., Queen Elizabeth Way (QEW) Winston Churchill eastbound on-16 ramp near Toronto, Canada. This network can apply emergent brake, stay in main lane and choose to exit.

2.1 stay in main lane in qew
![Alt text](figures/stay-main-lane-qew.gif?raw=true "qew network Traffic scenario with goal as stay in main lane. Red vehicle is the AV agent")

2.2 choose to exit in qew
![Alt text](figures/exit-qew.gif?raw=true "qew network Traffic scenario with goal as choose to exit the network. Red vehicle is the AV agent")

2.3 emergent brake in qew
![Alt text](figures/emergent-brake-qew.gif?raw=true "qew network Traffic scenario with emergent brake. Red vehicle is the AV agent")

2.4 merge in qew
![Alt text](figures/mergeqew.gif?raw=true "qew network Traffic scenario with merge case. Red vehicle is the AV agent")

2.5 intersection in qew
![Alt text](figures/intersectionqew.gif?raw=true "qew network Traffic scenario with intersection case. Red vehicle is the AV agent")


## Multi agent Example scenario

3.1  Loop network with multi-agent:

![Alt text](figures/multi-agent-loop.gif?raw=true "Loop network with multi-agent Traffic scenario. Red vehicle is the AV agent")


3.2  Merge network with multi-agent:

![Alt text](figures/multi-agent-merge.gif?raw=true "merge network with multi-agent Traffic scenario. Red vehicle is the AV agent")



## Installation
1. First install SUMO, please follow this  https://sumo.dlr.de/docs/Installing/Linux_Build.html

2. Check sumo version: (you could also use other version)

Eclipse SUMO sumo Version v1_16_0+1958-0ab20a374a1
 Build features: Linux-5.19.0-35-generic x86_64 GNU 11.3.0 Release FMI Proj GUI Intl SWIG GDAL GL2PS Eigen
 Copyright (C) 2001-2023 German Aerospace Center (DLR) and others; https://sumo.dlr.de
 License EPL-2.0: Eclipse Public License Version 2 <https://eclipse.org/legal/epl-v20.html>
 Use --help to get the list of options.

3. Remember to change sumo path to your own in the gym_sumo/gym_sumo/envs/

4. And then the required packages from environment.yml, i.e., conda env create -f environment.yml

5. Remember to install gym-sumo, go to gym_sumo and run pip install -e .

## others
main.py to train model

test.py to test model

main_step.py to check with environment setup and visualization, which include different lane change controller or car following controller

agents/ it integrate differnet possible control algorithm, such as RL models or formula-based driving models

RL algorithms modified based on this repo: https://github.com/seolhokim/Mujoco-Pytorch 

## To Do:

Need to debug the action space, decide on and implement the obs space, and implement the reward

Multi-agent visualization (on going)

Add LLM control
