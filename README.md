# Lane Changer Agent with SUMO simulator
Project developed for Sapienza Honor's Programme.
The project aims at developing a reinforcement learning application to make an agent drive safely in acondition of dense traffic. For doing so, SUMO was used to simulate the behaviour of the ego vehicletogether with a fleet of autonomous vehicles and to train two model of RL algorithms, namely DQNand A2C.The development flow was the following: creating the highway with NetEdit, get the parameters of thesimulation, create a custom Gym environment and train the networks


![Alt text](figures/traffic.png?raw=true "Traffic scenario. Red vehicle is the agent")


## Installation
First install SUMO and then the required packages from environment.yml

check sumo version: 

Eclipse SUMO sumo Version v1_16_0+1958-0ab20a374a1
 Build features: Linux-5.19.0-35-generic x86_64 GNU 11.3.0 Release FMI Proj GUI Intl SWIG GDAL GL2PS Eigen
 Copyright (C) 2001-2023 German Aerospace Center (DLR) and others; https://sumo.dlr.de
 License EPL-2.0: Eclipse Public License Version 2 <https://eclipse.org/legal/epl-v20.html>
 Use --help to get the list of options.

## others
main.py to train model

test.py to test model

main_step.py to check with environment setup and visualization, which include different lane change controller or car following controller

modified based on this repo: https://github.com/seolhokim/Mujoco-Pytorch 

## To Do:

Need to debug the action space, decide on and implement the obs space, and implement the reward

multi-agent visualization
