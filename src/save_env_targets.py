#!/usr/bin/env python3

from environments import SimulatorKinovaGripper, MultiPointReacher
import numpy as np
try:
    from environments import TwoJointPlanarKinova, TwoJointVisualPlanarKinova, FullDOFKinovaReacher
except Exception as e:
    print("couldn't load robot environment")
    print(e)
from utils import set_seeds, args_to_str
import torch
import os


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#generates targets beginning from specified initial state for 1 pt and multi-pt envs


def PickEnvironment(name, dt):
    # this reward is mostly for RL policies
    rewards = 'l2,precision,action-norm'

    name = name.lower()
    if name == "end-effector":
        gym = SimulatorKinovaGripper(dt = dt, reward_type=rewards)
        num_pts = 1
        pt_dim = 3
    elif name == "multi-point":
        gym = MultiPointReacher(dt= dt, reward_type=rewards, points_config="pose_and_origin")
        num_pts = 4
        pt_dim = 3
    elif name == "kinova-2d-planar":
        bound = [0.0, 0.0, 0.0, 1.90, 0.0, 1.1, 0.0]
        targ_bounds = TwoJointPlanarKinova.generate_custom_bounds(symmetric_bound= bound)
        gym = TwoJointPlanarKinova (dt = dt, reward_type=rewards, target_bounds= targ_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == "kinova-camera-2d-planar":
        bound = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        targ_bounds = TwoJointVisualPlanarKinova.generate_custom_bounds(symmetric_bound= bound)
        gym = TwoJointVisualPlanarKinova(dt = dt, reward_type=rewards, target_bounds= targ_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == "7-dof-kinova":
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(symmetric_bound = [np.inf, 1.0, np.inf, 1.5, np.inf, 1.5, np.inf])
        gym = FullDOFKinovaReacher(dt= dt, 
                joint_bounds=custom_bound,
                target_bounds=custom_bound,
                reset_joint_angles=np.array([0., 0.20, 0.0, -0.3, 0.0, 0.1, 0]),
                reward_type=rewards)

        num_pts = 1
        pt_dim = 3
    else:
        raise Exception("Invalid environment {}".format(name))

    return gym
 

def generate_targets(gym, seed, num_targets):

    set_seeds(seed, gym)

    targets = [None] * num_targets

    for i in range(num_targets):
        _ = gym.reset()
        target = gym.target
        targets[i] = target

    return targets


#environments = ['end-effector', 'multi-point']
#environments=["kinova-2d-planar"]
#environments=["kinova-camera-2d-planar"]
environments=["7-dof-kinova"]

seeds = [12345, 45,  212, 458, 30, 84, 11893, 27948, 8459, 984]
#NUM_TARGETS=25
NUM_TARGETS=10

for env in environments:
    for seed in seeds:
        gym = PickEnvironment(env, dt=0.05)


        targets = generate_targets(gym, seed, NUM_TARGETS)

        exp_folder = './.targets/{}/{}/'.format(env, seed)

        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        for targ in targets:
            gym.target = targ
            gym.render()

        torch.save(targets, exp_folder + 'targets.pth')




