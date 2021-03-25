from environments import SimulatorKinovaGripper, MultiPointReacher
from visualservoing.state_extractory import StateExtractor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import MovieWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import torch
import glob

#GET DATA
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#visualize = "multipoint-inversejacobian"
#tovisualize = ["blackbox-kinematics-custom", "global-neuralnetwork-custom", "global-neuralnetwork-multitask-custom", "multipoint-inversejacobian"] #multi point
tovisualize = ["blackbox-kinematics", "global-neuralnetwork", "global-neuralnetwork-multitask", "inversejacobian"] #single point



#directories = glob.glob('.experiments/multi-point/{}/*/result*.pth'.format(visualize))
#directories = glob.glob('.experiments/multi-point-pose/{}/*/result*.pth'.format(visualize))
#we could do more, but want to keep it simple
for visualize in tovisualize:
    #directories = glob.glob('.experiments/multi-point_200000-validation_I_think/{}/*/result*.pth'.format(visualize))
    directories = glob.glob('.experiments/end-effector_validate_set/{}/*/result*.pth'.format(visualize))


    #load each 
    results = {}
    seed = 0
    for pth in directories:
        dt = pth.split('/')[-1].split('-dt-')[-1].split('-')[0]
        algorithm = pth.split('/')[2] #+ '-' + dt
        
        result = torch.load(pth)
        if algorithm in results:
            start = len(results[algorithm])
            end = len(results[algorithm]) + len(result)
            j = 0
            for i in range(start, end):
                results[algorithm][i] = result[j]
                j += 1        
        else:
            results[algorithm] = result

    result = results[visualize]

    #LOOK AT TRAJECTORIES AND POSES IN ENVIRONMENT
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #num_pts = 4
    num_pts = 1
    pts_dim= 3
    num_actuators = 7
    state_extractor = StateExtractor(num_points=num_pts, point_dim=pts_dim, num_angles=num_actuators) 

    dt = 0.05
    rewards = 'l2,precision,action-norm'

    points_config = "pose_and_origin"
    #points_config = "pose"
    #points_config = "min_pose_and_origin"

    #gym = MultiPointReacher(dt= dt, reward_type=rewards)
    gym = SimulatorKinovaGripper(dt= dt, reward_type=rewards)
    gym.reset()
    gym.render()

    writer = FFMpegWriter(fps=50)
    writer.setup(gym.fig, "{}-failures-performance.mp4".format(visualize), dpi=200)
    #writer.setup(gym.fig, "{}-performance.mp4".format(visualize), dpi=200)


    TOTAL_TRAJECTORIES = 20
    num_traj = 0
    for k in result.keys():
        trajectory = result[k]
        print("Trajectory: {}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++".format(k))

        psn, trg = state_extractor.get_position_and_target(trajectory[-1][5])
        #if np.linalg.norm(trg - psn, 2) < 0.1: #multipoint
        #if np.linalg.norm(trg - psn, 2) > 0.5: #multipoint

        #if np.linalg.norm(trg - psn, 2) < 0.001: #single point 
        if np.linalg.norm(trg - psn, 2) > 0.2: #single point
            num_traj += 1
            for v in trajectory:
                state = v[5]


                ths = state_extractor.get_angles(state)
                angles = np.rad2deg(ths)
                psn, trg = state_extractor.get_position_and_target(state)
                
                #gym.target = trg.reshape(num_pts, pts_dim).T
                gym.target = trg
                gym.kinematics.set_thetas(ths)
                gym.update_psn()

                gym.render()
                writer.grab_frame()
            if num_traj > TOTAL_TRAJECTORIES:
                print("Maximum numbr of trajectories reached")
                break

            
    writer.finish()
