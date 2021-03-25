from environments import SimulatorKinovaGripper, MultiPointReacher
from visualservoing.state_extractory import StateExtractor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import MovieWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import torch
import glob


#HELPER FUNCTION
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_pos_definite(A):
    eigs = np.linalg.eigvals(A)
    return (eigs > 0).all()


#GET DATA
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#visualize = "multipoint-inversejacobian"
visualize = "blackbox-kinematics-custom"
#visualize="global-locallinear"

#directories = glob.glob('.experiments/multi-point/{}/*/result*.pth'.format(visualize))
#directories = glob.glob('.experiments/multi-point-min-pos-and-origin/{}/*/result*.pth'.format(visualize))
#directories = glob.glob('.experiments/multi-point-pose/{}/*/result*.pth'.format(visualize))
directories = glob.glob('.experiments/multi-point_200000-validation_I_think/{}/*/result*.pth'.format(visualize))




#load each 
results = {}
seed = 0
is_one_model = False
for pth in directories:
    dt = pth.split('/')[-1].split('-dt-')[-1].split('-')[0]
    algorithm = pth.split('/')[2] #+ '-' + dt
    
    result = torch.load(pth)
    if algorithm in results and not is_one_model:
        start = len(results[algorithm])
        end = len(results[algorithm]) + len(result)
        j = 0
        for i in range(start, end):
            results[algorithm][i] = result[j]
            j += 1        
    elif algorithm in results and is_one_model:
        results[algorithm + "-" + str(seed)] = result
        seed += 1
    else:
        results[algorithm] = result



result = results[visualize]

#LOOK AT TRAJECTORIES AND POSES IN ENVIRONMENT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

num_pts = 4
pts_dim= 3
num_actuators = 7
state_extractor = StateExtractor(num_points=num_pts, point_dim=pts_dim, num_angles=num_actuators) 

dt = 0.05
rewards = 'l2,precision,action-norm'

points_config = "pose_and_origin"
#points_config = "pose"
#points_config = "min_pose_and_origin"

gym = MultiPointReacher(dt= dt, reward_type=rewards)
gym.reset()
gym.render()

writer = FFMpegWriter(fps=15)
writer.setup(gym.fig, "{}-performance.mp4", dpi=200)


for k in result.keys():
    trajectory = result[k]
    print("Trajectory: {}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++".format(k))
    num_pos_def = 0
    steps = 0

    

    for v in trajectory:
        steps += 1
        state = v[0]

        J_hat = v[-1]
        iJ_hat = np.linalg.pinv(J_hat)

        J = v[-2]

        JiJ = np.matmul(J_hat, iJ_hat)

        ths = state_extractor.get_angles(state)
        angles = np.rad2deg(ths)
        psn, trg = state_extractor.get_position_and_target(state)
        
        gym.target = trg.reshape(num_pts, pts_dim).T
        gym.kinematics.set_thetas(ths)
        gym.update_psn()

        gym.render()
        writer.grab_frame()

        if not check_pos_definite(JiJ):
            #angles[np.where(angles < 0)] += 360
            #print("the ANGLES in Degrees====================")
            #print(angles)
            #print("the angles in radians+++++++++++++++++++")
            #print(ths)
            #print("The Jacobian+++++++++++++++++")
            #print(J)
            #print("THe approximation inverse+++++++++++++++++")
            #print(iJ_hat)
            #print("detminant of J*iJ_hat ============================")
            #print(np.linalg.det(JiJ))
            #print(v[1])
            #print("action+++++++++++++++++")
            #print(np.matmul(iJ_hat, trg-psn))
            #print(psn.reshape(num_pts, pts_dim).T)

            #print("NOT POSITIVE DEFINITE OH NOOOOOO")
            #psn = psn.reshape(num_pts, pts_dim).T
            #for i in range(0, num_pts):
            #    for j in range(0, i):
            #        print("{} - {}".format(i,j), psn[i,:] - psn[j,:])
            _ = 0
        else:
            num_pos_def += 1
    break
    writer.finish()

        #print("condition numbers+++++++++++++++++++")
        #print("True J", np.linalg.cond(J))
        #print("Approx J", np.linalg.cond(J_hat))

        


    print("num pos def {}/{}".format(num_pos_def, steps))
    #input("next")


