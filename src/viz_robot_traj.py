from environments import SerialTwoDOFGym
from visualservoing.state_extractory import StateExtractor

import numpy as np
import matplotlib.pyplot as plt
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
visualize = "2-dof-inversejacobian"
#visualize="blackbox-kinematics-custom"
#visualize="global-neuralnetwork-multitask-custom"


directories = glob.glob('.experiments/kinova-2d-planar_analysis_take2/{}/*/result*.pth'.format(visualize))



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

num_pts = 1
pts_dim= 2
num_actuators = 2
state_extractor = StateExtractor(num_points=num_pts, point_dim=pts_dim, num_angles=num_actuators) 

dt = 0.05

L1 = 0.3143
L2 = 0.1674 + 0.120
Z_OFFSET = 0.7052

gym = SerialTwoDOFGym(L1= L1, L2= L2)

def transform_coords(vec):
    #Converts (x,y,z) ==> (x,y) plane treating new_x = z, new_y = x
    new_x = vec[2] - self.Z_OFFSET
    new_y = vec[0]
    return np.array([new_x, new_y])

for k in result.keys():
    trajectory = result[k]
    print("Trajectory: {} length {}".format(k, len(trajectory)))
    steps = 0
    
    psn, trg = state_extractor.get_position_and_target(trajectory[0][5])
    if np.linalg.norm(trg - psn, 2) < 1.0 and np.linalg.norm(trg - psn, 2) >= 0.75:
        for v in trajectory:
            steps += 1
            state = v[0]

            ths = state_extractor.get_angles(state)

            psn, trg = state_extractor.get_position_and_target(state)
            
            #trg = transform_coords(trg)
            gym.target = trg
            gym.th1 = ths[0]
            gym.th2 = ths[1]
            gym.update_psn()

            gym.render()



