import glob
import matplotlib.pyplot as plt
import torch
import numpy as np


Z_OFFSET = 0.7052
def transform_coords(vec):
    #Converts (x,y,z) ==> (x,y) plane treating new_x = z, new_y = x
    new_x = vec[2] - Z_OFFSET
    new_y = vec[0]
    return [new_x, new_y]


#target_pths = glob.glob(".targets/kinova-2d-planar/*/targets.pth")
target_pths = glob.glob(".targets/kinova-camera-2d-planar/*/targets.pth")



TO_RUN=10#25
for pth in target_pths:
    seed = pth.split('/')[-2]
    targets = torch.load(pth)
    #targets =np.array([transform_coords(vec) for vec in targets]) #needed for 2 DOF 
    targets =np.array([vec for vec in targets])
    print(targets.shape)
    plt.scatter(targets[:TO_RUN,0], targets[:TO_RUN,1], label=seed)

plt.show()
