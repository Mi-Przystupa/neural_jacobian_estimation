from environments import SimulatorKinovaGripper, MultiPointReacher

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


points_config = "pose_and_origin"
gym = MultiPointReacher(points_config=points_config)



_ = gym.reset()


gym.render()



input("show next one")
gym = SimulatorKinovaGripper()


_ = gym.reset()

gym.render()

input("show next one")
