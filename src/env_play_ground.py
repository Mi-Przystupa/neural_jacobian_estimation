#!/usr/bin/env python3

from environments import SimulatorKinovaGripper, SimulatorKinovaGripperInverseJacobian, MultiPointReacher, FullDOFKinovaReacher, TwoJointPlanarKinova
from environments import TwoJointVisualPlanarKinova, FOURDOFKinovaReacher

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from visualservoing import PickPolicy, TwoDOFInverseJacobian
import time
import numpy as np
try:
    import rospy
except:
    print("no rospy")


#TODO figure out how to just run things in the module folders
#this is a work around for annoying error I was getting when I'd run things in the modules...
if __name__ == "__main__":
    #env = SimulatorKinovaGripper(target_generation = 'kinematic')
    #env = SimulatorKinovaGripper(target_generation = 'fixed')
    #policy = InverseJacobian(gain= 1.0)
    #policy = MultiPointInverseJacobian(gain= 1.0)
    #policy = LocalLeastSquareUVS(gain=1.0, num_pts=4, pts_dim=3, min_experience=10000, k=50, solve_least_square_together=True)
    #policy = AmmortizedLocalLeastSquareUVS(gain=1.0, num_pts=4, pts_dim=3, min_experience=100000,capacity=100000, inputs=45, outputs=84)
    #policy = UncalibratedVisuoServoing(gain= 1.0, n_updates=10, use_broyden=False)
    rewards ='l2' #'l2,l1,precision,action-norm,discrete-time-penalty,keep-moving'
    print("hello world")

    custom_bound = FullDOFKinovaReacher.generate_custom_bounds(symmetric_bound = [np.inf, 1.2, np.inf, 1.2, np.inf, 1.2, np.inf])
    print(custom_bound)
    env = FullDOFKinovaReacher(joint_bounds=custom_bound, target_bounds=custom_bound, H = 200, reward_type=rewards)
    #env = TwoJointPlanarKinova(H=200)
    #env = TwoJointVisualPlanarKinova(H=200, target_bounds = custom_bound)
    #env = FOURDOFKinovaReacher(H = 200, target_bounds = custom_bound, 
    #        joint_bounds = custom_bound, reward_type=rewards)

    #policy = PickPolicy("local-uvs", 0.5, 1, 2, 2)
    #policy = PickPolicy("broyden", 0.1, 1, 2, 2)
    policy = PickPolicy("inversejacobian", 0.5, num_pts= 1, pts_dim= 3, num_actuators=7)
    #policy = TwoDOFInverseJacobian(TwoJointPlanarKinova.L1, TwoJointPlanarKinova.L2, gain = 0.5)
    print("we created environment")
    #env = SimulatorKinovaGripper(dt=0.05, reward_type=rewards)
    #env = SimulatorKinovaGripperInverseJacobian(target_generation='fixed')
    #env = MultiPointReacher(dt=0.1)
    policy.learn(env)
    print('hello world of gym parameters')
    print("observation space", env.observation_space)
    print("action space", env.action_space)
    print("env type", type(env))
    i = 0
    print(env.kinova_listener.control_joints)
    if isinstance(env, TwoJointVisualPlanarKinova):
        env.publish_to_camera(capture_target=True, save_path='./video.avi')
    obs = env.reset()
    returns = 0.0
    #while True:
    for i in range(100 * 20):
        env.render()
        #a = env.action_space.sample()

        a = policy.act(obs)
        #a = np.array([0, 0.0, 0, 0.0, 0.0, 0.1, 0.0])
        if rospy.is_shutdown():
            break
        #act = env.action_space.sample()#np.ones((7, 3)) #np.random.randn(7)
        #J = policy.J
        #iJ = np.linalg.pinv(J)
        obs, rew, done, info = env.step(a)
        #print(obs[0:2], obs[8:10], 'position', 'target')
        print(rew, obs[0:3], obs[-3:], obs.shape)
        #print(obs)
        #print(rew)
        #print(done)
        #print(info)

        returns += rew
        if done:
            print(returns)
            returns = 0.0
            obs = env.reset()
            policy.reset()
    env.close()



