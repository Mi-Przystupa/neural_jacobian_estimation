#!/usr/bin/env python3

from environments import SimulatorKinovaGripper, SimulatorKinovaGripperInverseJacobian, MultiPointReacher


from visualservoing import PickPolicy, TwoDOFInverseJacobian
import time
import numpy as np
import torch
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
    rewards = 'l2,l1,precision,action-norm,discrete-time-penalty,keep-moving'


    #policy = PickPolicy("linear-td", 1.0, 1, 3, 7)
    #policy = PickPolicy("blackbox-kinematics-direct", 1.0, 1, 3, 7)
    #policy = PickPolicy("blackbox-el", 1.0, 1, 3, 7)
    #policy = PickPolicy("inversejacobian", 1.0, 1, 3, 7)
    #policy = PickPolicy("multipoint-inversejacobian", 4.0, 4, 3, 7)
    #policy = PickPolicy("blackbox-kinematics-custom", 1.0, 4, 3, 7, activation= "sigmoid", num_hiddens=2) 
    #policy = PickPolicy("blackbox-kinematics", 1.0, 1, 3, 7, activation= "sigmoid", num_hiddens=2) 
    #policy = PickPolicy("blackbox-kinematics", 1.0, 1, 3, 7) 
    #policy = PickPolicy("global-neuralnetwork", 1.0, 4, 3, 7) 
    policy = PickPolicy("blackbox-kinematics", 1.0, 4, 3, 7) 
    #policy = PickPolicy("global-locallinear-kd", 1.0, 4, 3, 7) 
    #policy = PickPolicy("global-locallinear", 1.0, 1, 3, 7) 
    #policy = PickPolicy("broyden", 1.0, 1, 3, 7) 



    print("we created environment")
    #env = SimulatorKinovaGripper(dt=0.05, reward_type=rewards) #,
    env = MultiPointReacher(dt=0.05, reward_type=rewards)
                                    #)target_generation='kinematics') #, fixed_target=[0.40, -0.05, 0.9])
    #data_load = torch.load('./.data-control/end-effector/30/30-data.pth')
    data_load = torch.load('./.data-control/multi-point/30/30-data.pth')

    #data_load = torch.load('./.data/end-effector/4511/4511-data.pth')
    #data_load = None

    #policy.use_kd_tree = True
    if data_load is not None:
        print(sum([len(v) for k,v in data_load.items()]))
    policy.learn(env, data_load)

    print('hello world of gym parameters')
    print("observation space", env.observation_space)
    print("action space", env.action_space)
    print("env type", type(env))
    i = 0
    obs = env.reset()
    print(obs[0:3])
    returns = 0.0
    #while True:
    for i in range(100 * 20):
        env.render()
        s = time.time()
        a = policy.act(obs)
        print(time.time() - s)
        obs, rew, done, info = env.step(a)

        returns += rew
        if done:
            print(returns)
            returns = 0.0
            obs = env.reset()
            policy.reset()
    env.close()



