from environments import SimulatorKinovaGripper, MultiPointReacher, JacobianWrapper, SimulatorKinovaGripperInverseJacobian
try:
    from environments import TwoJointPlanarKinova, FullDOFKinovaReacher, TwoJointVisualPlanarKinova
except Exception as e:
    print("couldn't load robot environment")
    print(e)

from visualservoing import PickPolicy
import argparse
import numpy as np
import torch
import os
import json
from utils import set_seeds, args_to_str
import time

def PickEnvironment(name, dt, use_jacobian_wrapper):
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
    elif name =='multi-point-min-pos-and-origin':
        gym = MultiPointReacher(dt= dt, reward_type=rewards, points_config="min_pose_and_origin")
        num_pts = 3
        pt_dim = 3
    elif name == 'multi-point-pose':
        gym = MultiPointReacher(dt= dt, reward_type=rewards, points_config="pose")
        num_pts = 3
        pt_dim = 3
    elif name == "kinova-2d-planar":
        bound = [0.0, 0.0, 0.0, 2.40, 0.0, 1.90, 0.0]
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(
                symmetric_bound = bound)

        print("bound we want to use", custom_bound)
        gym = TwoJointPlanarKinova(dt = dt, reward_type=rewards, 
                                    H=100, reset_joint_angles= np.array([-0.25, 0.25]),
                                    joint_bounds=custom_bound)
        print("Bounds for Joints", gym.joint_bounds)
        print("Bounds for Targets", gym.target_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == "kinova-camera-2d-planar":
        bound = [0.0, 0.0, 0.0, 2.00, 0.0, 1.50, 0.0]
        custom_bound = TwoJointVisualPlanarKinova.generate_custom_bounds(
                symmetric_bound = bound)

        print("bound we want to use", custom_bound)
        gym = TwoJointVisualPlanarKinova(dt = dt, reward_type=rewards, 
                                    H=100, reset_joint_angles= np.array([-0.25, 0.25]),
                                    joint_bounds=custom_bound)
        print("Bounds for Joints", gym.joint_bounds)
        print("Bounds for Targets", gym.target_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == '7-dof-kinova':
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(symmetric_bound = [np.inf, 1.0, np.inf, 1.5, np.inf, 1.5, np.inf])
        print("bound we want to use", custom_bound)

        gym = FullDOFKinovaReacher(dt= dt, 
                joint_bounds=custom_bound,
                target_bounds=custom_bound,
                H = 100,
                reset_joint_angles=np.array([0., 0.20, 0.0, -0.3, 0.0, 0.1, 0]),
                reward_type=rewards)

        print("Bounds for Joints", gym.joint_bounds)
        print("Bounds for Targets", gym.target_bounds)

        num_pts = 1
        pt_dim = 3

    else:
        raise Exception("Invalid environment {}".format(name))

    return gym, num_pts, pt_dim
        

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_pth", type=str, default=None,
            help="path to load model from")
    ap.add_argument("--seed", type=int, default="12345", help="random seed for experiment")
    ap.add_argument("--gain", type=float, default=1.0, help="policy gain")
    ap.add_argument("--dt", type=float, default=0.05, help="simulation delta t")
    ap.add_argument("--runs", type=int, default=10, help="number of runs for evaluation")
    ap.add_argument("--environment", type=str, default="end-effector", help="environment to run: end-effector,multi-point")
    ap.add_argument("--render", default=False, action="store_true", help="render evaluation results")
    ap.add_argument("--rand_init", default=False, action="store_true", help="change initial position for data collection")
    ap.add_argument("--save_dir", default="data", help="folder to create, it will be prepended with a . so actually path is like ./.<argument>/")

    args = vars(ap.parse_args())
    print(args.keys())
    print(args)
    return args


ACTUATORS = 7
def main():
    import time
    import matplotlib.pyplot as plt

    args = get_args()
    exp_folder = './.{}/{}/{}/'.format(args["save_dir"], args["environment"], args["seed"])

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    with open(exp_folder + 'arguments.json', 'w') as f:
        json.dump(args, f, indent = 4)
    
    gym, num_pts, pt_dim = PickEnvironment(args["environment"], args["dt"], False)

    set_seeds(args["seed"], gym)
    if args["environment"] == "7-dof-kinova":
        print('using inverse jacobian, gotta protect the robot')
        policy = PickPolicy("inversejacobian", args["gain"], num_pts, pt_dim, ACTUATORS) 
    else:
        policy = PickPolicy("random", args["gain"], num_pts, pt_dim, ACTUATORS) 
    runs = args["runs"]

    #reset seed again...in my experience, changing unrelated things affects random state
    set_seeds(args["seed"], gym)

    #reset randoms seed before evaluation
    results = {}

    #set seed again so, hypothetically each evaluation will be comparable for algorithms 1-to-1
    set_seeds(args["seed"], gym)
    start = time.time()
    for e in range(runs):
        #so change the starting position every time
        if args["rand_init"]:
            thetas = np.random.uniform(-2.0, 2.0, 7)
            print('random initial pose', thetas)
            obs = gym.reset(thetas=thetas)
        else:
            obs = gym.reset()
        policy.reset()
        tuples = [None] * gym.H

        tuples[0] = obs
        noise_mask = None
        for h in range(1, gym.H):
            a = policy.act(obs)
            if args["environment"] == "7-dof-kinova":
                update_noise = bool(np.random.binomial(1, 0.05))
                if update_noise or noise_mask is None:
                    noise_mask = np.random.normal(0.0, 0.1, (ACTUATORS))
                a += noise_mask 

            obs, reward, done, info = gym.step(a)
            tuples[h] = obs

            if args["render"]:
                gym.render()
            if done:
                break
        results[e] = tuples
    gym.reset() #mostly for robot environments
    duration = time.time() - start
    print("Duration of data Collection: {}".format(duration))
    print("Saving data")
    torch.save(results, exp_folder + "{}-data.pth".format(args["seed"]))
    print("Data saved")
    torch.save([duration], exp_folder + "duration-{}.pth".format(duration))
    try:
        gym.close()
    except Exception as e:
        print("Close not implemented, or weirdness")
        print(e)




if __name__ == "__main__":
    main()


