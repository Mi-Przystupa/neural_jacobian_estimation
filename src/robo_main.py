#!/usr/bin/env python3

from environments import TwoJointPlanarKinova, FullDOFKinovaReacher, TwoJointVisualPlanarKinova, CameraUpdateTrackerInfo

import rospy
from visualservoing import PickPolicy
import argparse
import numpy as np
import torch
import os
import json
from utils import set_seeds, args_to_str


def publish_to_camera(capture_target, pub_cam, save_path='./video.avi', repeat = 10):
        if pub_cam is not None:
            message = {
                    "target_image" : None,
                    "capture_flag" : capture_target, 
                    "save_path" : save_path
            }

            message = json.dumps(message)
            repeat = 10
            for _ in range(repeat):
                pub_cam.publish(message)
 

def PickEnvironment(name, dt, H, camera_save_name):
    # this reward is mostly for RL policies
    rewards = 'l2,precision,action-norm'

    name = name.lower()
    if name == "kinova-2d-planar":
        bound = [0.0, 0.0, 0.0, 2.45, 0.0, 1.90, 0.0]
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(
                symmetric_bound = bound)

        print("bound we want to use", custom_bound)
        gym = TwoJointPlanarKinova (dt = dt, reward_type=rewards, 
                                    H=H, reset_joint_angles= np.array([-0.25, 0.25]),
                                    joint_bounds=custom_bound)
        print("Bounds for Joints", gym.joint_bounds)
        print("Bounds for Targets", gym.target_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == "kinova-camera-2d-planar":
        bound = [0.0, 0.0, 0.0, 2.0, 0.0, 1.50, 0.0]
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(
                symmetric_bound = bound)

        print("bound we want to use", custom_bound)
        gym = TwoJointVisualPlanarKinova(dt = dt, reward_type=rewards, 
                                    H=H, reset_joint_angles= np.array([-0.25, 0.25]),
                                    joint_bounds=custom_bound,
                                    video_save_path = camera_save_name)
        print("Bounds for Joints", gym.joint_bounds)
        print("Bounds for Targets", gym.target_bounds)
        num_pts = 1
        pt_dim = 2
    elif name == "7-dof-kinova":
        bound = [np.inf, 1.0, np.inf, 1.5, np.inf, 1.5, np.inf]
        custom_bound = FullDOFKinovaReacher.generate_custom_bounds(symmetric_bound = bound)
        gym = FullDOFKinovaReacher(dt= dt, 
                joint_bounds=custom_bound,
                target_bounds=custom_bound,
                reset_joint_angles=np.array([0., 0.20, 0.0, -0.3, 0.0, 0.1, 0]),
                H = H,
                reward_type=rewards)
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
    ap.add_argument("--policy_name", type=str, default="inversejacobian",
            help="inversejacobian,local_uvs,broyden, global_locallinear,global_neuralnetwork,global_neuralnetwork-multitask,rl_uvs")
    ap.add_argument("--environment", type=str, default="end-effector", help="environment to run: kinova-2d-planar,kinova-camera-2d-planar")
    ap.add_argument("--render", default=False, action="store_true", help="render evaluation results")

    #custom network parameters
    ap.add_argument("--num_hiddens", type=int, default=1, help="number of layers to use each with 100 neurons")
    ap.add_argument("--activation", type=str, default="sigmoid", help="custo activation function used")

    ap.add_argument("--partial_state", type=str, default="position,angles,velocity",#we don't include target because it doesn't play into the optimization
                        help="comma separated list of state to use options: position,angles,velocity,target")
    ap.add_argument("--load_episodes", type=str, default=None, help="path to previously collected episodes for training")
    ap.add_argument("--targets_pth", type=str, default=None, help="path to predefined targets to use")
    ap.add_argument("--eval_horizon", type=int, default=100, help="horizon for length for evaluation")
    ap.add_argument("--epochs", type=int, default=30, help="number of epochs to train neural models")
    ap.add_argument("--k", type=int, default=10, help="number of neighbors for KNN algorithms")


    args = vars(ap.parse_args())
    print(args.keys())
    print(args)
    return args


#ACTUATORS = 2
ACTUATORS= 7
print("NUM ACTUATORS", ACTUATORS)
def main():
    import time
    import matplotlib.pyplot as plt

    args = get_args()
    extra_to_remove = ["num_hiddens", "activation"] if "custom" not in args["policy_name"] else []
    extra_to_remove = extra_to_remove + (["partial_state"] if "neuralnetwork" not in args["policy_name"] else [])
    args_string = args_to_str(args, ["load_pth", "policy_name", "render", "environment", "runs", "load_episodes", "targets_pth"] + extra_to_remove)
    exp_folder = './.experiments/{}/{}/{}/'.format(args["environment"], args["policy_name"], args_string)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    with open(exp_folder + 'arguments.json', 'w') as f:
        json.dump(args, f, indent = 4)
    
    gym, num_pts, pt_dim = PickEnvironment(args["environment"], args["dt"], H=100, camera_save_name = exp_folder + "video.avi") 

    type_multi_pts = "pose_and_origin"

    set_seeds(args["seed"], gym)

    policy_name = args["policy_name"]
    policy = PickPolicy(policy_name, args["gain"], num_pts, pt_dim, num_actuators= ACTUATORS, pose_config= type_multi_pts,
                         activation= args["activation"], num_hiddens= args["num_hiddens"], partial_state=args["partial_state"], epochs=args["epochs"],
                         k=args["k"]) 
    runs = args["runs"]

    #reset seed again...in my experience, changing unrelated things affects random state
    set_seeds(args["seed"], gym)
    if args["load_pth"] is not None:
        policy.load(args["load_pth"])

    data_load = None
    if args["load_episodes"] is not None:
        data_load = torch.load(args["load_episodes"])
        print("successfully loaded episodes!!!! {}".format(args["load_episodes"]))

    print("learning policy")
    policy.learn(gym, data_load)
    print("finished policy learning")
    policy.save(exp_folder + '{}.npy'.format(args["policy_name"]))

    targets = None
    if args["targets_pth"] is not None:
        targets = torch.load(args["targets_pth"])
        print("successfully loaded targets!!!! {}".format(args["targets_pth"]))
        #a bit of hacking...
        gym.target_mode = "fixed"


    #reset randoms seed before evaluation
    results = {}
    if num_pts > 1:
        check_against = "multipoint-inversejacobian"
    else:
        check_against = "2-dof-inversejacobian"
    # hack city 
    if args["environment"] == "7-dof-kinova":
        inverse_jacobian = PickPolicy("inversejacobian", args["gain"], num_pts, pt_dim, num_actuators= ACTUATORS)
    else:
        inverse_jacobian = PickPolicy(check_against, args["gain"], num_pts, pt_dim, num_actuators= ACTUATORS, pose_config = type_multi_pts, lengths=[gym.L1, gym.L2])


    filming = None
    if isinstance(gym, FullDOFKinovaReacher):
        print("going to film 7 DOF")
        filming = CameraUpdateTrackerInfo('target_image')

    #set seed again so, hypothetically each evaluation will be comparable for algorithms 1-to-1
    gym.H = args["eval_horizon"]
    set_seeds(args["seed"], gym)
    for e in range(runs):
        publish_to_camera(capture_target=True, pub_cam = filming, save_path= exp_folder + "video.avi", repeat = 10)
        if targets is not None:
            target = targets[e]
            obs = gym.reset(target= targets[e], use_target_as_is = isinstance(gym, TwoJointVisualPlanarKinova))
        else:
            obs = gym.reset()

        policy.reset()
        tuples = [None] * gym.H 

        s = time.time()
        for h in range(gym.H):
            prev_obs = obs
            a = policy.act(obs)
            obs, reward, done, info = gym.step(a)
            tuples[h] = [obs, a, reward, done, info, prev_obs, policy.J]
            if args["render"]:
                gym.render()
            if done:
                break
        print("Run {}/{}. Duration {}".format(e + 1, runs, time.time() - s))
        print("target {}, position {}".format(gym.get_target(), gym.get_cartesian()))
        print("dist to target {:.3f}, ".format(info['l2']))
        results[e] = tuples
    print("Experiments Finished! Calculating Jacobians")
    gym.reset(target=targets[e], use_target_as_is = isinstance(gym, TwoJointVisualPlanarKinova))

    #Calculate Inverse Jacobian after experiment since....every calculation counts on actual robot
    
    for e in results.keys():
        #for each trajectory
        print("Jacobian calculation for Run {}/{}".format(e + 1, runs))
        tuples = results[e]
        for h in range(len(tuples)):
            #for each step in 
            obs = tuples[h][0]
            if isinstance(gym, TwoJointVisualPlanarKinova):
                tuples[h].append(None) #dummy fill variable
            else:
                publish_to_camera(capture_target= False, pub_cam = filming, save_path= exp_folder + "video.avi", repeat = 10)
                _ = inverse_jacobian.act(obs)
                tuples[h].append(inverse_jacobian.J)
        if rospy.is_shutdown():
            break

        results[e] = tuples
    print("saving experiment results")
    torch.save(results, exp_folder + "results-{}-{}.pth".format(args["seed"], args["policy_name"]))
    print('closing gym')
    publish_to_camera(capture_target= False, pub_cam = filming, save_path= exp_folder + "video.avi", repeat = 10)
    gym.close()




if __name__ == "__main__":
    main()


