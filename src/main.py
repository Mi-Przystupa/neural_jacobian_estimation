from environments import SimulatorKinovaGripper, MultiPointReacher, JacobianWrapper, SimulatorKinovaGripperInverseJacobian
from visualservoing import PickPolicy
import argparse
import numpy as np
import torch
import os
import json
from utils import set_seeds, args_to_str

def PickEnvironment(name, dt, use_jacobian_wrapper):
    # this reward is mostly for RL policies
    rewards = 'l2,precision,action-norm'

    name = name.lower()
    if name == "end-effector":

        if use_jacobian_wrapper:
            print("Using jacobian environment instead")
            gym = SimulatorKinovaGripperInverseJacobian(dt = dt, reward_type=rewards)
        else:
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
    ap.add_argument("--environment", type=str, default="end-effector", help="environment to run: end-effector,multi-point")
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
    ap.add_argument("--beta", type=float, default=1.0, help="weighing term specifically for global_neuralnetwork-multitask model for inverse relation fitting")
    ap.add_argument("--k", type=int, default=10, help="number of neighbors for KNN algorithms")
    ap.add_argument("--l2", type=float, default=0.0, help="amount of weight decay (L2 reg.) for Neural Kinematics model")

    args = vars(ap.parse_args())
    print(args.keys())
    print(args)
    return args


ACTUATORS = 7
def main():
    import time
    import matplotlib.pyplot as plt

    args = get_args()
    extra_to_remove = ["num_hiddens", "activation"] if "custom" not in args["policy_name"] else []
    extra_to_remove = extra_to_remove + (["partial_state"] if "neuralnetwork" not in args["policy_name"] else [])
    extra_to_remove = extra_to_remove + (["beta"] if "multitask" not in args["policy_name"] else [])
    args_string = args_to_str(args, ["load_pth", "policy_name", "render", "environment", "runs", "load_episodes", "targets_pth"] + extra_to_remove)
    exp_folder = './.experiments/{}/{}/{}/'.format(args["environment"], args["policy_name"], args_string)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    with open(exp_folder + 'arguments.json', 'w') as f:
        json.dump(args, f, indent = 4)
    
    gym, num_pts, pt_dim = PickEnvironment(args["environment"], args["dt"], "rl" in args["policy_name"] )

    type_multi_pts = "pose_and_origin"
    if "min-pos-and-origin" in args["environment"]:
        type_multi_pts  = "min_pose_and_origin"
    elif 'multi-point-pose' in args["environment"]:
        type_multi_pts  = 'pose'

    set_seeds(args["seed"], gym)

    policy_name = args["policy_name"]
    policy = PickPolicy(policy_name, args["gain"], num_pts, pt_dim, ACTUATORS, pose_config= type_multi_pts,
                         activation= args["activation"], num_hiddens= args["num_hiddens"],
                          partial_state=args["partial_state"], epochs=args["epochs"], beta= args["beta"],
                          k=args["k"], l2=args["l2"]) 
    runs = args["runs"]

    #reset seed again...in my experience, changing unrelated things affects random state
    set_seeds(args["seed"], gym)
    if args["load_pth"] is not None:
        policy.load(args["load_pth"])

    data_load = None
    if args["load_episodes"] is not None:
        data_load = torch.load(args["load_episodes"])
        print("successfully loaded episodes!!!! {}".format(args["load_episodes"]))


    policy.learn(gym, data_load)
    policy.save(exp_folder + '{}.npy'.format(args["policy_name"]))

    targets = None
    if args["targets_pth"] is not None:
        targets = torch.load(args["targets_pth"])
        print("successfully loaded targets!!!! {}".format(args["targets_pth"]))
        #a bit of hacking...
        gym.target_generation = "fixed"


    #reset randoms seed before evaluation
    results = {}
    if num_pts > 1:
        check_against = "multipoint-inversejacobian"
        
    else:
        check_against = "inversejacobian"
    inverse_jacobian = PickPolicy(check_against, args["gain"], num_pts, pt_dim, ACTUATORS, pose_config = type_multi_pts)

    #set seed again so, hypothetically each evaluation will be comparable for algorithms 1-to-1
    gym.H = args["eval_horizon"]
    set_seeds(args["seed"], gym)
    for e in range(runs):
        if targets is not None:
            obs = gym.reset(target= targets[e])
        else:
            obs = gym.reset()

        policy.reset()
        tuples = [None] * gym.H 
        for h in range(gym.H):
            s = time.time()
            prev_obs = obs
            a = policy.act(obs)
            _ = inverse_jacobian.act(obs)
            obs, reward, done, info = gym.step(a)
            tuples[h] = [obs, a, reward, done, info, prev_obs, policy.J, inverse_jacobian.J]
            if args["render"]:
                gym.render()
            if done:
                break
        results[e] = tuples
    torch.save(results, exp_folder + "results-{}-{}.pth".format(args["seed"], args["policy_name"]))




if __name__ == "__main__":
    main()


