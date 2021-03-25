from environments import SimulatorKinovaGripper, SimulatorKinovaGripperInverseJacobian
import argparse
import numpy as np
import torch

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import DDPG, PPO1
from gym.wrappers import RescaleAction
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common import set_global_seeds

import os
from utils import set_seeds, args_to_str 


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default="12345", help="random seed for experiment")
    ap.add_argument("--runs", type=int, default=10, help="number of runs for evaluation")
    ap.add_argument("--target_generation", type=str, default="kinematic",
                    help="kinematic,half-sphere,fixed")
    ap.add_argument("--algorithm", type=str, default='ppo1',
                    help="ppo1,ddpg")
    ap.add_argument("--load_path", type=str, default=None,
                    help="load previously trained policy")
    ap.add_argument("--reward_type", type=str, default="l2,precision",
            help="a common separate list of reward functions: ")
    ap.add_argument("--precision_tau", type=float, default=0.01,
            help="temperature term when using precision reward: exp(-dist / tau) if not using precision is ignored")

    args = vars(ap.parse_args())
    print(args.keys())
    print(args)
    return args

def main():
    import time
    import matplotlib.pyplot as plt
    args = get_args()

    #env = SimulatorKinovaGripper(reward_type='l2,precision')
    env = SimulatorKinovaGripperInverseJacobian(reward_type=args["reward_type"],#'l2,precision',
                                        target_generation=args["target_generation"],
                                        precision_tau=args["precision_tau"])

    arg_str = args_to_str(args)
    env = RescaleAction(env, -1, 1)
    os.makedirs('./logs/', exist_ok=True)
    env = Monitor(env, filename="./logs/seed-" + arg_str)
    print(env.action_space.low, env.action_space.high)
    check_env(env)

    if args["algorithm"] == "ddpg":
        n_actions = env.action_space.shape[-1]
        param_noise = None

        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        set_global_seeds(args["seed"])
        set_seeds(args["seed"], env)
        policy = DDPG(LnMlpPolicy, env, verbose=1,
                param_noise=param_noise,
                action_noise=action_noise,
                normalize_observations=True,
                normalize_returns = False,
                gamma=0.99,
                memory_limit=1000000, batch_size=256)
    elif args["algorithm"] == "ppo1":
        policy = PPO1(MlpPolicy, env, verbose=1)

    set_global_seeds(args["seed"])
    set_seeds(args["seed"], env)
    if args["load_path"] is not None:
        print(args["load_path"])
        #load parameters for an already MADE policy
        policy.load_parameters(args["load_path"])
    policy.learn(total_timesteps=5000000)
    #TODO: saving is screwed up...the loaded policy doesn't perform as expected
    policy.save("reacher-" + arg_str)

    #reset randoms seed before evaluation
    env = SimulatorKinovaGripperInverseJacobian(reward_type='l2,precision',
                                        target_generation=args["target_generation"])
    env = RescaleAction(env, -1, 1)

    set_global_seeds(args["seed"])
    set_seeds(args["seed"], env)


    runs = args["runs"]
    results = {}
    for e in range(runs):
        obs = env.reset()
        tuples = []
        for h in range(100):
            s = time.time()
            a = policy.predict(obs)[0]
            obs, reward, done, info = env.step(a)

            tuples.append([obs, reward, a, done, info])
            env.render()
            if done:
                break
        results[e] = tuples

    torch.save(results, "results-{}.pth".format(arg_str ))




if __name__ == "__main__":
    main()


