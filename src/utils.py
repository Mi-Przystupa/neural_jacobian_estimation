import numpy as np
import torch
from stable_baselines.common import set_global_seeds


def set_seeds(seed, gym):
    gym.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_global_seeds(seed)

def args_to_str(args, to_pop):
    #args is dictionary of arguments
    s = "args"
    for k in args:
        if k not in to_pop:
            entry = "{}-{}".format(k, args[k])
            s = s + "-" + entry
    return s

