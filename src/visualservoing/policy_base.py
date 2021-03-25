import numpy as np
from visualservoing.state_extractory import StateExtractor

NUM_ACTUATORS=7
MIN_POINTs=1
CARTESIAN = 3
class Policy:
    def __init__(self, gain, state_extractor):
        self.gain = gain
        self.state_extractor = state_extractor if state_extractor is not None else  StateExtractor(MIN_POINTs, CARTESIAN, NUM_ACTUATORS)

    def learn(self, gym):
        assert False, "implement learn"

    def load(self, pth='base.npy'):
        assert False, "implement load"

    def save(self, pth='base.npy'):
        assert False, "implement save"

    def reset(self):
        assert False, "implement reset"
 
    def act(self, obs):
        assert False, "implement act"
