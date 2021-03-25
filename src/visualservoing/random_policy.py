from random_policy import OrnsteinUhlenbeckActionNoise #TODO really should be IN the visual servoing module...
from visualservoing.policy_base import Policy

class RandomPolicy(Policy):

    def __init__(self, gain, num_actuators=7, mu = 0, theta = 0.15, sigma = 1.00):
        super(RandomPolicy, self).__init__(gain= gain, state_extractor= None)
        self._rand_policy = OrnsteinUhlenbeckActionNoise(num_actuators, mu=mu, theta=theta, sigma = sigma)

    def learn(self, gym):
        print('random policy, nothing to learn')

    def load(self, pth='base.npy'):
        print("nothing to load")

    def save(self, pth='base.npy'):
        print("nothing to save")

    def reset(self):
        self._rand_policy.reset()
 
    def act(self, obs):
        action = self._rand_policy.sample()
        return self.gain * action






