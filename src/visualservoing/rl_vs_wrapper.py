import numpy as np
from visualservoing.policy_base import Policy
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy



class RL_UVS(Policy):

    def __init__(self, num_actuators, num_pts, pts_dim, state_extractor=None, predict_inverse=False, time_steps=5000000):
        #We let the agent decide what the gain should be
        super(RL_UVS, self).__init__(gain=1.0, state_extractor=state_extractor)
        self.agent = None
        self.predict_inverse = predict_inverse
        self.time_steps = time_steps
        self.num_actuators = num_actuators
        self.num_pts = num_pts
        self.pts_dim = pts_dim
        

    def learn(self, env):
        self.agent = PPO1(MlpPolicy, env, verbose=0)
        self.agent.learn(total_timesteps=self.time_steps)

    def load(self, pth='rl_uvs.npy'):
        pth = pth.split('.npy')[0]
        self.agent = PPO1.load(pth)

    def save(self, pth='base.npy'):
        pth = pth.split('.npy')[0]
        self.agent.save(pth)

    def reset(self):
        #nothing to reset
        i = 0

    def act(self, obs, calc_dq=False):
        #TODO: need to test predicting J...
        prediction = self.agent.predict(obs)[0]
        if self.predict_inverse:
            iJ = np.reshape(prediction, (self.num_actuators, self.num_pts * self.pts_dim))
            self.J = np.linalg.pinv(iJ)
        else:
            self.J = np.reshape(prediction, (self.num_pts * self.pts_dim, self.num_actuators))
            iJ = np.linalg.pinv(self.J)

        #TODO this is...a bit of work around and kinda couples the code unfortunately
        if calc_dq:
            psn, trg = self.state_extractor.get_position_and_target(obs)
            x_dot = trg - psn
            action = np.matmul(iJ, x_dot)
        else:
            action = iJ

        return action





