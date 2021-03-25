import numpy as np
import gym
from environments.DHKinematics import DHKinematics
from collections import namedtuple


Bound = namedtuple('Bound', 'low high')
class KinovaEnv(gym.core.Env):

    #Information from Kinova
    #https://www.kinovarobotics.com/sites/default/files/UG-014_KINOVA_Gen3_Ultra_lightweight_robot_User_guide_EN_R06_0.pdf
    GRIPPER_LENGTH = 120
    NUM_JOINTS = 7 #our robot has 7-DOF
    LINK_LENGTHS = np.array([156.4, 128.4, 410.0, 208.4, 105.9, 61.5])
    TOTAL_LENGH = 1070.6 # is just sum of link lengths
    MAX_VELOCITY = 1.22 #in Radians pg 86 set to max speed for small joints

    #0 - 6 ==> actuator 1 - 7 in documents pg 86
    JOINT_BOUNDS = [
                    Bound(-np.inf, np.inf), 
                    Bound(-2.2497, 2.2497),
                    Bound(-np.inf, np.inf),
                    Bound(-2.5796, 2.5796), 
                    Bound(-np.inf, np.inf),
                    Bound(-2.0996, 2.0996),
                    Bound(-np.inf, np.inf),
                ]


    #Conversion
    MILLIMETERS_TO_METERS = 1000


    def __init__(self, ths, action_space=None, observation_space=None, 
            dt=0.05, use_gripper=True, max_velocity=None, H=100):
        self.kinematics = DHKinematics(ths, use_gripper)
        self.link_lengths = self.LINK_LENGTHS / self.MILLIMETERS_TO_METERS #put it in meters
        if use_gripper:
            self.link_lengths[-1] += self.GRIPPER_LENGTH /  self.MILLIMETERS_TO_METERS

        self._max_velocity = self.MAX_VELOCITY if max_velocity is None else max_velocity
        self.dt = dt
        #H for ...horizon 
        self.H = H #TODO: change this name to horizon...

        #must be set in subclasses
        self.action_space = action_space
        self.observation_space = observation_space

    @staticmethod
    def generate_custom_bounds( lows=None, highs=None, symmetric_bound=None):
        low_high = (lows is not None and highs is not None)
        assert  low_high or (symmetric_bound is not None and not low_high), "use either lows/highs or bound, not both"
        if symmetric_bound is not None:
            lows = [-1.0 * v for v in symmetric_bound]
            highs = [v for v in symmetric_bound]

        return [Bound(low, high) for low, high in zip(lows, highs)]

    @staticmethod
    def check_boundary(low, high, current, rate, dt, safety_coefficient=10):
        next = current + safety_coefficient * rate * dt
        in_low = next > low
        in_high = next < high
        return np.logical_and(in_low, in_high)

    def _calc_reward(self):
        assert False, "implement calculate reward"

    def step(self, obs):
        assert False, "Implement step" 

    def reset(self):
        assert False, "Implement reset" 

    def render(self, mode='human'):
        assert False, "Implement render"

   




