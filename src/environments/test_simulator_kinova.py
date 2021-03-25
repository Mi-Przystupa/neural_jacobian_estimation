import pytest
from environments.simulator_kinova import SimulatorKinovaGripper
import numpy as np

def check_float(value, test, thresh):
    return value <= test + thresh and value >= test - thresh 


#To really be robust I should probably do a lot more tests.....
#the idea was mostly some things are super easy to check if they're being handled to run so better to just do that
class TestSimulatorKinovaGripper:

    EPS=1e-6

    @pytest.fixture
    def actions(self):

        return np.array([[0.0, 0.0], [-0.1, 0.2], [1.0, 0.5], [-.5, -.99]])

    @pytest.fixture
    def states(self):
        angles = np.zeros((7,))
        c = np.cos(angles)
        s = np.sin(angles)
        velocity = np.zeros((7,))

        target = np.array([ 1.0, 1.5, 2.0])
        x_psn = np.array([ 2.0, -0.3, -0.1])
        x_transition = np.array([2.3, -0.4,  0.01])

        state = np.concatenate([x_psn, c, s, velocity, target])
        state_p = np.concatenate([x_transition, c, s, velocity, target])
        
        return np.stack([state, state_p],axis=0)

    def test_action_reward(self, actions, states):
        #I think technically I should use 7 actions but 2 should be find for testing..
        gym = SimulatorKinovaGripper(reward_type='action-norm')

        state = states[0]
        state_p = states[1]
        reward = gym._calc_reward(state, actions[0], state_p)
        assert check_float(reward, 0.0, self.EPS)

        reward = gym._calc_reward(state, actions[1], state_p)
        assert check_float(reward, -0.05, self.EPS)

        reward = gym._calc_reward(state_p, actions[2], state)
        assert check_float(reward, -1.25, self.EPS)

        reward = gym._calc_reward(state_p, actions[3], state) 
        assert check_float(reward, -1.2301, self.EPS)

    def test_distance_rewards(self, actions, states):
        gym = SimulatorKinovaGripper(reward_type='l2')

        state = states[0]
        state_p = states[1]

        # the actions should be meaningless
        reward = gym._calc_reward(state, actions[0], state_p)
        assert check_float(reward, -3.0430412419157253, self.EPS)

        reward = gym._calc_reward(state_p, actions[1], state)
        assert check_float(reward, -2.94108823, self.EPS)

        gym = SimulatorKinovaGripper(reward_type='l1')

        reward = gym._calc_reward(state, actions[3], state_p)
        assert check_float(reward, -5.19, self.EPS)

        reward = gym._calc_reward(state_p, actions[2], state)
        assert check_float(reward, -4.9, self.EPS)

    def test_keep_moving_rewards(self, actions, states):
        gym = SimulatorKinovaGripper(reward_type='discrete-time-penalty')

        state = states[0]
        state_p = states[1]

        reward = gym._calc_reward(state, actions[0], state_p)
        assert check_float(reward, -1.0, self.EPS)


        reward = gym._calc_reward(state_p, actions[0], state)
        assert check_float(reward, -1.0, self.EPS)


        state_p = state.copy()

        state_p[-3:] = state[0:3]
        reward = gym._calc_reward(state, actions[2], state_p)
        assert check_float(reward, 0.0, self.EPS)

        # continuous version
        gym = SimulatorKinovaGripper(reward_type='keep-moving')

        state = states[0]
        state_p = states[1]
        reward = gym._calc_reward(state, actions[1], state_p)
        assert check_float(reward, 0.33481338085, self.EPS)

        reward = gym._calc_reward(state_p, actions[1], state)
        assert check_float(reward, 0.33481338085, self.EPS)



        
     






