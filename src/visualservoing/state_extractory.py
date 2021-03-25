import numpy as np
import torch 
class StateExtractor:
    """
    This whole class assumes a very specific structure of the state :
    points come first, angles next, and target is last
    """
    PARTIAL_STATE_OPTIONS = ["position", "angles", "velocity", "target", "raw_angles"]
    def __init__(self, num_points, point_dim, num_angles, im_w=640, im_h = 480, im_d=1.0, partial_state="position,angles,velocity"):
        self.num_points = num_points
        self.point_dim = point_dim
        self.num_angles = num_angles

        self.img_w = im_w
        self.img_h = im_h
        self.img_d = im_d

        self.partial_state = partial_state.split(',')
        for p in self.partial_state:
            assert p in self.PARTIAL_STATE_OPTIONS, "{} not valid partial state information"

    def get_num_features(self):
        return self.num_points * self.point_dim

    def to_batch(self, obs):
        if len(obs.shape) <=1:
            obs = obs.reshape(1,-1)
        return obs

    def get_batch_position(self, states):
        return states[:, 0:self.num_points*self.point_dim]

    def get_position(self, state):
        state = self.to_batch(state)
        return self.get_batch_position(state)[0]

    def get_batch_target(self, states):
        return states[:, -self.point_dim * self.num_points:]

    def get_target(self, state):
        state = self.to_batch(state)
        return self.get_batch_target(state)[0]

    def get_batch_position_and_target(self, states):
        return self.get_batch_position(states), self.get_batch_target(states)

    def get_position_and_target(self, state):
        state = self.to_batch(state)
        psn, trg = self.get_batch_position_and_target(state)
        return psn[0], trg[0]

    def get_batch_sinusoidal(self, states):
        n = self.point_dim * self.num_points
        c_bound = n + self.num_angles
        s_bound = n + 2 * self.num_angles
        c_th = states[:, n:c_bound ]
        s_th = states[:, c_bound:s_bound]
        return c_th, s_th

    def get_batch_velocity(self, states):
        n = self.point_dim * self.num_points
        start = n + 2 * self.num_angles
        end = n + 3 * self.num_angles
        return states[:, start:end]

    def get_batch_angles(self, states):
        c_th, s_th = self.get_batch_sinusoidal(states)
        th = np.arctan2(s_th, c_th)
        return th

    def get_angles(self, state):
        states = self.to_batch(state)
        return self.get_batch_angles(states)[0]

    def get_batch_partial_state(self, states): 
        partial_state = []
        if "position" in self.partial_state:
            position = self.get_batch_position(states)
            partial_state.append(position)
        
        if "angles" in self.partial_state:
            c_th, s_th = self.get_batch_sinusoidal(states)
            partial_state.append(c_th)
            partial_state.append(s_th)
        elif "raw_angles" in self.partial_state:
            th = self.get_batch_angles(states)
            partial_state.append(th)

        if "velocity" in self.partial_state:
            vel = self.get_batch_velocity(states)
            partial_state.append(vel)

        if "target" in self.partial_state:
            targ = self.get_batch_target(states)
            partial_state.append(targ)
        if isinstance(states, torch.Tensor):
            partial_state = torch.cat(partial_state, dim=1)
        else:
            partial_state = np.concatenate(partial_state, dim=1)
        return partial_state

    def get_partial_state_dimensions(self):
        state_dim = 0
        if "position" in self.partial_state:
            state_dim += self.num_points * self.point_dim        
            
        if "angles" in self.partial_state:
            state_dim += self.num_angles * 2  #using sinusoidal representation
        elif "raw_angles" in self.partial_state:
            state_dim += self.num_angles #just angles

        if "velocity" in self.partial_state:
            state_dim += self.num_angles

        if "target" in self.partial_state:
            state_dim += self.num_points * self.point_dim

        return int(state_dim)


    def resize_dimensions(self, point):
        return np.array([point[0] * self.img_w, point[1] * self.img_h, point[2] * self.img_d])


        
