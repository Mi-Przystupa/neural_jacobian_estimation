import numpy as np
from visualservoing.policy_base import Policy
import time


class JacobianLinearTD(Policy):
    def __init__(self, inputs, rows, cols, lr, lam=1.0): 
        self.w = np.random.randn(inputs, rows *cols)

        self.inputs = inputs
        self.outputs = rows * cols

        self.dims = np.array([rows, cols])
        self.lr = lr
        self.lam = lam
        self.dw = 0

    def predict(self, x):
        return np.matmul(x, self.w)

    def jacobian_predict(self, s):
        J = self.predict(s).reshape(self.dims[0], self.dims[1])
        return J

    def td_update(self, sequence, target):
        s = sequence[0][0]
        dq = sequence[0][1]

        J = self.jacobian_predict(s)
        p_prev = np.matmul(J, dq)

        self.e = np.zeros((self.inputs, self.outputs))
        for o in range(self.outputs):
            j_q = o % self.dims[1] #joint 0 - n
            self.e[:, o] = dq[j_q] * s 

        dw = np.zeros((self.inputs, self.outputs))
        for tup in sequence[1:]:
            s = tup[0]
            dq = tup[1]

            J = self.jacobian_predict(s)
            p_next = np.matmul(J, dq)

            for o in range(self.outputs):
                j_q = o % self.dims[1] #joint 0 - n
                j_x = int(o / self.dims[1]) #feature 0 - m

                dw[:,o] += self.lr * (p_next[j_x] - p_prev[j_x]) * self.e[:,o]
                self.e[:, o] = dq[j_q] * s + self.e[:,o] * self.lam
            
            p_prev = p_next

        for o in range(self.outputs):
            j_q = o % self.dims[1] #joint 0 - n
            j_x = int(o / self.dims[1]) #feature 0 - m

            dw[:,o] += self.lr * (target[j_x] - p_prev[j_x]) * self.e[:,o]
        print(target - p_prev )

        self.dw += dw
        return dw 
    
    def update_weights(self):
        self.w = self.w - self.dw
    
class TemporalDifferenceUVS(Policy):
    def __init__(self, inputs, num_feats, 
                        gain=1.0, num_actuators=7, lr=0.1, lam=1.0,
                        state_extractor = None, num_sequences=50):
        super(TemporalDifferenceUVS, self).__init__(gain=gain, state_extractor=state_extractor)
        self.num_sequences = num_sequences
        self.model = JacobianLinearTD(inputs, num_feats, num_actuators, lr=lr, lam=lam)
        self.sequence = []
        self.q_prev = None


    def learn(self, gym):
        for _ in range(self.num_sequences): 
            done = False
            obs = gym.reset() 
            while not done:
                a = self.act(obs)
                obs, _, done, _ = gym.step(a)
            s = time.time()
            self.update_model(self.sequence, self.target)
            print(time.time() - s)
            self.reset() #reset internal state

    def update_model(self, sequence, target):
        self.model.td_update(sequence, target)
        self.model.update_weights()

    def load(self, pth='base.npy'):
        self.model = np.load(pth, allow_pickle=True)

    def save(self, pth='base.npy'):
        with open(pth, 'wb') as f:
            np.save(f, pth)

    def reset(self):
        self.q_prev = None
        del self.sequence
        self.sequence = []
 
    def act(self, obs):
        ths = self.state_extractor.get_angles(obs)
        q = ths

        psn, trg = self.state_extractor.get_position_and_target(obs)
        x_dot = trg - psn

        J = self.model.jacobian_predict(obs) 
        iJ = np.linalg.pinv(J)

        action = self.gain * np.matmul(iJ, x_dot)
        if self.q_prev is not None:
            self.target = x_dot
            self.sequence.append([obs, q - self.q_prev])

        self.q_prev = q
        return action 

    
