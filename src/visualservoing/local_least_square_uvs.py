import numpy as np
from visualservoing.policy_base import Policy
from visualservoing.memory import MemoryFactory
from random_policy import OrnsteinUhlenbeckActionNoise
from sklearn.neighbors import KDTree

import time


class DataSelector:

    def __init__(self, eps=None, k=None):
        eps_or_k = (eps is not None or k is not None)
        not_both = (eps is not None and k is not None)
        if eps_or_k and not_both:
            raise ValueError("can only select data by epsilon or k")
        self._eps = eps
        self._k = k 
        self.kd_tree = None
        self.data_size = 0

    def build_KD_tree(self, data):
        self.data_size = len(data)
        self.kd_tree = KDTree(data, leaf_size=5)

    def _do_KNN(self, val, data, targs=None):
        if len(val.shape) <= 1:
            val = val.reshape(1, -1)
        if self.kd_tree is None:
            dist = np.linalg.norm(data - val, ord=2, axis=1) 
            dist[dist == 0.0] = np.inf #if the distance is 0...it's the same point
            indexes = np.argsort(dist)
            indexes = indexes[:self._k]
        else:
            #TODO this...is a hack because it ignores data input
            #TODO: the 3 is arbitrary...but is motivated to give some wiggle room 
            #for case where we (for some reason) do have query point in the dataset
            #this is a hack that could probably be solved...in other ways
            MAGIC_NUMBER = 3

            dist, indexes = self.kd_tree.query(val, k=self._k * MAGIC_NUMBER)
            dist = dist.squeeze()
            indexes = indexes.squeeze()
            indexes = indexes[dist != 0.0]
            if len(indexes) < self._k:
                #TODO: hack for experiments when initial pose is fixed...
                #other fix is to change action behavior if not at target but proposed action is 0
                try:
                    dist, indexes = self.kd_tree.query(val, k= int(MAGIC_NUMBER * self.data_size) // 4)
                except Exception as e:
                    dist, indexes = self.kd_tree.query(val, k=self.data_size // 4 )

                dist = dist.squeeze()
                indexes = indexes.squeeze()
                indexes = indexes[dist != 0.0]
            indexes = indexes[:self._k]

        if targs is not None:
            targs = targs[indexes]

        return data[indexes], targs 


    def _do_norm(self, val, data, targs=None):
        if len(val.shape) <= 1:
            val = val.reshape(1, -1)

        dist = np.linalg.norm(data - val, ord=2, axis=1)
        #return only those values within threshold
        thresh = dist <= self._eps 

        indexes = np.argsort(dist)
        dist = dist[thresh]
        data = data[thresh]
        indexes = np.argsort(dist)
        if targs is not None:
            targs = targs[thresh]
            targs = targs[indexes]        

        return data[indexes], targs

    def get_eps(self):
        return self._eps

    def get_k(self):
        return self._k

    def select_data(self, q, data_q, data_x):
        if self._eps is not None:
            return self._do_norm(q, data_q, data_x)
        if self._k is not None:
            return self._do_KNN(q, data_q, data_x)

DEFAULT_CAPACITY=100000
class LocalLeastSquareUVS(Policy):
    def __init__(self, gain=1.0, num_actuators=7, 
                    min_experience = DEFAULT_CAPACITY,
                    capacity = DEFAULT_CAPACITY,
                    eps=None, k=None, memory="fixed",
                    solve_least_square_together=False,
                    state_extractor = None,
                    use_kd_tree = True):
        super(LocalLeastSquareUVS, self).__init__(gain=gain, state_extractor=state_extractor)

        print("Using k={} neighbors".format(k))
        self._selector = DataSelector(eps, k)
        self._memory = MemoryFactory.factory(memory, capacity=capacity)
        self._min_experience = min_experience

        self._gain = gain
        self._solve_least_square_together = solve_least_square_together

        self._rand_policy = OrnsteinUhlenbeckActionNoise(num_actuators, sigma=1.00)

        self.num_actuators = num_actuators
        self.J = None
        self.X = None
        self.Q = None

        self.use_kd_tree = use_kd_tree

    def calculate_least_squares(self, dQ, dX):
        [_, d] = dQ.shape
        [_, t] = dX.shape

        if not self._solve_least_square_together:
            J = np.zeros((t, d))
            for i in range(t):
                X = dQ
                y = dX[:, i].reshape(-1, 1)
                XtX = np.matmul(X.T, X)
                Xty = np.matmul(X.T, y)

                det = np.linalg.det(XtX)
                if abs(det) < 1e-7 and abs(det) > -1e-7:
                    XtX = XtX + np.identity(self.num_actuators)* 1e-5
                
                w = np.matmul(np.linalg.inv(XtX), Xty)
                for j in range(d):
                    J[i, j] = w[j]
        else:
            X = dQ
            y = dX
            XtX = np.matmul(X.T, X)
            Xty = np.matmul(X.T, y)
            d = np.linalg.det(XtX)
            if abs(d) < 1e-7 and abs(d) > -1e-7:
                #TODO if we change the constant to lambda this becomes L2-ridge regression...
                XtX = XtX + np.identity(self.num_actuators)* 1e-10


            J = np.matmul(np.linalg.inv(XtX), Xty)
            J = J.T
        return J

    def learn(self, gym, external_data=None):
        #external data assumes... a very specific structure:
        #dictionary where keys are episode tally and values are each step of the episode
        if external_data is not None:
            print('external data found, using it all')
            self._min_experience= sum([len(v) for k,v in external_data.items()])
            self._memory._capacity = self._min_experience
            self._memory.flush()
            print("min experience is now {}".format(self._min_experience))

            self.reset()
            for e in external_data.keys():
                for step in external_data[e]:
                    a = self.act(step)

        obs = gym.reset()
        while not self.at_minimal_experience():
            #act pushes experience internally at the moment
            a = self.act(obs)
            obs, reward, done, info = gym.step(a)
            if done:
                obs = gym.reset()
                self.reset() #reset internal state

        self.Q, self.X = self._memory.get_tuples_as_lists()
        self.Q = np.array(self.Q)
        self.X = np.array(self.X)

        print("Memory contains {} number of interactions".format(self._memory.get_index()))

        if self.use_kd_tree:
            print("using KD tree for selecting")
            self._selector.build_KD_tree(self.Q)


    def load(self, pth='local_linear_uvs.npy'):
        features = np.load(pth, allow_pickle=True).item()
        for s, sp in zip(features["Q"], features["X"]):
            self._memory.push(s, sp)

    def save(self, pth='local_linear_uvs.npy'):
        #Q and X because that is what we push in act
        Q, X = self._memory.get_tuples_as_lists()
        features = {"Q": Q, "X": X}
        with open(pth, 'wb') as f:
            np.save(f, features)

    def reset(self):
        self._rand_policy.reset()
    
    def createFiniteDifferenceDataset(self, q, x):
        [n, d] = q.shape
        [n, t] = x.shape

        dQ = np.zeros((n*n, d))
        dX = np.zeros((n*n, t))
        for i in range(n):
            for j in range(n):
                indx = i*n + j
                dQ[indx,:] = q[i,:] - q[j,:]
                dX[indx,:] = x[i,:] - x[j,:]

        return dQ, dX
    def at_minimal_experience(self):
        return self._memory.get_index() >= self._min_experience

    def act(self, obs):

        ths = self.state_extractor.get_angles(obs)
        q = ths

        psn, trg = self.state_extractor.get_position_and_target(obs)
        x_dot = trg - psn
        
        if self.at_minimal_experience():
            #q = joint angles, x = image features
            if self.X is None or self.Q is None:
                Q, X = self._memory.get_tuples_as_lists()
                Q = np.array(Q)
                X = np.array(X)

            else:
                Q, X = self.Q, self.X

            (Q, X) = self._selector.select_data(q, Q, X)

            dQ, dX = self.createFiniteDifferenceDataset(Q, X)
            J = self.calculate_least_squares(dQ, dX)
            self.J = J

            d = 1.0
            if abs(d) < 1e-7 and abs(d) > -1e-7:
                action = self._rand_policy.sample()
            else:
                #only use estimate if it's good
                iJ = np.linalg.pinv(J)
                th_dot = np.matmul(iJ, x_dot)
                action = self._gain * th_dot
                #TODO: apparently if the changes are tiny enough this can still give all 0 actions even far from target

        else:
            #take a random action
            action = self._gain * self._rand_policy.sample()

        if not self.at_minimal_experience():
            self._memory.push(q, psn)
        
        return action
