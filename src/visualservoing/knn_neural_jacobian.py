import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from visualservoing.memory import MemoryFactory 
from visualservoing.policy_base import Policy
from visualservoing.utils import RBFNetwork, ExtremeLearning
from visualservoing.knn_target_dataset import KnnTargetDataset

from random_policy import OrnsteinUhlenbeckActionNoise

DEFAULT_CAPACITY=100000
class NeuralJacobianKNN(Policy):

    def __init__(self, gain, min_experience=DEFAULT_CAPACITY,
                    capacity=DEFAULT_CAPACITY,
                    num_actuators=7, 
                    memory="fixed",
                    inputs=27, #TODO this value is ignored, should probably remove it entirely....
                    outputs = 21, epochs=30,
                    state_extractor=None, fit_inverse_relation=False, fit_null_space=False,
                    use_linear=False, k=5,
                    use_rbf=False,
                    custom_network=None,
                    val_size=0.15,
                    beta=1.0 #how much to weigh inverse relation
                    ):
        super(NeuralJacobianKNN, self).__init__(gain= gain, state_extractor= state_extractor)

        self._memory = MemoryFactory.factory(memory, capacity=capacity)
        self._min_experience = min_experience

        self._gain = gain

        inputs = self.state_extractor.get_partial_state_dimensions()

        if use_linear:
            #TODO really, all network creation stuff should probably be external....
            self.network = nn.Linear(inputs, outputs)
        elif use_rbf:
            #arbitrary choice
            self.network =  RBFNetwork(inputs, 300, outputs)
        elif custom_network is not None:
            #TODO: since we change the actual inputs we will use in here, technically an external network should use
            #state_extractor get_partial_state_dimensions to set inputs
            self.network = custom_network
        else:
            self.network = nn.Sequential(*[
                        nn.Linear(inputs, 100),
                        nn.ReLU(),
                        nn.Linear(100, 100),
                        nn.ReLU(),
                        nn.Linear(100, outputs)
                    ])

        self._prev_obs = None
        self.k = k

        print("Using k={} neighbors".format(self.k))
        #vanilla optimizer
        print("nothing special with optimizer")
        self._optim = optim.Adam(self.network.parameters())

        print("random policy is hard coded")
        self.num_actuators = num_actuators
        self._rand_policy = OrnsteinUhlenbeckActionNoise(num_actuators, sigma=1.00)
        self.J = None
        self.fit_inverse_relation = fit_inverse_relation
        self.fit_null_space = fit_null_space
        self.beta = beta #how much to weigh learning inverse relation

        self._epochs = epochs
        self.train_loss = []
        self.val_loss = []
        self.val_size = val_size

    def load(self, pth='neural_uvs.npy', load_model=False):
        features = np.load(pth, allow_pickle=True).item()
        for s, sp in zip(features["S"], features["SP"]):
            self._memory.push(s, sp)

        if load_model:
            torch_pth = pth.split('.npy')[0]
            self.network.load_state_dict(torch.load(torch_pth + '_torch.pth'))
        else:
            print("did not load neural model from path")

    def save(self, pth='neural_uvs.npy'):
        S, SP = self._memory.get_tuples_as_lists()
        features = {"S": S, "SP": SP}
        with open(pth, 'wb') as f:
            np.save(f, features)

        torch_pth = pth.split('.npy')[0]
        torch.save(self.network.state_dict(), torch_pth + '_torch.pth')
        torch.save(self.train_loss, torch_pth + '_train_loss.pth')
        torch.save(self.val_loss, torch_pth + '_val_loss.pth')

    def learn(self, gym, external_data=None):
        if external_data is not None:
            print("external data found, using it all")
            self._min_experience= sum([len(v) for k,v in external_data.items()])
            self._memory._capacity = self._min_experience
            self._memory.flush()

            print("min experience is now {}".format(self._min_experience))

            for e in external_data.keys():
                self.reset()
                for step in external_data[e]:
                    a = self.act(step)

        obs = gym.reset()
        import time
        while not self.check_min_experience():
            #act pushes experience internally at the moment
            a = self.act(obs)
            obs, reward, done, info = gym.step(a)
            if done:
                obs = gym.reset()
                self.reset() #reset internal state

        print("Memory contains {} number of interactions".format(self._memory.get_index()))
        self.update_network()

    def reset(self):
        self._rand_policy.reset()
        self._prev_obs = None

    def create_tensor_dataset(self):
        #do something
        S, Sp = self._memory.get_tuples_as_lists()
        return torch.tensor(S).float(), torch.tensor(Sp).float()

    def null_space_loss(self, dX, dQ, J):
        iJ = torch.pinverse(J)
        dX_fit = dX.unsqueeze(-1)
        dQ = dQ.unsqueeze(-1) #unsqueeze so dimensions align
        yhat_p = torch.matmul(iJ, dX_fit)

        residual = (dQ - yhat_p) #residual of which null space we are for
        null_space = torch.matmul(J, residual).squeeze(-1)
    
        #null_space_transpose = torch.transpose(null_space, 1, 2)
        norm = torch.norm(null_space, 2, dim=1)

        #Jr=0 is what we are solving so target is 0
        # so MSE(Jr, 0) as we want to fine J that minimizes to 0
        target = torch.zeros(null_space.size())
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(null_space, target)

        return  loss

    def forward_jacobian(self, S, to_numpy=False):
        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S).float()

        if len(S.size()) <= 1.0:
            S = S.unsqueeze(0)

        S_input = self.state_extractor.get_batch_partial_state(S)

        vecs = self.network(S_input)
        #reshape as Jacobian
        d = self.state_extractor.num_points * self.state_extractor.point_dim
        J = vecs.view(-1, d, self.num_actuators)

        return J.detach().numpy() if to_numpy else J

    def update_network(self):
        #Create dataset
        S, _ = self.create_tensor_dataset()

        Q = self.state_extractor.get_batch_angles(S)
        X = self.state_extractor.get_batch_position(S)

        #dataset = torch.utils.data.TensorDataset(*dataset)
        dataset = KnnTargetDataset(S, Q, X, k=self.k, cache=True)
        train_length = int(len(dataset) * (1 - self.val_size))
        val_length = int(len(dataset) * self.val_size)
        dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length]) 

        print("Train size {}, val size {}".format(len(dataset), len(val_dataset)))
        #thought of using more threads, helped with 1st epoch (2x's faster) 
        #limitation to using more threads was...it slowed subsequent steps by like 10x's
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False) 

        epochs = self._epochs

        loss_fn = nn.MSELoss()
        best_model = self.network.state_dict().copy()
        best_val_loss = np.inf
        best_epoch = 0
        for e in range(epochs):
            self._optim.zero_grad()
            for b in dataloader:
                S = b[0]
                dQ = b[1]
                dX = b[2] 

                J = self.forward_jacobian(S)
                
                dQ_fit = dQ
                yhat = torch.matmul(J, dQ_fit)
                loss = loss_fn(yhat, dX)
                if self.fit_inverse_relation:
                    #minimize: || dQ - iJ*dX|| relation
                    iJ = torch.pinverse(J)
                    dX_fit = dX
                    yhat_p = torch.matmul(iJ, dX_fit)
                    aux_loss = self.beta * loss_fn(yhat_p, dQ)
                    loss = loss + aux_loss

                if self.fit_null_space:
                    aux_loss = self.null_space_loss(dX, dQ, J)
                    loss = loss + self.beta * aux_loss
                
                loss.backward()
                self._optim.step()
                self._optim.zero_grad()

            loss_fn = nn.MSELoss(reduction='sum')
            total_loss = 0.0
            for b in dataloader:
                S = b[0]
                dQ = b[1]
                dX = b[2]

                with torch.no_grad():
                    J = self.forward_jacobian(S)
                    
                    dQ_fit = dQ
                    yhat = torch.matmul(J, dQ_fit)
                
                    loss = loss_fn(yhat.squeeze(-1), dX) 
                    if self.fit_inverse_relation:
                        #minimize: || dQ - iJ*dX|| relation
                        iJ = torch.pinverse(J)
                        dX_fit = dX
                        yhat_p = torch.matmul(iJ, dX_fit)
                        aux_loss = self.beta * loss_fn(yhat_p, dQ)
                        loss = loss +  aux_loss

                    if self.fit_null_space:
                        #optimize null space
                        aux_loss = self.null_space_loss(dX, dQ, J)
                        loss = loss + self.beta * aux_loss

                    total_loss = total_loss + loss.item()
            #fyi, technically should be dividd by dataset * neighborhoodsize 
            #this is true at each part we do this
            self.train_loss.append(total_loss / len(dataset))
            #validation results
            total_loss = 0.0
            for b in val_dataloader:
                S = b[0]
                dQ = b[1]
                dX = b[2]

                with torch.no_grad():
                    J = self.forward_jacobian(S)
                    
                    dQ_fit = dQ
                    yhat = torch.matmul(J, dQ_fit)
                
                    loss = loss_fn(yhat, dX) 
                    if self.fit_inverse_relation:
                        #minimize: || dQ - iJ*dX|| relation
                        iJ = torch.pinverse(J)
                        dX_fit = dX
                        yhat_p = torch.matmul(iJ, dX_fit)
                        aux_loss = self.beta * loss_fn(yhat_p, dQ)
                        loss = loss + aux_loss

                    if self.fit_null_space:
                        aux_loss = self.null_space_loss(dX, dQ, J)
                        loss = loss + self.beta * aux_loss


                    total_loss = total_loss + loss.item()
            self.val_loss.append(total_loss / len(val_dataset))
            if self.val_loss[-1] <= best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_model = self.network.state_dict().copy()
                best_epoch = e

            print("Epoch {} mean train loss: {} val loss: {}".format(e, self.train_loss[-1], self.val_loss[-1]))
        print("Best epoch {}".format(best_epoch))
        self.network.load_state_dict(best_model)

            
    def check_min_experience(self):
        return self._memory.get_index() >= self._min_experience

    def act(self, obs):
        ths = self.state_extractor.get_angles(obs)
        q = ths

        psn, trg = self.state_extractor.get_position_and_target(obs)
        x_dot = trg - psn

        if self.check_min_experience():
            #q = joint angles, x = image features

            with torch.no_grad():
                J = self.forward_jacobian(obs, to_numpy = True)

            #d = np.linalg.det(J)
            d = 0.1
            if abs(d) < 1e-7 and abs(d) > -1e-7:
                print('singular matrix')
                action= self._gain * self._rand_policy.sample()
            else:
                #only use estimate if it's good
                J = np.squeeze(J)
                self.J = J
                iJ = np.linalg.pinv(J)
                th_dot = np.matmul(iJ, x_dot)
                action = self._gain * th_dot 
        else:
            #take a random action
            action = self._rand_policy.sample()

        #if self._prev_obs is not None:
        #might need to copy
        #KNN 
        self._memory.push(obs, obs)

        self._prev_obs = obs
        return action

class ReverseNeuralJacobianKNN(NeuralJacobianKNN):
    def __init__(self, gain, min_experience=DEFAULT_CAPACITY,
                    capacity=DEFAULT_CAPACITY,
                    num_actuators=7, 
                    memory="fixed",
                    inputs=27, #TODO this value is ignored, should probably remove it entirely....
                    outputs = 21, epochs=30,
                    state_extractor=None, fit_inverse_relation=False,
                    use_linear=False,
                    use_rbf=False,
                    custom_network=None,
                    val_size=0.15,
                    beta=1.0 #how much to weigh inverse relation
                    ):
        #fundamentally the same idea, but we directly predict iJ and the inverse is the original J
        super(ReverseNeuralJacobianKNN, self).__init__(
                    gain = gain, 
                    min_experience= min_experience,
                    capacity=capacity,
                    num_actuators= num_actuators, 
                    memory= memory,
                    inputs= inputs, 
                    outputs = outputs, epochs= epochs,
                    state_extractor= state_extractor, fit_inverse_relation= fit_inverse_relation,
                    use_linear= use_linear,
                    use_rbf= use_rbf,
                    custom_network= custom_network,
                    val_size= val_size,
                    beta= beta #how much to weigh inverse relation
                )

    def forward_jacobian(self, S, to_numpy=False):
        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S).float()

        if len(S.size()) <= 1.0:
            S = S.unsqueeze(0)

        S_input = self.state_extractor.get_batch_partial_state(S)

        vecs = self.network(S_input)
        #reshape as Jacobian
        d = self.state_extractor.num_points * self.state_extractor.point_dim

        #this is fundamentally the only major difference 
        iJ = vecs.view(-1, self.num_actuators, d)

        return iJ.detach().numpy() if to_numpy else iJ

    def update_network(self):
        #Create dataset
        dataset = self.create_tensor_dataset()
        dataset = torch.utils.data.TensorDataset(*dataset)

        train_length = int(len(dataset) * (1 - self.val_size))
        val_length = int(len(dataset) * self.val_size)
        dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length]) 

        print("Train size {}, val size {}".format(len(dataset), len(val_dataset)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False) 

        epochs = self._epochs

        loss_fn = nn.MSELoss()
        best_model = self.network.state_dict().copy()
        best_val_loss = np.inf
        best_epoch = 0
        for e in range(epochs):
            self._optim.zero_grad()
            for b in dataloader:
                S = b[0]
                Sp = b[1]
                
                dX, dQ = self.calculate_deltas_dX_dQ(S, Sp)
                
                #Directly predict iJ here
                #default is minimize || dQ - iJ*dX||
                #TODO this is broken, would need fixes
                iJ = self.forward_jacobian(S)
                dX_fit = dX.unsqueeze(-1)
                yhat = torch.matmul(iJ, dX_fit)
                loss = loss_fn(yhat.squeeze(-1), dQ)

                if self.fit_inverse_relation:
                    #minimize: || dx - J*dq|| relation
                    J = torch.pinverse(iJ)
                    dQ_fit = dQ.unsqueeze(-1)
                    yhat_p = torch.matmul(J, dQ_fit)
                    aux_loss = self.beta * loss_fn(yhat_p.squeeze(-1), dX)
                    loss = loss + aux_loss
                
                loss.backward()
                self._optim.step()
                self._optim.zero_grad()

            loss_fn = nn.MSELoss(reduction='sum')
            total_loss = 0.0
            for b in dataloader:
                S = b[0]
                Sp = b[1]

                with torch.no_grad():
                    dX, dQ = self.calculate_deltas_dX_dQ(S, Sp)
                    iJ = self.forward_jacobian(S)
                    dX_fit = dX.unsqueeze(-1)
                    yhat = torch.matmul(iJ, dX_fit)
                    loss = loss_fn(yhat.squeeze(-1), dQ)

                    if self.fit_inverse_relation:
                        #minimize: || dx - J*dq|| relation
                        J = torch.pinverse(iJ)
                        dQ_fit = dQ.unsqueeze(-1)
                        yhat_p = torch.matmul(J, dQ_fit)
                        aux_loss = self.beta * loss_fn(yhat_p.squeeze(-1), dX)
                        loss = loss + aux_loss
                    total_loss = total_loss + loss.item()
            
            self.train_loss.append(total_loss / len(dataset))
            total_loss = 0.0
            for b in val_dataloader:
                S = b[0]
                Sp = b[1]

                with torch.no_grad():
                    dX, dQ = self.calculate_deltas_dX_dQ(S, Sp)
                    iJ = self.forward_jacobian(S)
                    dX_fit = dX.unsqueeze(-1)
                    yhat = torch.matmul(iJ, dX_fit)
                    loss = loss_fn(yhat.squeeze(-1), dQ)

                    if self.fit_inverse_relation:
                        #minimize: || dx - J*dq|| relation
                        J = torch.pinverse(iJ)
                        dQ_fit = dQ.unsqueeze(-1)
                        yhat_p = torch.matmul(J, dQ_fit)
                        aux_loss = self.beta * loss_fn(yhat_p.squeeze(-1), dX)
                        loss = loss + aux_loss

                    total_loss = total_loss + loss.item()
            self.val_loss.append(total_loss / len(val_dataset))
            if self.val_loss[-1] <= best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_model = self.network.state_dict().copy()
                best_epoch = e

            print("Epoch {} mean train loss: {} val loss: {}".format(e, self.train_loss[-1], self.val_loss[-1]))
        print("Best epoch {}".format(best_epoch))
        self.network.load_state_dict(best_model)

    def act(self, obs):
        ths = self.state_extractor.get_angles(obs)
        q = ths

        psn, trg = self.state_extractor.get_position_and_target(obs)
        x_dot = trg - psn

        if self.check_min_experience():
            #q = joint angles, x = image features

            with torch.no_grad():
                iJ = self.forward_jacobian(obs, to_numpy = True)

            #d = np.linalg.det(J)
            d = 0.1
            if abs(d) < 1e-7 and abs(d) > -1e-7:
                print('singular matrix')
                action= self._gain * self._rand_policy.sample()
            else:
                #only use estimate if it's good
                iJ = np.squeeze(iJ)
                J = np.linalg.pinv(iJ)
                self.J = J
                
                th_dot = np.matmul(iJ, x_dot)
                action = self._gain * th_dot 
        else:
            #take a random action
            action = self._rand_policy.sample()

        #if self._prev_obs is not None:
        #might need to copy
        #since we use KNN we don't need sequential relation
        self._memory.push(obs, obs)#self._prev_obs, obs)

        self._prev_obs = obs
        return action


