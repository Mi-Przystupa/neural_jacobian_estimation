import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from visualservoing.memory import MemoryFactory 
from visualservoing.policy_base import Policy
from visualservoing.utils import RBFNetwork, ExtremeLearning
from random_policy import OrnsteinUhlenbeckActionNoise

DEFAULT_CAPACITY=100000
class BlackBoxForwardKinematicsUVS(Policy):

    def __init__(self, gain, min_experience=DEFAULT_CAPACITY,
                    capacity=DEFAULT_CAPACITY,
                    num_actuators=7, 
                    num_feats = 3,
                    memory="fixed",
                    epochs=30,
                    state_extractor=None, 
                    use_linear=False,
                    use_rbf=False,
                    use_el =False,
                    direct_optimize=False,
                    custom_network=None,
                    val_size=0.15,
                    l2 = 0.0
                    ):
        super(BlackBoxForwardKinematicsUVS, self).__init__(gain= gain, state_extractor= state_extractor)

        self._memory = MemoryFactory.factory(memory, capacity=capacity)
        self._min_experience = min_experience
        self.direct_optimize = direct_optimize

        self._gain = gain
        if use_linear:
            #TODO really, all network creation stuff should probably be external....
            self.network = nn.Linear(num_actuators, num_feats)
        elif use_el:
            self.network = ExtremeLearning(num_actuators, 50000, num_feats)
        elif use_rbf:
            #arbitrary choice
            self.network =  RBFNetwork(num_actuators, 300, num_feats)
        elif custom_network is not None:
            self.network = custom_network
        else:
            self.network = nn.Sequential(*[
                        nn.Linear(num_actuators , 100),
                        nn.ReLU(),
                        nn.Linear(100, 100),
                        nn.ReLU(),
                        nn.Linear(100, num_feats)
                    ])

        self._prev_obs = None

        #vanilla optimizer
        print("nothing special with optimizer, using l2 = {}".format(l2))
        self._optim = optim.Adam(self.network.parameters(), weight_decay= l2)
        self.l2 = l2

        print("random policy is hard coded")
        self.num_actuators = num_actuators
        self.num_feats = num_feats 
        self._rand_policy = OrnsteinUhlenbeckActionNoise(num_actuators, sigma=1.00)
        self.J = None

        self._epochs = epochs
        self.train_loss = []
        self.val_loss = []
        self.val_size = val_size

    def load(self, pth='neural_uvs.npy', load_model=False):
        features = np.load(pth, allow_pickle=True).item()
        for s, sp in zip(features["S"], features["SP"]):
            self._memory.push(s, sp)

        if load_model:
            torch_pth = pth.split('.')[0]
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

    def update_network(self):
        #Create dataset
        dataset = self.create_tensor_dataset()
        dataset = torch.utils.data.TensorDataset(*dataset)

        train_length = int(len(dataset) * (1 - self.val_size))
        val_length = int(len(dataset) * self.val_size)
        dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length]) 

        print("Train size {}, val size {}".format(len(dataset), len(val_dataset)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        #batch size doesn't really  matter for validation....
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False) 
        epochs = self._epochs

        loss_fn = nn.MSELoss()
        best_model = self.network.state_dict().copy()
        best_val_loss = np.inf
        best_epoch = 0
        for e in range(epochs):
            self._optim.zero_grad()
            for b in dataloader:
                Qs = b[0]
                pose = b[1]
                
                pose_hat = self.network(Qs)
                loss = loss_fn(pose_hat, pose)

                loss.backward()
                self._optim.step()
                self._optim.zero_grad()
            #check train loss
            loss_fn = nn.MSELoss(reduction='sum')
            total_loss = 0.0
            for b in dataloader:
                S = b[0]
                Sp = b[1]

                with torch.no_grad():
                    Qs = b[0]
                    pose = b[1]
                    
                    pose_hat = self.network(Qs)
                    loss = loss_fn(pose_hat, pose)

                    total_loss = total_loss + loss.item()

            self.train_loss.append(total_loss / len(dataset))
            #check validation loss
            total_loss = 0.0
            for b in val_dataloader:
                S = b[0]
                Sp = b[1]

                with torch.no_grad():
                    Qs = b[0]
                    pose = b[1]
                    
                    pose_hat = self.network(Qs)
                    loss = loss_fn(pose_hat, pose)

                    total_loss = total_loss + loss.item()
            self.val_loss.append(total_loss / len(val_dataset))
            if self.val_loss[-1] <= best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_model = self.network.state_dict().copy()
                best_epoch = e
            print("Epoch {} mean train loss: {} val loss: {}".format(e, self.train_loss[-1], self.val_loss[-1]))
        print("Best epoch {}".format(best_epoch))
        self.network.load_state_dict(best_model)
            

    def JacobianEstimate(self, q):
        J = torch.zeros(self.num_feats, self.num_actuators)

        self.network.eval()
        for i in range(self.num_feats):
            self.network.zero_grad()
            q_grad = torch.tensor(q.tolist(), requires_grad = True)

            approx_pose = self.network(q_grad).squeeze()
            approx_pose[i].backward()
            J[i,:] = q_grad.grad
        self.network.train()

        return J

    def check_min_experience(self):
        return self._memory.get_index() >= self._min_experience

    def act(self, obs):
        ths = self.state_extractor.get_angles(obs)
        q = ths

        psn, trg = self.state_extractor.get_position_and_target(obs)
        x_dot = trg - psn

        if self.check_min_experience():
            #q = joint angles, x = image features
            J = self.JacobianEstimate(q)
            self.J = J

            if self.direct_optimize:
                #we can actually just do this directly...
                #interesintgly not 100% perfect
                q_grad = torch.tensor(q.tolist(), requires_grad=True)
                target = torch.tensor(trg.tolist())
                x_true = torch.tensor(psn.tolist())

                self.network.train()
                x_hat = self.network(q_grad).squeeze()
                #L1 norm or L2 norm seem best bet
                #L1 (based on a single run mind you) 
                # seemed to better converge to the desired point
                # L2 seemed to exhibit a more...rigid limitation for minima
                loss = torch.norm(target - x_hat, 1)#0.5 * torch.norm(target - x_hat, 1) #**2

                loss.backward()
                #need negative sign to make it a descent step
                action = -1.0* q_grad.grad.clone().numpy() * self._gain
                self.network.zero_grad()
                self.network.train()
            else:
                iJ = np.linalg.pinv(J)
                th_dot = np.matmul(iJ, x_dot)
                action = self._gain * th_dot 
        else:
            #take a random action
            action = self._gain * self._rand_policy.sample()

        #might need to copy
        self._memory.push(ths, psn)

        return action


