from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.neighbors import KDTree
from visualservoing.local_least_square_uvs import DataSelector


class KnnTargetDataset(Dataset):

    def __init__(self, inputs, Q, X, k, cache=True):
        #this will be slow on first epoch because of calculating neighbors "online"
        self.inputs = inputs
        self.Q = Q
        self.X = X
        self.k = k
        self.cache = cache 

        self.neighbor_finder =  DataSelector(None, k)
        self.neighbor_finder.build_KD_tree(self.Q.numpy())

        if self.cache:
            #store targets for each data point
            self.targets = [None] * len(self.X)
        else:
            self.targets = None
            #if we don't cache...I can see this running fairly slow

    def createFiniteDifferenceDataset(self, q, x):
        [n, d] = q.shape
        [n, t] = x.shape

        dQ = torch.zeros(n*n, d)
        dX = torch.zeros(n*n, t)
        for i in range(n):
            for j in range(n):
                indx = i*n + j
                dQ[indx,:] = q[i,:] - q[j,:]
                dX[indx,:] = x[i,:] - x[j,:]

        return dQ.transpose(0, 1), dX.transpose(0, 1)


    def __getitem__(self, index):

        #for target
        inputs = self.inputs[index]
        q = self.Q[index] #use corresponding angles to find

        if self.cache and self.targets[index] is not None:
            dQ, dX = self.targets[index]
        else:
            (Q, X) = self.neighbor_finder.select_data(q, self.Q, self.X)
            dQ, dX = self.createFiniteDifferenceDataset(Q, X)
            if not self.targets is None:
                self.targets[index] = (dQ, dX)

        return inputs, dQ, dX


    def __len__(self):
        return len(self.X)

