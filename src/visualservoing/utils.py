import torch
import torch.nn as nn
from visualservoing.rbf_layer import RBF, basis_func_dict

class RadialBasisFunction(nn.Module):
    def __init__(self, dims, smoothing=1e6):
        super(RadialBasisFunction, self).__init__()
        self.u = nn.Parameter(torch.zeros(1, dims).normal_(0., 1.)) 
        self.precision = nn.Parameter(torch.zeros(1, dims).normal_(0., 1.))

    def forward(self, x):
        diff = torch.pow(x - self.u, 2)
        xpx = diff / torch.pow(self.precision, 2)
        return torch.exp(-0.5 * torch.sum(xpx, dim=1))

class RBFNetwork(nn.Module):

    def __init__(self, input_dim, num_basis, outputs, basis_func='gaussian'):
        super(RBFNetwork, self).__init__()
        #self.basis = nn.ModuleList([RadialBasisFunction(input_dim) for _ in range(num_basis)]) 
        basis_func = basis_func_dict()[basis_func]
        self.basis = RBF(input_dim, num_basis, basis_func)

        self.w = nn.Linear(num_basis, outputs, bias=True)
        self.num_basis = num_basis

    def forward(self, x):

        if len(x.size()) < 2:
            x = x.unsqueeze(0)
        """
        b = x.size(0)
        x_p = []
        for x_i in x:
            x_i_b = []
            for b_i in self.basis:
                x_i_b.append( b_i(x_i))          
            x_p.append(torch.cat(x_i_b))
        x_p = torch.stack(x_p)
        """
        x_p = self.basis(x)
        return self.w(x_p)


class ExtremeLearning(nn.Module):
    def __init__(self, inp, hid, outputs):
        super(ExtremeLearning, self).__init__()
        self.input = nn.Linear(inp, hid)
        #self.W = nn.Parameter(torch.randn(hid, outputs))
        self.model = nn.Linear(hid, outputs)


    def forward(self, x):
        x = torch.sigmoid(self.input(x))
        if self.training:
            #Keep the hidden layer features FIXED
            x = x.detach()
            
        return self.model(x)

    def fit_model(self, X, Y ):
        with torch.no_grad():
            PHI = self.input(X)

        PHI_I = torch.pinverse(PHI)
        self.W.data = torch.matmul(PHI_I, Y)

        



