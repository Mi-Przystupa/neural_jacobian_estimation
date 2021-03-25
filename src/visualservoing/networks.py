from __future__ import generators
import torch.nn as nn
import numpy as np


class Network(object):

    def factory(type, **kwargs):
        if type == "mlp": 
            return MLP(**kwargs)
        elif type == "fixed":
            total_params = kwargs["num_params"]
            num_hiddens = kwargs["num_hiddens"]
            inputs = kwargs["inputs"]
            outputs = kwargs["outputs"]
            #num_layers really is number of intermitent layers
            h_size = hidden_size_for_fixed_params(num_layers= num_hiddens - 1, inputs= inputs, outputs= outputs, total_params= total_params)
            h_size = int(round(h_size))
            activation = kwargs["activation"]

            return MLP(**{"inputs": inputs, "hiddens": [h_size for i in range(num_hiddens)], "out": outputs, "activation": activation})

    factory = staticmethod(factory)

class MLP(nn.Module, Network):
    def __init__(self, inputs, hiddens, out, activation): 
        super(MLP, self).__init__()
        activation = self._select_activation(activation)
        layers = [nn.Linear(inputs, hiddens[0]), activation()]
        for (in_d, out_d) in zip(hiddens[:-1], hiddens[1:]):
            
            layers = layers + [nn.Linear(in_d, out_d)]
            layers = layers + [activation()]
        layers = layers + [nn.Linear(hiddens[-1], out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _select_activation(self, act):
        if act == 'tanh':
            return nn.Tanh
        elif act == 'relu':
            return nn.ReLU
        elif act == 'sigmoid':
            return nn.Sigmoid
        

def count_parameters(model):
    #source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def hidden_size_for_fixed_params(num_layers, inputs, outputs, total_params):
    #if you assume all the intermittent hidden layers have the same size you'll find the equations are quadratic
    #so just solve to find the roots
    a = num_layers
    b = (num_layers + inputs + outputs + 1)
    c = (outputs - total_params)
        
    return  -c / b if a == 0 else (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

if __name__ == "__main__":

    net = Network.factory("mlp", 
                        **{"inputs": 7, "hiddens": [100 for i in range(2)], "out": 3, "activation": "relu"})
    print(net)
    print(count_parameters(net))
    params = count_parameters(net)

    for N in [1, 2, 4, 8, 16, 32, 64, 128]: 

        hid_size = hidden_size_for_fixed_params(N - 1, 7, 3, params)

        print("{} hidden layers".format(N), hid_size)

        ceil = int(np.ceil(hid_size))
        floor = int(np.floor(hid_size))
        rounded = int(round(hid_size))

        net_ceil = Network.factory("mlp", **{"inputs": 7, "hiddens": [ceil for i in range(N)], "out": 3, "activation": "relu"})
        net_floor = Network.factory("mlp", **{"inputs": 7, "hiddens": [floor for i in range(N)], "out": 3, "activation": "relu"})
        net_rounded = Network.factory("mlp", **{"inputs": 7, "hiddens": [rounded for i in range(N)], "out": 3, "activation": "relu"})


        param_ceil = count_parameters(net_ceil)
        param_floor = count_parameters(net_floor)
        param_round = count_parameters(net_rounded)

        print('ceil', param_ceil - params, 'floor', param_floor - params,'round',param_round - params, 'params', params)
        print('')
        #print('floor', param_floor)


    print('sanity test+++++++++++++++++++++++++++++++++++++++')
    net = Network.factory("mlp", 
                        **{"inputs": 1, "hiddens": [98 for i in range(2)], "out": 1, "activation": "relu"})

    num_params = count_parameters(net)
    print(num_params)
    print(hidden_size_for_fixed_params(1, 1, 1, 9997))


    net = Network.factory("mlp", 
                        **{"inputs": 27, "hiddens": [100 for i in range(7)], "out": 3, "activation": "relu"})

    net = Network.factory("mlp",  **{"inputs": 27, "hiddens": [100 for i in range(7)], "out": 3, "activation": "relu"})
    num_params = count_parameters(net)
    print(num_params)
    print(hidden_size_for_fixed_params(6, 27, 3, num_params))








