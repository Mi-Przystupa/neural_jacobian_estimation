import pytest
from simulator.ammortized_local_least_square_uvs import AmmortizedLocalLeastSquareUVS
import numpy as np
import torch 

#theres a better way...
INPUTS = 10
BATCH = 32


class TestAmmortizedLLSUVS:

    @pytest.fixture
    def policy(self):
        pol = AmmortizedLocalLeastSquareUVS(min_experience=300, max_action=0.5, gain=1.0,
                    memory="fixed", img_w=1.0, img_h=1.0, inputs=10, outputs = 4)
        return pol

    def test_forward_jacobian_shape(self, policy):
        #these inputs are invalid, only thing we can definitively test are shapes and output types

        S = torch.randn(BATCH, INPUTS)

       
        # Training policy scenario
        J = policy.forward_jacobian(S, to_numpy=False)
        assert isinstance(J, torch.Tensor)
        assert len(J.size()) == 3
        assert J.size()[0] == BATCH
        assert J.size()[1] == 2 and J.size()[2] == 2

        # running policy scenario
        S = np.random.randn(10)

        J = policy.forward_jacobian(S, to_numpy=True)
        assert isinstance(J, np.ndarray)
        assert len(J.shape) == 3
        assert J.shape[0] == 1
        assert J.shape[1] == 2 and J.shape[2] == 2


     







