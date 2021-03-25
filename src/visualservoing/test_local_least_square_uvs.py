import pytest
from visualservoing.local_least_square_uvs import DataSelector, LocalLeastSquareUVS
import numpy as np 


def check_float(value, test, thresh):
    assert value <= test + thresh and value >= test - thresh 

def check_float_vec(value, test, thresh):
    assert (value <= test + thresh).all() and (value >= test - thresh).all()

class TestLocalLeastSquareUVS:

    @pytest.fixture
    def smallData(self):
        q = np.array([
            [0.1, 0.2],
            [2.0, 2.0]
            ])
        x = np.array([
            [10., 5.],
            [2., 3.]
            ])

        return (q, x)

    @pytest.fixture
    def bigData(self):
        q = np.array([
            [0.5, 0.2],
            [0.2, 0.75],
            [1.0, 2.0]
            ])
        
        x = np.array([
            [13., 24.],
            [10., 9.],
            [40, 25],
            ])

        return (q, x)

    def testCreatingDataset(self, smallData, bigData):
        uvs = LocalLeastSquareUVS(min_experience = 2, k=5)

        q, x = smallData
        dQ, dX = uvs.createFiniteDifferenceDataset(q, x)
        assert len(dQ) == 4 and len(dX) == 4
        trueQ = np.array([
                [0., 0.],
                [-1.9, -1.8],
                [1.9, 1.8],
                [0., 0.]
            ])
        trueX = np.array([
                [0., 0.],
                [8.,2.],
                [-8.,-2.],
                [0., 0.]
            ])

        for i, dq in enumerate(dQ):
            check_float_vec(dq, trueQ[i], 1e-6)
        for i, dx in enumerate(dX):
            check_float_vec(dx, trueX[i], 1e-6)


        q, x = bigData
        dQ, dX = uvs.createFiniteDifferenceDataset(q, x)

        #TODO add test fof 3 entries as well


class TestDataSelector:

    @pytest.fixture
    def data(self):

        q = np.array([
            [1.0, 2.0],
            [0.1, 0.2],
            [1.5, 0.0],
            [2.0, 2.0]
            ])
        return q

    @pytest.fixture
    def targets(self):
        targets = np.array([0, 1, 2, 3])
        targets = targets.reshape(-1, 1)
        return targets

    def testValidConstructor(self):

        #Test constructing with Epislon
        selector = DataSelector(eps=.03)
        check_float(selector.get_eps(), 0.03, 1e-5)
        assert selector.get_k() is None

        selector = DataSelector(eps = 1.0)
        check_float(selector.get_eps(), 1.0, 1e-5)
        assert selector.get_k() is None

        #Test constructing with K
        selector = DataSelector(k=10)
        assert selector.get_k() == 10
        assert selector.get_eps() is None

        selector = DataSelector(k = 100)
        assert selector.get_k() == 100
        assert selector.get_eps() is None
        
    def testInvalidConstructor(self):
        #actual realistic case
        with pytest.raises(ValueError):
            selector = DataSelector(eps=.01, k=10)
        #should raise error so long as you put anything in 
        with pytest.raises(ValueError):
            selector = DataSelector(eps="23412", k=0.1)

    def test_do_KNN(self, data, targets):
        k = 2
        selector = DataSelector(k=k)
        q = np.array([0., 0.])
        neighbors, targs = selector._do_KNN(q, data, targets)
        assert len(neighbors) == k
        #should be sorted
        assert (targs == np.array([[1], [2]])).all()
        assert (neighbors[0] == data[1]).all()
        assert (neighbors[1] == data[2]).all()

        k = 3
        selector = DataSelector(k=k)
        q = np.array([2., 2.])
        neighbors, targs = selector._do_KNN(q, data, targets)

        assert len(neighbors) == 3
        assert (targs == np.array([[3], [0], [2]])).all()
        assert (neighbors[0] == data[3]).all()
        assert (neighbors[1] == data[0]).all()
        assert (neighbors[2] == data[2]).all()

    def test_do_norm(self, data, targets):
        eps = 0.3
        q = np.array([0., 0.])
        selector = DataSelector(eps=eps)
        neighbors, _ = selector._do_norm(q, data)
        assert len(neighbors) == 1
        assert (neighbors[0] == data[1]).all()

        eps = 1.5
        selector = DataSelector(eps=eps)
        neighbors, targs = selector._do_norm(q, data, targets)
        assert len(neighbors) == 2
        assert (targs == np.array([[1], [2]])).all()
        assert (neighbors[0] == data[1]).all()
        assert (neighbors[1] == data[2]).all()

        eps = 0.5
        q = np.array([1.5, 2.0])
        neighbors, targs = selector._do_norm(q, data, targets)
        assert len(neighbors) == 2
        assert (targs == np.array([[0], [3]])).all()
        assert (neighbors[0] == data[0]).all()
        assert (neighbors[1] == data[3]).all()

