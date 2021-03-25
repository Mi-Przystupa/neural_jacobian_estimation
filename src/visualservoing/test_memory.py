import pytest
import numpy as np
from simulator.memory import Memory, FixedMemory, CircularMemory, MemoryFactory


class TestMemoryFactory:

    @pytest.fixture
    def large_memory(self):
        args = {"capacity": 4e6}
        return args

    @pytest.fixture
    def small_memory(self):
        args = {"capacity": 2}
        return args

    def test_get_fix_memory(self, large_memory, small_memory):
        memory = MemoryFactory.factory("fixed", **small_memory)
        assert isinstance(memory, FixedMemory) 
        assert memory.get_capacity() == int(small_memory["capacity"])

        memory = MemoryFactory.factory("Fixed")
        assert isinstance(memory, FixedMemory)

        memory = MemoryFactory.factory("FIXED", **large_memory)
        assert isinstance(memory, FixedMemory)
        assert memory.get_capacity() == int(large_memory["capacity"])

    def test_get_circular_memory(self, large_memory, small_memory):
        memory = MemoryFactory.factory("circular", **small_memory)
        assert isinstance(memory, CircularMemory) 
        assert memory.get_capacity() == int(small_memory["capacity"])

        memory = MemoryFactory.factory("CiRcular")
        assert isinstance(memory, CircularMemory)

        memory = MemoryFactory.factory("CIRCULAR", **large_memory)
        assert isinstance(memory, CircularMemory)
        assert memory.get_capacity() == int(large_memory["capacity"])


    def test_invalid_memory(self):
        with pytest.raises(ValueError):
            memory = MemoryFactory.factory("fakememory")

        with pytest.raises(AttributeError):
            memory = MemoryFactory.factory(12345)

        with pytest. raises(ValueError):
            memory = MemoryFactory.factory("fixed12345")

        with pytest. raises(ValueError):
            memory = MemoryFactory.factory("circular12345")

@pytest.fixture
def transition_tuples():
    values = []

    s = np.array([0., 0.])
    for i in range(0, 4):
        a = i % 2
        sp = s.copy()
        sp[a] = sp[a] + 1
        a = np.array([a])
        values.append([s.copy(), a.copy()])
        s = sp.copy()

    return values




class TestFixedMemory:
    @pytest.fixture
    def fixed_memory(self):
        args = {"capacity": 2}
        memory = MemoryFactory.factory("fixed", **args)
        return memory

        
    def test_push_tuple(self, fixed_memory):
        s = np.array([0., 0.])
        a = np.array([1.])
        done = False

        fixed_memory.push(s, a)
        assert not fixed_memory.get_is_full()

        fixed_memory.push(s, a)
        assert fixed_memory.get_is_full()

        fixed_memory.push(s, a)
        assert fixed_memory.get_is_full() and fixed_memory.get_index() == fixed_memory.get_capacity()

    def test_check_lists(self, fixed_memory, transition_tuples):
        for t in transition_tuples:
            fixed_memory.push(t[0], t[1])

        (s, a) = fixed_memory.get_tuples_as_lists()
        #should have equal sized lists
        l = fixed_memory.get_capacity()
        assert len(s) == l and len(a) == l 

        #should be pushed in order we expected
        for i in range(len(s)):
            t = transition_tuples[i]
            assert (t[0] == s[i]).all()
            assert (t[1] == a[i]).all()

        fixed_memory.flush()
        (s, a) = fixed_memory.get_tuples_as_lists()
        assert len(s) == 0 and 0 == len(a)

        for i, t in enumerate(transition_tuples):
            fixed_memory.push(t[0], t[1])
            if i >= 0:
                break

        (s, a) = fixed_memory.get_tuples_as_lists()
        assert len(s) == 1 and 1 == len(a)
        t = transition_tuples[0]
        assert (t[0] == s[0]).all()
        assert (t[1] == a[0]).all()


class TestCircularMemory:
    @pytest.fixture
    def circular_memory(self):
        args = {"capacity": 2}
        memory = MemoryFactory.factory("circular", **args)
        return memory

    def test_push_tuple(self, circular_memory):
        s = np.array([0., 0.])
        a = np.array([1.])

        circular_memory.push(s, a)
        assert not circular_memory.get_is_full()
        #after filling memory, index should go back to start of list
        circular_memory.push(s, a)
        assert circular_memory.get_is_full()
        assert circular_memory.get_index() == 0
        #after memory is full, index should update
        circular_memory.push(s, a)
        assert circular_memory.get_is_full() and circular_memory.get_index() == 1

    def test_check_lists(self, circular_memory, transition_tuples):
        for t in transition_tuples:
            circular_memory.push(t[0], t[1])

        (s, a) = circular_memory.get_tuples_as_lists()
        #should have equal sized lists
        l = circular_memory.get_capacity()
        assert len(s) == l and len(a) == l 

        #should be pushed in order we expected
        for i, p in enumerate([2, 3]):
            t = transition_tuples[p]
            assert (t[0] == s[i]).all()
            assert (t[1] == a[i]).all()

        circular_memory.flush()
        (s, a) = circular_memory.get_tuples_as_lists()
        assert len(s) == 0 and 0 == len(a)

        for i, t in enumerate(transition_tuples):
            circular_memory.push(t[0], t[1])
            if i >= 0:
                break

        (s, a) = circular_memory.get_tuples_as_lists()
        assert len(s) == 1 and 1 == len(a) 
        t = transition_tuples[0]
        assert (t[0] == s[0]).all()
        assert (t[1] == a[0]).all()

        

