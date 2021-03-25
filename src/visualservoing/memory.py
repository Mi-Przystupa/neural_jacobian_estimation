import numpy as np


class MemoryFactory:
    def factory(type, capacity=1e6):
        type = type.lower()
        if type == "fixed":
            return FixedMemory(capacity)
        elif type == "circular":
            return CircularMemory(capacity)
        else:
            raise ValueError("Invalid memory type {}".format(type))
        
    factory = staticmethod(factory)

class Memory:
    
    def __init__(self, capacity=1e6):
        self._capacity = int(capacity)
        self._buffer = []
        self._index = 0
        self._is_full = False
        self.flush()

    def flush(self):
        self._buffer = [None] * self._capacity
        self._index = 0
        self._is_full = False


    def push(self, x, y):
        assert False, "need to implement" 

    def get_is_full(self):
        return self._is_full

    def get_capacity(self):
        return self._capacity

    def get_index(self):
        return self._index

    def get_tuples_as_lists(self):
        s_a_sp_r_d = self._buffer

        dim_to_list = lambda i: [x[i] for x in s_a_sp_r_d if x is not None]
        x = dim_to_list(0)
        y = dim_to_list(1)

        return x, y
    
class FixedMemory(Memory):
    def __init__(self, capacity=1e6):
        super(FixedMemory, self).__init__(capacity)

    def push(self, x, y):
        if not self._is_full:
            self._buffer[self._index] = [x, y]
            self._index = self._index + 1 
            self._is_full = (self._index ) >= self._capacity 



class CircularMemory(Memory):
    def __init__(self, capacity=1e6):
        super(CircularMemory, self).__init__(capacity)

    def push(self, x, y):
        self._buffer[self._index] = [x, y]
        entries = self._index + 1
        self._is_full = entries >= self._capacity or self._is_full
        self._index = (entries) % self._capacity





        
