import numpy as np

class stacker():
    def __init__(self, columns, inital_length=100):
        self._data = np.zeros((inital_length, columns))
        self.capacity = len(self._data)
        self.size = 0
        self.columns = columns

    def update(self, x):
        self._add(x)

    def _add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            temp = self._data
            self._data = np.zeros((self.capacity, self.columns))
            self._data[0:self.size, :] = temp
        self._data[self.size, :] = x
        self.size += 1

    def data(self):
        return self._data[0:self.size, :]

class replay_stacker():
    def __init__(self, columns, window_length=100):
        self._data = np.zeros((window_length, columns))
        self.capacity = window_length
        self.size = 0
        self.columns = columns

    def update(self, x):
        self._add(x)

    def _add(self, x):
        if self.size == self.capacity:
            self._data = np.roll(self._data, -1)
            self._data[self.size-1, :] = x
        else:
            self._data[self.size, :] = x
            self.size += 1

    def data(self):
        return self._data[0:self.size, :]


if __name__=='__main__':
    from datetime import datetime

    # Test general stacking
    # start = datetime.now()
    # test_array = stacker(10)
    # for i in range(0, 1000000):
    #     test_array.update(np.random.random(10))
    # print len(test_array.data())
    # print len(test_array._data)
    # print 'General Stacking : Time ', datetime.now() - start

    # Test Replay Memory
    start = datetime.now()
    test_array = replay_stacker(1, 10000)
    for i in range(0, 200000):
        test_array.update(i)
    print len(test_array.data())
    print len(test_array._data)
    print 'Replay Memory : Time ', datetime.now() - start
