from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np

np.seterr(all='ignore')

class TestFunction(object):
    __metaclass__ = ABCMeta

    def __init__(self, dim, verify=True):
        assert dim > 0
        self.dim = dim
        self.verify = verify
        self.num_evals = 0
        self.min_loc = None
        self.fmin = None
        self.local_fmin = []
        self.fmax = None
        self.bounds = None
        self.classifiers = []

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.dim)

    def evaluate(self, x):
        if self.verify and (not isinstance(x, np.ndarray) or x.shape != (self.dim, )):
            raise ValueError('Argument must be a numpy array of length {}'.format(self.dim))
        self.num_evals += 1
        value = self.do_evaluate(x)
        to_be_returned = value.item() if hasattr(value, 'item') else value
        return to_be_returned

    @abstractmethod
    def do_evaluate(self, x):
        raise NotImplementedError

class ShiftedBranin01(TestFunction):
    def __init__(self, dim=2, shift=None):
        assert dim == 2
        assert shift is None or isinstance(shift, np.ndarray)
        if isinstance(shift, np.ndarray):
            assert len(shift.shape) == 1
            assert shift.shape[0] == 2
        super(Branin01, self).__init__(dim)
        self.bounds = [[-5, 10], [0, 15]]
        self.min_loc = [-np.pi, 12.275]
        self.fmin = 0.39788735772973816
        self.fmax = 308.129096012
        self.classifiers = ['multi_min']
        if shift is None:
            self.shift = np.array([0.0, 0.0])
        else:
            self.shift = shift

    def do_evaluate(self, x):
        x1, x2 = x - self.shift
        return (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


