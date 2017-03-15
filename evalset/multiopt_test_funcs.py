from evalset.test_funcs import TestFunction, lzip
import numpy

class MultioptTestFunction(TestFunction):

  def __init__(self, dim):
    super(MultioptTestFunction, self).__init__(dim)
    self.local_minima_loc = [] # Sorted in increasing order of function value at the local minima

class LowDMixtureOfGaussians(MultioptTestFunction):

  def __init__(self, dim=2):
    assert dim == 2
    super(LowDMixtureOfGaussians, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.502124885135
    self.fmax = 0
    self.local_minima_loc = [(-0.2, -0.5), (0.8, 0.3)]

  def do_evaluate(self, x):
    x1, x2 = x
    return -(
        .5 * numpy.exp(-10 * (.8 * (x1 + .2) ** 2 + .7 * (x2 + .5) ** 2)) +
        .5 * numpy.exp(-8 * (.3 * (x1 - .8) ** 2 + .6 * (x2 - .3) ** 2))
    )
