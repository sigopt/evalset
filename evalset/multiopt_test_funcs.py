from evalset.test_funcs import TestFunction, lzip
import numpy

class MultioptTestFunction(TestFunction):

  def __init__(self, dim):
    super(MultioptTestFunction, self).__init__(dim)
    self.local_minima_loc = [] # Sorted in increasing order of function value at the local minima

class LowDMixtureOfGaussians01(MultioptTestFunction):

  def __init__(self, dim=2):
    assert dim == 2
    super(LowDMixtureOfGaussians01, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.50212488514
    self.fmax = -0.00001997307
    self.local_minima_loc = [(-0.2, -0.5), (0.8, 0.3)]

  def do_evaluate(self, x):
    x1, x2 = x
    return -(
        .5 * numpy.exp(-10 * (.8 * (x1 + .2) ** 2 + .7 * (x2 + .5) ** 2)) +
        .5 * numpy.exp(-8 * (.3 * (x1 - .8) ** 2 + .6 * (x2 - .3) ** 2))
    )

class LowDMixtureOfGaussians02(MultioptTestFunction):

  def __init__(self, dim=2):
    assert dim == 2
    super(LowDMixtureOfGaussians02, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.63338923402
    self.fmax = -0.00993710053
    self.local_minima_loc = [(-0.17918253215, -0.46292606370),
                             (0.79994926640, 0.29998223419)]

  def do_evaluate(self, x):
    x1, x2 = x
    return -(
        .5 * numpy.exp(-10 * (.8 * (x1 + .2) ** 2 + .7 * (x2 + .5) ** 2)) +
        .5 * numpy.exp(-2 * (.3 * (x1 - .8) ** 2 + .6 * (x2 - .3) ** 2))
    )

class LowDMixtureOfGaussians03(MultioptTestFunction):

  def __init__(self, dim=2):
    assert dim == 2
    super(LowDMixtureOfGaussians03, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.68874252567
    self.fmax = -0.00207854059
    self.local_minima_loc = [(-0.04454170197, 0.03290524075),
                             (0.29681835510, -0.79175480224),
                             (-0.78382191954, 0.29696540520),
                             (-0.79970566681, -0.79755011032),
                             (0.79955980829, 0.79964895583)]

  def do_evaluate(self, x):
    x1, x2 = x
    return -(
        .5 * numpy.exp(-10 * (.8 * (x1 + .8) ** 2 + .7 * (x2 + .8) ** 2)) +
        .5 * numpy.exp(-8 * (.3 * (x1 + .8) ** 2 + .6 * (x2 - .3) ** 2)) +
        .5 * numpy.exp(-9 * (.8 * x1 ** 2 + .7 * x2 ** 2)) +
        .5 * numpy.exp(-9 * (.8 * (x1 - .3) ** 2 + .7 * (x2 + .8) ** 2)) +          
        .5 * numpy.exp(-10 * (.8 * (x1 - .8) ** 2 + .7 * (x2 - .8)** 2))
    )

class MidDMixtureOfGaussians01(MultioptTestFunction):

  def __init__(self, dim=8):
    assert dim == 8
    super(MidDMixtureOfGaussians01, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.50212691955
    self.fmax = -0.00001997307
    self.local_minima_loc = [(-0.2, -0.5, 0, 0, 0, 0, 0, 0),
                             (0.8, 0.3, 0, 0, 0, 0, 0, 0)]

  def do_evaluate(self, x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    return -(
        .5 * numpy.exp(-10 * (.8 * (x1 + .2) ** 2 + .7 * (x2 + .5) ** 2)) +
        .5 * numpy.exp(-8 * (.3 * (x1 - .8) ** 2 + .6 * (x2 - .3) ** 2))
    )


class MidDMixtureOfGaussians02(MultioptTestFunction):

  def __init__(self, dim=8):
    assert dim == 8
    super(MidDMixtureOfGaussians02, self).__init__(dim)
    self.bounds = lzip([-1] * self.dim, [1] * self.dim)
    self.fmin = -0.50016818373
    self.fmax = -0.00004539993
    self.local_minima_loc = [(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5),
                             (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]

  def do_evaluate(self, x):
    mu1 = 0.5 * numpy.ones(8)
    mu2 = -0.5 * numpy.ones(8)
    return -(
        0.5 * numpy.exp(-sum((x - mu1)**2)) +
        0.5 * numpy.exp(-sum((x - mu2)**2))
    )
