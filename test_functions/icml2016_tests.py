"""
    These are the functions that were used to create the results in the submitted paper
        A Strategy for Ranking Optimizers using Multiple Criteria

    Because certain functions have special properties, additional modifiers may be present to specify the
    properties of the optimization.
      * All functions have a name and dimension corresponding to a function in the test_funcs.py file.
      * If test['int'] and test['res'] are both None, just the baseline function was tested:
           import test_funcs
           test_function_class = getattr(test_funcs, test['name'])
           test_function = test_function_class(test['dim'])
      * If test['res'] is not None then the function is supposed to be run with a fixed resolution
        This is a function that can only return a discrete set of values
           import test_funcs
           test_function_class = getattr(test_funcs, test['name'])
           test_function_base = test_function_class(test['dim'])
           test_function = test_funcs.Discretizer(test_function_base, test['res'])
      * If test['int'] is not None it contains a list of the dimensions which should only accept integer values
        This is just a flag, to alert you to the test that we ran; there is no additional tool for integer testing
"""

# Functions used in the nonparametric/parametric comparison (Table 2)
tests_for_nonparametric = [
    {'name': 'Ackley', 'dim': 11, 'int': None, 'res': None},
    {'name': 'Ackley', 'dim': 3, 'int': None, 'res': 1},
    {'name': 'Adjiman', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Alpine02', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'CarromTable', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Csendes', 'dim': 2, 'int': None, 'res': None},
    {'name': 'DeflectedCorrugatedSpring', 'dim': 4, 'int': None, 'res': None},
    {'name': 'DeflectedCorrugatedSpring', 'dim': 7, 'int': None, 'res': None},
    {'name': 'Easom', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Easom', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Easom', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Hartmann3', 'dim': 3, 'int': [0], 'res': None},
    {'name': 'Hartmann6', 'dim': 6, 'int': None, 'res': 10},
    {'name': 'HelicalValley', 'dim': 3, 'int': None, 'res': None},
    {'name': 'LennardJones6', 'dim': 6, 'int': None, 'res': None},
    {'name': 'McCourt01', 'dim': 7, 'int': None, 'res': 10},
    {'name': 'McCourt03', 'dim': 9, 'int': None, 'res': None},
    {'name': 'McCourt06', 'dim': 5, 'int': None, 'res': None},
    {'name': 'McCourt07', 'dim': 6, 'int': None, 'res': 12},
    {'name': 'McCourt08', 'dim': 4, 'int': None, 'res': None},
    {'name': 'McCourt09', 'dim': 3, 'int': None, 'res': None},
    {'name': 'McCourt10', 'dim': 8, 'int': None, 'res': None},
    {'name': 'McCourt11', 'dim': 8, 'int': None, 'res': None},
    {'name': 'McCourt12', 'dim': 7, 'int': None, 'res': None},
    {'name': 'McCourt13', 'dim': 3, 'int': None, 'res': None},
    {'name': 'McCourt14', 'dim': 3, 'int': None, 'res': None},
    {'name': 'McCourt16', 'dim': 4, 'int': None, 'res': None},
    {'name': 'McCourt16', 'dim': 4, 'int': None, 'res': 10},
    {'name': 'McCourt17', 'dim': 7, 'int': None, 'res': None},
    {'name': 'McCourt18', 'dim': 8, 'int': None, 'res': None},
    {'name': 'McCourt19', 'dim': 2, 'int': None, 'res': None},
    {'name': 'McCourt20', 'dim': 2, 'int': None, 'res': None},
    {'name': 'McCourt23', 'dim': 6, 'int': None, 'res': None},
    {'name': 'McCourt26', 'dim': 3, 'int': None, 'res': None},
    {'name': 'McCourt28', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Michalewicz', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Michalewicz', 'dim': 4, 'int': None, 'res': 20},
    {'name': 'Michalewicz', 'dim': 8, 'int': None, 'res': None},
    {'name': 'Mishra06', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Ned01', 'dim': 2, 'int': None, 'res': None},
    {'name': 'OddSquare', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Parsopoulos', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Pinter', 'dim': 2, 'int': [0, 1], 'res': None},
    {'name': 'Plateau', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Problem03', 'dim': 1, 'int': None, 'res': None},
    {'name': 'RosenbrockLog', 'dim': 11, 'int': None, 'res': None},
    {'name': 'Sargan', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Sargan', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Schwefel20', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Schwefel20', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Schwefel36', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Shekel05', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Sphere', 'dim': 7, 'int': [0, 1, 2, 3, 4], 'res': None},
    {'name': 'StyblinskiTang', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Tripod', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Xor', 'dim': 9, 'int': None, 'res': None},
]

# Functions used in the with/withou AUC comparison (Table 3)
tests_for_auc = [
    {'name': 'Ackley', 'dim': 3, 'int': None, 'res': None},
    {'name': 'Ackley', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Ackley', 'dim': 11, 'int': None, 'res': None},
    {'name': 'Ackley', 'dim': 3, 'int': None, 'res': 1},
    {'name': 'Ackley', 'dim': 11, 'int': [0, 1, 2], 'res': None},
    {'name': 'Branin02', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Bukin06', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'CarromTable', 'dim': 2, 'int': None, 'res': None},
    {'name': 'CarromTable', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Deb02', 'dim': 6, 'int': None, 'res': None},
    {'name': 'DeflectedCorrugatedSpring', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Easom', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Easom', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Exponential', 'dim': 6, 'int': None, 'res': None},
    {'name': 'Hartmann3', 'dim': 3, 'int': None, 'res': None},
    {'name': 'LennardJones6', 'dim': 6, 'int': None, 'res': None},
    {'name': 'McCourt01', 'dim': 7, 'int': None, 'res': 10},
    {'name': 'McCourt02', 'dim': 7, 'int': None, 'res': None},
    {'name': 'McCourt06', 'dim': 5, 'int': None, 'res': 12},
    {'name': 'McCourt07', 'dim': 6, 'int': None, 'res': 12},
    {'name': 'McCourt19', 'dim': 2, 'int': None, 'res': None},
    {'name': 'McCourt22', 'dim': 5, 'int': None, 'res': None},
    {'name': 'McCourt27', 'dim': 3, 'int': None, 'res': None},
    {'name': 'Michalewicz', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Mishra06', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Ned01', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Plateau', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Rastrigin', 'dim': 8, 'int': None, 'res': None},
    {'name': 'Rastrigin', 'dim': 8, 'int': None, 'res': .1},
    {'name': 'Sargan', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Schwefel20', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Schwefel20', 'dim': 2, 'int': [0], 'res': None},
    {'name': 'Shekel05', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Shekel07', 'dim': 4, 'int': None, 'res': None},
    {'name': 'Sphere', 'dim': 7, 'int': None, 'res': None},
    {'name': 'Sphere', 'dim': 7, 'int': [0, 1, 2, 3, 4], 'res': None},
    {'name': 'StyblinskiTang', 'dim': 5, 'int': None, 'res': None},
    {'name': 'Trid', 'dim': 6, 'int': None, 'res': None},
    {'name': 'Tripod', 'dim': 2, 'int': None, 'res': None},
    {'name': 'Weierstrass', 'dim': 3, 'int': None, 'res': None},
    {'name': 'Xor', 'dim': 9, 'int': None, 'res': None},
    {'name': 'YaoLiu', 'dim': 5, 'int': None, 'res': None},
]
