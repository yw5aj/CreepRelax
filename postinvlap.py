from sympy import diff, S, lambdify
from sympy.abc import s, t
from math import factorial
import unittest
import numpy as np


def postinvlap(fs, q=10000, use_lambdify=False):
    """
    Compute the numerical inverse Laplace transform using Post method.

    Parameters
    ----------
    fs : sympy expression
        Analytical form of Laplace transform to be inversed.
        Must be in terms of `s`.
    qmax : int
        Maximum derivative order in the Post method.

    Returns
    -------
    ft : sympy expression or function
        If `use_lambdify` is `False`, then return a sympy expression in terms
        of t;
        If `use_lambdify` is `True`, then return a function ft(t), where t may
        be a float or numpy.ndarray.
    """
    dfsds = diff(fs, s, q)
    ft = S(-1) ** q / factorial(q) * (q / t) ** (q + 1) * dfsds.subs(s, q / t)
    if use_lambdify:
        ft = lambdify(t, ft, 'numpy')
    return ft


class TestPostInvLap(unittest.TestCase):

    def test_s2(self):
        fs = 1 / s ** 2
        ft = postinvlap(fs, q=5000, use_lambdify=True)
        testnum = np.random.rand() * 100
        self.assertTrue(np.abs((ft(testnum) - testnum) / testnum) < 1e-3)


if __name__ == '__main__':
    unittest.main()
