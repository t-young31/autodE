"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import pytest
import numpy as np
from autode.species import Molecule
from autode.wrappers.base import Method
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.bfgs import BFGSOptimiser


def quadratic(x, y):
    """Energy, gradient and Hessian of a simple quadratic potential in 2D"""

    energy = x**2 + y**2
    gradient = np.array([2*x, 2*y])
    hessian = 2.0 * np.eye(2)

    return energy, gradient, hessian


class TestBFGSOptimiser(BFGSOptimiser):
    """Simple 2D optimiser using a BFGS update step, where the objective
    function is::

        E = x^2 + y^2 + xy/10

    so

        ∇E  = (2x + 0.1y, 2y + 0.1x)

    and
            (  2   0.1  )
        H = (
            (  0.1   2  )
    """

    __test__ = False

    def _initialise_run(self) -> None:
        init_arr = np.array([0.1, 0.2])
        self._coords = CartesianCoordinates(init_arr)

        # Guess the Hessian as the identity matrix
        self._coords.h = np.eye(len(self._coords))

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.g = np.array([2*x, 2*y])
        self._species.energy = x**2 + y**2


def test_opt():

    blank_mol = Molecule(name='blank')
    blank_method = Method()

    optimiser = TestBFGSOptimiser()
    #optimiser.run(blank_mol, method=blank_method)
    # print(optimiser.converged)

