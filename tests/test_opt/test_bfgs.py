"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import numpy as np
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.bfgs import BFGSOptimiser
from autode.opt.optimisers.line_search import NullLineSearch
from autode.species import Molecule
from autode.wrappers.base import Method
from .optimiers import TestBFGSOptimiser


def test_simple_quadratic_opt():

    optimiser = TestBFGSOptimiser()
    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged


def test_inv_hessian_update():

    coords = CartesianCoordinates(np.array([0.4, 0.2]))

    # Initial imperfect guess of the Hessian matrix for E = x^2 + y^2
    init_h = np.array([[1.0, 0.1],
                       [0.1, 1.0]])

    coords.h = init_h.copy()
    h_inv_true = np.array([[0.5, 0.0],
                           [0.0, 0.5]])

    optimiser = TestBFGSOptimiser(coords=coords)
    optimiser._species, optimiser._method = Molecule(name='blank'), Method()

    optimiser._update_gradient_and_energy()
    optimiser._step()

    # Should take the inverse of the guess after the first step
    assert np.allclose(optimiser._history[-1].h_inv,
                       np.linalg.inv(init_h))

    # Then for the new set of coordinates generate a better guess of the
    # inverse Hessian once the gradient has been updated
    optimiser._update_gradient_and_energy()
    optimiser._update_h_inv()

    assert (np.linalg.norm(optimiser._coords.h_inv - h_inv_true)
            < np.linalg.norm(np.linalg.inv(init_h) - h_inv_true))

    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged

    # By the end of the optimisation the Hessian should be pretty good
    assert np.allclose(optimiser._coords.h_inv,
                       h_inv_true,
                       atol=1E-2)


class TestBFGSOptimiser2D(BFGSOptimiser):
    """Simple 2D optimiser using a BFGS update step"""

    __test__ = False

    def __init__(self, e_func, g_func,
                 maxiter=100, etol=1E-4, gtol=1E-3, coords=None):
        super().__init__(maxiter=maxiter, line_search_type=NullLineSearch,
                         etol=etol, gtol=gtol, step_size=0.4, coords=coords)

        self.e_func = e_func
        self.g_func = g_func

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e = self.e_func(x, y)
        self._coords.g = self.g_func(x, y)

    def _initialise_run(self) -> None:
        init_arr = np.array([1.0, 1.0])
        self._coords = CartesianCoordinates(init_arr)

        # Guess the Hessian as the identity matrix
        self._coords.h = np.eye(len(self._coords))
        self._update_gradient_and_energy()


def _test_quadratic_opt():
    """E = x^2 + y^2 + xy/10,  ∇E  = (2x + 0.1y, 2y + 0.1x)"""

    optimiser = TestBFGSOptimiser2D(e_func=lambda x, y: x**2 + y**2 + x*y/10.0,
                                    g_func=lambda x, y: np.array([2.0*x + 0.1*y, 2.0*y + 0.1*x]))
    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged
    assert optimiser._coords.e < 1E-3
    assert np.allclose(optimiser._coords,
                       np.zeros(2),       # Minimum is at (0, 0)
                       atol=1E-3)


def test_gaussian_well_opt():
    """E = -exp(-(x^2 + y^2)), ∇E  = (2xE, 2yE)"""

    def energy(x, y):
        return -np.exp(-(x**2 + y**2))

    def grad(x, y):
        e = np.exp(-(x**2 + y**2))
        return np.array([2.0*x*e, 2.0*y*e])

    def hessian(x, y):
        h_xy = -4.0*x*y*np.exp(-x**2 - y**2)
        return np.array([[2*(1-2.*x**2)*np.exp(-x**2 - y**2), h_xy],
                         [h_xy, 2*(1-2.*y**2)*np.exp(-x**2 - y**2)]])

    optimiser = TestBFGSOptimiser2D(e_func=energy, g_func=grad)
    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert optimiser._coords.e < 1E-3
    assert np.allclose(optimiser._coords,
                       np.zeros(2),       # Minimum is at (0, 0)
                       atol=1E-3)
