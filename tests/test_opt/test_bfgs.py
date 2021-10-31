"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import numpy as np
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.species import Molecule
from autode.wrappers.base import Method
from .optimiers import TestBFGSOptimiser


def test_opt():

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
