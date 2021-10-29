import numpy as np
from copy import deepcopy
from autode.species import Molecule
from autode.wrappers.base import Method
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.line_search import (LineSearchOptimiser,
                                    ArmijoLineSearch)


def quadratic(x, y):
    return x**2 + y**2, np.array([2*x, 2*y])


class TestSDLineSearch(LineSearchOptimiser):
    """Line search for E = x^2 + y^2"""

    __test__ = False

    def __init__(self,
                 init_step_size=0.1,
                 energy_grad_func=quadratic,
                 ):
        super().__init__(maxiter=100)

        self.energy_grad_func = energy_grad_func
        self.alpha = init_step_size

    @property
    def converged(self) -> bool:
        """Simple convergence criteria"""
        return self._species.energy is not None and self._species.energy < 0.001

    def _log_convergence(self) -> None:
        pass  # print(self._e_prev, self._species.energy)

    def _initialise_coordinates(self) -> None:
        self._coords = CartesianCoordinates(np.array([0.1, 0.2]))

    def _step(self) -> None:
        self._coords += self.alpha * self.p

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._species.energy, self._coords.g = self.energy_grad_func(x, y)


class TestArmijoLineSearch(ArmijoLineSearch):

    __test__ = False

    def __init__(self, init_step_size=1.0, energy_grad_func=quadratic):
        super().__init__(maxiter=10,
                         alpha_init=init_step_size)

        self.energy_grad_func = energy_grad_func

    def _initialise_coordinates(self) -> None:
        self._coords = CartesianCoordinates(np.array([-0.8, 1.0]))

    def _update_gradient_and_energy(self) -> None:
        return TestSDLineSearch._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        pass  # print(self._e_prev, self._species.energy)


def test_simple_line_search():

    blank_mol = Molecule(name='blank')
    blank_method = Method()

    optimiser = TestSDLineSearch()
    optimiser.run(blank_mol, method=blank_method)
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]))


def test_armijo_line_search_default():

    optimiser = TestArmijoLineSearch()
    assert not optimiser.converged

    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that point with 0 energy
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]))
    assert np.isclose(optimiser._species.energy, 0.0)


def test_armijo_line_search_diff_step_sizes():

    # using different step sizes should also converge
    for init_step_size in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
        optimiser = TestArmijoLineSearch(init_step_size=init_step_size)
        optimiser.run(Molecule(name='blank'), method=Method())

        assert optimiser.converged


def test_armijo_line_search_complex_func():

    def energy_grad(x, y):
        energy = 10*(y-x**2)**2 + (x-1)**2
        gradient = np.array([2*(20*x**3 - 20*x*y + x -1),
                             20*(y-x**2)])

        return energy, gradient

    optimiser = TestArmijoLineSearch(energy_grad_func=energy_grad,
                                     init_step_size=0.1)
    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert optimiser._species.energy < optimiser._init_e

