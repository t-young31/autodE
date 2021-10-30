import numpy as np
from autode.species import Molecule
from autode.wrappers.base import Method
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.line_search import ArmijoLineSearch
from .optimiers import TestSDLineSearch


def quadratic(x, y):
    return x**2 + y**2, np.array([2*x, 2*y])


class TestArmijoLineSearch(ArmijoLineSearch):

    __test__ = False

    def __init__(self, init_step_size=1.0, energy_grad_func=quadratic):
        super().__init__(maxiter=10,
                         init_alpha=init_step_size)

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


def _test_armijo_line_search_default():

    optimiser = TestArmijoLineSearch()
    assert not optimiser.converged

    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that point with 0 energy
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]))
    assert np.isclose(optimiser._coords.e, 0.0)


def _test_armijo_line_search_diff_step_sizes():

    # using different step sizes should also converge
    for init_step_size in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
        optimiser = TestArmijoLineSearch(init_step_size=init_step_size)
        optimiser.run(Molecule(name='blank'), method=Method())

        assert optimiser.converged


def _test_armijo_line_search_complex_func():

    def energy_grad(x, y):
        energy = 10*(y-x**2)**2 + (x-1)**2
        gradient = np.array([2*(20*x**3 - 20*x*y + x -1),
                             20*(y-x**2)])

        return energy, gradient

    optimiser = TestArmijoLineSearch(energy_grad_func=energy_grad,
                                     init_step_size=0.1)
    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert optimiser._coords.e < optimiser._init_coords.e
