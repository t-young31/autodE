import numpy as np
from autode.species import Molecule
from autode.wrappers.base import Method
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.line_search import LineSearchOptimiser


class TestLineSearch(LineSearchOptimiser):
    """Line search for E = x^2 + y^2"""

    __test__ = False

    def __init__(self, init_step_size=0.1):
        super().__init__(maxiter=100)

        self.alpha = init_step_size

    @classmethod
    def optimise(cls, species, method, **kwargs):
        raise NotImplementedError

    @property
    def converged(self) -> bool:
        """Simple convergence criteria"""
        return self._species.energy is not None and self._species.energy < 0.001

    def _log_convergence(self) -> None:
        pass  # print(self._e_prev, self._species.energy)

    def _initialise_coordinates(self) -> None:
        pass

    def _initialise_run(self) -> None:
        self._coords = CartesianCoordinates(np.array([0.1, 0.2]))
        self._update_gradient_and_energy()
        self.p = -self._coords.g

    def _step(self) -> None:
        self._coords += self.alpha * self.p

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.g = np.array([2*x, 2*y])
        self._species.energy = x**2 + y**2


def test_simple_line_search():

    blank_mol = Molecule(name='blank')
    blank_method = Method()

    optimiser = TestLineSearch()
    optimiser.run(blank_mol, method=blank_method)
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]))

