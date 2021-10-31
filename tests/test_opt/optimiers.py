import numpy as np
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.line_search import LineSearchOptimiser
from autode.opt.optimisers.bfgs import BFGSOptimiser


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

    def __init__(self, maxiter=100, etol=1E-4, gtol=1E-3, coords=None):
        super().__init__(maxiter=maxiter, line_search_type=TestSDLineSearch,
                         etol=etol, gtol=gtol, step_size=0.1, coords=coords)

    def _log_convergence(self) -> None:
        print(self._coords.e)

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords

        self._coords.e, self._coords.g = quadratic(x, y)[:2]

    def _initialise_run(self) -> None:
        init_arr = np.array([0.1, 0.2])
        self._coords = CartesianCoordinates(init_arr)

        # Guess the Hessian as the identity matrix
        self._coords.h = np.eye(len(self._coords))
        self._update_gradient_and_energy()


class TestSDLineSearch(LineSearchOptimiser):
    """Line search for E = x^2 + y^2"""

    __test__ = False

    def __init__(self,
                 init_alpha=0.1,
                 energy_grad_func=quadratic,
                 direction=None,
                 coords=None
                 ):
        super().__init__(maxiter=10, direction=direction, init_alpha=init_alpha
                         )

        self._coords = coords
        self.energy_grad_func = energy_grad_func

    @property
    def converged(self) -> bool:
        """Simple convergence criteria"""
        return self.iteration > 0 and self._coords.e is not None and self._coords.e < 0.01

    def _log_convergence(self) -> None:
        pass  # print(self._coords.e)

    def _initialise_coordinates(self) -> None:
        self._coords = CartesianCoordinates(np.array([1.1, 0.2]))

    def _step(self) -> None:
        self._coords += self.alpha * self.p

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e, self._coords.g = self.energy_grad_func(x, y)[:2]
