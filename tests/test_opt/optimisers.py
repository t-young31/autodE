import numpy as np
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.optimisers import Optimiser


class TestBFGSOptimiser(Optimiser):
    """Simple 2D optimiser using a BFGS update step, where the objective
    funtion is::

        E = x^2 + y^2 + xy/10

    so

        ∇E  = (2x + 0.1y, 2y + 0.1x)

    and
            (  2   0.1  )
        H = (
            (  0.1   2  )
    """

    __test__ = False

    def __init__(self, init_step_size=1.0):
        super().__init__(maxiter=100)

        self.alpha = init_step_size

    @classmethod
    def optimise(cls, species, method, **kwargs):
        raise NotImplementedError

    @property
    def converged(self) -> bool:
        """Simple convergence criteria"""

        return (self._species.energy is not None
                and abs(self._species.energy - self._e_prev) < 0.0001)

    def _log_convergence(self) -> None:
        pass

    def _initialise_coords(self) -> None:
        init_arr = np.array([0.1, 0.2])
        self._coords = CartesianCoordinates(init_arr)

        # Guess the Hessian as the identity matrix
        self._coords.h = np.eye(len(self._coords))

    def _step(self) -> None:

        # for h = I then this is just a steepest decent step
        p = np.matmul(np.linalg.inv(self._coords.h), -self._coords.g)

        raise NotImplementedError

        LineSearch.optimise(self._species, self._method,
                            coords=self._coords,
                            direciton=p)

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.g = np.array([2*x, 2*y])
        self._species.energy = x**2 + y**2
