"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import numpy as np
from autode.species import Molecule
from autode.wrappers.base import Method
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

    def __init__(self, init_step_size=1.0):
        super().__init__(maxiter=100, gtol=0.01, etol=0.001)

        self.alpha = init_step_size

    def _initialise_coords(self) -> None:
        init_arr = np.array([0.1, 0.2])
        self._coords = CartesianCoordinates(init_arr)

        # Guess the Hessian as the identity matrix
        self._coords.h = np.eye(len(self._coords))

    def _step(self) -> None:

        # for h = I then this is just a steepest decent step
        p = np.matmul(np.linalg.inv(self._coords.h), -self._coords.g)

        LineSearch.optimise(self._species, self._method,
                            coords=self._coords,
                            direciton=p)

        raise NotImplementedError

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.g = np.array([2*x, 2*y])
        self._species.energy = x**2 + y**2



def test_opt():

    blank_mol = Molecule(name='blank')
    blank_method = Method()

    optimiser = TestBFGSOptimiser()
    optimiser.run(blank_mol, method=blank_method)

    print(optimiser.converged)


