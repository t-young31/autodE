from abc import ABC
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.optimisers import NDOptimiser


class SteepestDecent(NDOptimiser, ABC):

    def __init__(self, maxiter, gtol, etol, step_size=0.2, **kwargs):
        """
        Steepest decent optimiser

        ----------------------------------------------------------------------
        Arguments:
            step_size (float): Size of the step to take. Units of distance

        See Also:

            :py:meth:`NDOptimiser <autode.opt.optimisers.NDOptimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self.step_size = step_size

    def _step(self) -> None:
        """
        Take a steepest decent step::

        .. math::

            x_{i+1} = x_{i} - d \nabla E

        where d is the step size.
        """
        self._coords -= self.step_size * self._coords.g


class CartesianSDOptimiser(SteepestDecent):
    """Steepest decent optimisation in Cartesian coordinates"""

    def _initialise_coords(self) -> None:
        """
        Initialise a set of cartesian coordinates. As a species' coordinates
        are already Cartesian there is nothing special to do
        """
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()


class DIC_SD_Optimiser(SteepestDecent):
    """Steepest decent optimisation in delocalised internal coordinates"""

    def _initialise_coords(self) -> None:
        """Initialise the delocalised internal coordinates"""
        self._coords = CartesianCoordinates(self._species.coordinates).to('dic')
        self._update_gradient_and_energy()
