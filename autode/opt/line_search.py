import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional
from autode.log import logger
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.optimisers import Optimiser


class LineSearchOptimiser(Optimiser, ABC):
    """Optimiser for a 1D line search in a direction"""

    def __init__(self,
                 maxiter:   int,
                 direction: Optional[np.ndarray] = None,
                 coords:    Optional['autode.opt.coordinates.OptCoordinates'] = None):
        """
        Line search optimiser

        -----------------------------------------------------------------------
        Keyword Arguments:
            direction (np.ndarray | None): Direction which to move the
                      coordinates. Shape must be broadcastable to the
                      coordinates. If None then will guess a sensible direction

            coords (autode.opt.coordinates.OptCoordinates | None): Initial
                    coordinates. If None then they will be initialised from the
                    species at runtime
        """
        super().__init__(maxiter=maxiter, coords=coords)

        self.p: Optional[np.ndarray] = direction

        # Saved copy of the initial coordinates
        self._init_coords = deepcopy(self._coords)

    @classmethod
    def optimise(cls,
                 species:   'autode.species.Species',
                 method:    'autode.wrappers.base.Method',
                 coords:     Optional['autode.opt.coordinates.OptCoordinates'] = None,
                 direction:  Optional[np.ndarray] = None,
                 maxiter:    int = 5,
                 n_cores:    Optional[int] = None
                 ) -> None:
        """
        Optimise a species along a single direction. If the direction is
        unspecified then guess the direction as just the steepest decent
        direction.
        """

        optimiser = cls(maxiter=maxiter, direction=direction, coords=coords)
        optimiser.run(species, method, n_cores=n_cores)

        return None

    @abstractmethod
    def _initialise_coordinates(self) -> None:
        """Initialise the coordinates, if not already specified"""

    def _initialise_run(self) -> None:
        """
        Initialise running the line search. Allows for both the coordinates
        and search direction to be unspecified.
        """
        if self._coords is None:
            self._initialise_coordinates()

        if self.p is None:
            logger.warning('Line search optimiser was initialised without a '
                           'search direction. Using steepest decent direction')
            if self._coords.g is None:
                self._update_gradient_and_energy()

            self.p = -self._coords.g

        self._init_coords = deepcopy(self._coords)
        return None


class ArmijoLineSearch(LineSearchOptimiser):

    def __init__(self,
                 maxiter:    int,
                 direction:  Optional[np.ndarray] = None,
                 beta:       float = 0.1,
                 tau:        float = 0.5,
                 alpha_init: float = 1.0):
        """
        Backtracking line search by Armijo. Reduces the step size iteratively
        until the convergence condition is satisfied

        [1] L. Armijo. Pacific J. Math. 16, 1966, 1. DOI:10.2140/pjm.1966.16.1.

        ----------------------------------------------------------------------
        Arguments:
            maxiter (int): Maximum number of iteration to perform. Should be
                           small O(1)

        Keyword Arguments:
            direction (np.ndarray): Direction to search along

            beta (float): β parameter in the line search

            tau (float): τ parameter. Multiplicative factor when reducing the
                         step size.

            alpha_init (float): α_0 parameter. Initial value of the step size.
        """
        super().__init__(maxiter=maxiter, direction=direction)

        # Initial energy, used for determining convergence of the line search
        self._init_e: Optional['autode.values.PotentialEnergy'] = None

        self.beta = float(beta)
        self.tau = float(tau)
        self.alpha = float(alpha_init)

    def _step(self) -> None:
        """Take a step in the line search"""

        self.alpha *= self.tau
        self._coords = self._init_coords + self.alpha * self.p

        return None

    def _initialise_coordinates(self) -> None:
        """Initialise the coordinates if they are not defined already.
        Defaults to CartesianCoordinates"""
        self._coords = CartesianCoordinates(self._species.coordinates)
        return None

    @property
    def converged(self) -> bool:
        r"""
        Is the line search converged? Defined by the Armijo condition

        .. math::

            f(x + \alpha^{(l)} p_k) \le f(x) + \alpha^{(l)} \beta g\cdot p

        where α is the step size at the current iteration (denoted by l) and
        β is a variable parameter. The search direction p, gradient are defined
        for the initial point only.

        Returns:
            (bool): If the search is converged
        """
        if self._init_coords is None or self._init_coords.g is None:
            logger.warning('No convergence without defined coordinates '
                           'or gradients')
            return False

        # Ensure the initial energy value is set base on the current energy
        if self._init_e is None:
            self._init_e = self._species.energy

        term_2 = self.alpha * self.beta * np.dot(self._init_coords.g, self.p)
        return self._species.energy < self._init_e + term_2

    def _log_convergence(self) -> None:
        pass
