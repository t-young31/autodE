from abc import ABC
from copy import deepcopy
import numpy as np
from typing import Optional
from autode.config import Config
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.optimisers import Optimiser


class LineSearchOptimiser(Optimiser, ABC):
    """Optimiser for a 1D line search in a direction"""

    def __init__(self,
                 maxiter:   int,
                 direction: np.ndarray):
        """
        Line search optimiser

        Arguments:
            direction (np.ndarray): Shape must be broadcastable to the
                                    coordinates
        """
        super().__init__(maxiter=maxiter)

        self.p: Optional[np.ndarray] = direction
        self._init_coords = deepcopy(self._coords)

    @classmethod
    def optimise(cls,
                 species:   'autode.species.Species',
                 method:    'autode.wrappers.base.Method',
                 coords:     Optional['autode.opt.coordinates.OptCoordinates'] = None,
                 direction:  Optional[np.ndarray] = None,
                 maxiter:    int = 20,
                 n_cores:    Optional[int] = None
                 ) -> None:
        """
        Optimise a species along a single direction
        """
        optimiser = cls(maxiter=maxiter, direction=direction)

        optimiser.run(species, method,
                      n_cores=Config.n_cores if n_cores is None else n_cores)

        return None


class CartesianArmijoLineSearch(LineSearchOptimiser):

    def __init__(self,
                 maxiter:    int,
                 direction:  np.ndarray,
                 beta:       float = 0.1,
                 tau:        float = 0.5,
                 alpha_init: float = 1.0):
        """



        """
        super().__init__(maxiter=maxiter, direction=direction)

        self._init_e: Optional['autode.values.PotentialEnergy'] = None

        self.beta = float(beta)
        self.tau = float(tau)
        self.alpha = float(alpha_init)

    def _step(self) -> None:

        self.alpha *= self.tau
        self._coords += self.alpha * self.p

        return None

    def _initialise_coords(self) -> None:
        """Initialised cartesian coordinates, if they're not already defined"""
        if self._coords is None:
            self._coords = CartesianCoordinates(self._species.coordinates)

        self._init_coords = self._coords.copy()
        return None

    @property
    def converged(self) -> bool:

        term_2 = self.alpha * self.beta * np.dot(self._init_coords.g, self.p)
        return self._species.energy < self._init_e + term_2

    def _log_convergence(self) -> None:
        pass
