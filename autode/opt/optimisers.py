import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
from autode.log import logger
from autode.values import GradientNorm, PotentialEnergy


class Optimiser(ABC):
    """Abstract base class for an optimiser"""

    def __init__(self, maxiter, gtol, etol, **kwargs):
        """
        Geometry optimiser. Signature follows that in scipy.minimize

        Arguments:

        """
        self.iteration = 0
        self.maxiter = maxiter

        self.gtol = GradientNorm(gtol)                     # Gradient tolerance
        self.etol = PotentialEnergy(etol)                  # Energy tolerance

        self._coords:  Optional['autode.opt.coordinates.OptCoordinates'] = None
        self._species: Optional['autode.species.Species'] = kwargs.get('species', None)
        self._method:  Optional['autode.wrappers.base.Method'] = kwargs.get('method', None)

        # Previous energy, used to check convergence of the energy
        self._e_prev = PotentialEnergy(np.inf, units='Ha')

    @classmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 maxiter: int = 500,
                 gtol:    Union[float, GradientNorm] = GradientNorm(1E-3, units='Ha Å-1'),
                 etol:    Union[float, PotentialEnergy] = PotentialEnergy(1E-4, units='Ha')
                 ) -> None:
        """
        Convenience function for constructing and running an optimiser

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            method (autode.methods.Method):

            maxiter (int): Maximum number of iteration to perform

            gtol (float | autode.values.GradientNorm): Tolerance on RMS(|∇E|)
                 i.e. the root mean square of the gradient components. If
                 a float then assume units of Ha Å^-1

            etol (float | autode.values.PotentialEnergy): Tolerance on |∆E|
                 between two consecutive iterations of the optimiser
        """

        optimiser = cls(maxiter=maxiter, gtol=gtol, etol=etol)
        optimiser.run(species, method)

        return None

    def run(self,
            species: Optional['autode.species.Species'],
            method:  Optional['autode.wrappers.base.Method']) -> None:
        """
        Run the optimiser. Updates species.atoms and species.energy

        ----------------------------------------------------------------------
        Keyword Arguments:
            species (autode.species.Species):

            method (autode.methods.Method):
        """
        self._method = method if method is not None else self._method
        logger.info(f'Using {self._method} to optimise')

        self._species = species if species is not None else self._species

        if self._species is None or self._method is None:
            raise ValueError('Must have a species and a method to run an '
                             f'optimisation. Had: {self._species} and '
                             f'{self._method}')

        logger.info(f'Optimising {self._species}')

        while not self.converged and self.iteration < self.maxiter:

            self._step()             # Updates self._species.coordinates
            self.iteration += 1

        return None

    def converged(self) -> bool:
        """
        Is this optimisation converged? Must be converged based on both energy
        and gradient tolerance.

        Returns:
            (bool): Converged?
        """
        if not hasattr(self._species, 'energy'):
            logger.error('Species did not have an energy attribute. Cannot '
                         'determine convergence. Assuming false')
            return False

        # Check energy tolerance |E - E_prev| < etol
        if abs(self._species.energy - self._e_prev) > self.etol:
            return False

        if not hasattr(self._coords, 'g'):
            logger.error('Optimiser coordinates did not have a gradient, thus '
                         'not converged')
            return False

        if self._coords.to('cart').g is None:
            logger.warning(f'Cartesian gradient for {self._species} was not '
                           f'defined, thus ildetermined convergence')
            return False

        # Check gradient tolerance: ||∇E|| < gtol
        if np.linalg.norm(self._coords.to('cart').g) > self.gtol:
            return False

        return True

    @abstractmethod
    def _step(self):
        """Take a step with this optimiser"""
