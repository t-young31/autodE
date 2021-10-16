import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
from autode.log import logger
from autode.config import Config
from autode.calculation import Calculation
from autode.values import GradientNorm, PotentialEnergy
from autode.opt.cartesian import CartesianCoordinates


class Optimiser(ABC):
    """Abstract base class for an optimiser"""

    def __init__(self,
                 maxiter: int,
                 gtol:    GradientNorm,
                 etol:    PotentialEnergy):
        """
        Geometry optimiser. Signature follows that in scipy.minimize so
        species and method are keyword arguments. Converged when both energy
        and gradient criteria are met.

        ----------------------------------------------------------------------
        Arguments:
            maxiter (int): Maximum number of iterations to perform

            gtol (autode.values.GradientNorm): Tolerance on RMS(|∇E|)

            etol (autode.values.PotentialEnergy): Tolerance on |E_i+1 - E_i|
        """
        self.iteration = 0
        self._maxiter, self._gtol, self._etol = None, None, None

        self.maxiter = maxiter
        self.gtol = gtol
        self.etol = etol

        self._ncores:  int = Config.n_cores

        self._coords:  Optional['autode.opt.coordinates.OptCoordinates'] = None
        self._species: Optional['autode.species.Species'] = None
        self._method:  Optional['autode.wrappers.base.Method'] = None

        # Previous energy, used to check convergence of the energy
        self._e_prev = PotentialEnergy(np.inf, units='Ha')

    @property
    def maxiter(self) -> int:
        """
        Maximum number of iterations (gradient evaluations) to perform

        Returns:
            (int): Maxiter
        """
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value: int):
        """Set the Maximum number of iterations to perform"""
        if int(value) <= 0:
            raise ValueError('An optimiser must be able to run at least one '
                             f'step, but tried to set maxiter = {value}')

        self._maxiter = int(value)

    @property
    def gtol(self) -> GradientNorm:
        """
        Gradient tolerance on |∇E| i.e. the root mean square of each component

        Returns:
            (autode.values.GradientNorm):
        """
        return self._gtol

    @gtol.setter
    def gtol(self, value: Union[int, float, GradientNorm]):
        """Set the gradient tolerance"""

        if float(value) <= 0:
            raise ValueError('Tolerance on the gradient (||∇E||) must be '
                             f'positive. Had: gtol={value}')

        self._gtol = GradientNorm(value)

    @property
    def etol(self) -> PotentialEnergy:
        """
        Energy tolerance between two consecutive steps of the optimisation

        Returns:
            (autode.values.PotentialEnergy): Energy tolerance
        """
        return self._etol

    @etol.setter
    def etol(self, value: Union[int, float, PotentialEnergy]):
        """Set the energy tolerance"""
        if float(value) <= 0:
            raise ValueError('Tolerance on the energy change is absolute so '
                             f'must be positive. Had etol = {value}')

        self._etol = PotentialEnergy(value)

    @classmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 maxiter: int = 500,
                 gtol:    Union[float, GradientNorm] = GradientNorm(1E-3, units='Ha Å-1'),
                 etol:    Union[float, PotentialEnergy] = PotentialEnergy(1E-4, units='Ha'),
                 n_cores: Optional[int] = None,
                 **kwargs
                 ) -> None:
        """
        Convenience function for constructing and running an optimiser

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            method (autode.methods.Method):

        Keyword Arguments
            maxiter (int): Maximum number of iteration to perform

            gtol (float | autode.values.GradientNorm): Tolerance on RMS(|∇E|)
                 i.e. the root mean square of the gradient components. If
                 a float then assume units of Ha Å^-1

            etol (float | autode.values.PotentialEnergy): Tolerance on |∆E|
                 between two consecutive iterations of the optimiser

            kwargs (Any): Additional keyword arguments to pass on
        """

        optimiser = cls(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)
        optimiser.run(species, method, n_cores=n_cores)

        return None

    def run(self,
            species: 'autode.species.Species',
            method:  'autode.wrappers.base.Method',
            n_cores: Optional[int] = None
            ) -> None:
        """
        Run the optimiser. Updates species.atoms and species.energy

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species): Species to optimise, if None
                    then use the species this optimiser was initalised with

            method (autode.methods.Method): Method to use. Calculations will
                   use method.keywords.grad for gradient calculations

        Keyword Arguments:
            n_cores (int | None): Number of cores to use for the gradient
                        evaluations. If None then use autode.Config.n_cores
        """
        self._method = method if method is not None else self._method
        self._ncores = n_cores if n_cores is not None else Config.n_cores
        self._species = species if species is not None else self._species

        self._check_species_and_method()
        self._initialise_coords()

        logger.info(f'Using {self._method} to optimise {self._species.name} '
                    f'with {self._ncores} cores using {self.maxiter} max '
                    f'iterations')
        logger.info('Iteration\t|∆E| / \\kcal mol-1 \t||∇E|| / Ha Å-1')

        while not self.converged:

            self._update_gradient_and_energy()   # Updates self._coords.g
            self._step()              # Updates self._species.coordinates

            self._log_convergence()
            self.iteration += 1
            self._e_prev = self._species.energy

            if self.iteration >= self.maxiter:
                logger.warning(f'Reached the maximum number of iterations '
                               f'*{self.maxiter}*. Did not converge')
                break

        logger.info(f'Converged: {self.converged}, in {self.iteration} cycles')
        self.iteration = 0
        return None

    @property
    def converged(self) -> bool:
        """
        Is this optimisation converged? Must be converged based on both energy
        and gradient tolerance.

        Returns:
            (bool): Converged?
        """
        return self.abs_delta_e < self.etol and self.g_norm < self.gtol

    @property
    def abs_delta_e(self) -> PotentialEnergy:
        """
        Calculate the absolute energy difference

        .. math::
            |∆E| = |E_i - E_{i-1}|   for a step i

        Returns:
            (autode.values.PotentialEnergy): Energy difference. Infinity if
                                  an energy difference cannot be calculated
        """

        if not hasattr(self._species, 'energy'):
            logger.error('self._species did not have an energy attribute. '
                         'Returning |∆E| = ∞')
            return PotentialEnergy(np.inf)

        if self._species.energy is None:
            return PotentialEnergy(np.inf)

        # NOTE: no abs() call to preserve PotentialEnergy type
        if self._species.energy > self._e_prev:
            return self._species.energy - self._e_prev
        else:
            return self._e_prev - self._species.energy

    @property
    def g_norm(self) -> GradientNorm:
        """
        Calculate ||∇E|| based on the current Cartesian gradient.

        Returns:
            (autode.values.GradientNorm): Gradient norm. Infinity if the
                                          gradient is not defined
        """
        if self._coords is None:
            logger.warning('Had no coordinates - cannot determine ||∇E||')
            return GradientNorm(np.inf)

        if self._coords.to('cart').g is None:
            return GradientNorm(np.inf)

        return GradientNorm(np.linalg.norm(self._coords.to('cart').g))

    def _log_convergence(self) -> None:
        """Log the convergence of the energy """
        logger.info(f'{self.iteration}\t'
                    f'{self.abs_delta_e.to("kcal mol-1"):.3f}\t'
                    f'{self.g_norm:.5f}')

        return None

    def _update_gradient_and_energy(self) -> None:
        """
        Update the gradient of the energy with respect to the coordinates

        Raises:
            (autode.exceptions.CalculationException):
        """
        # Calculations need to be performed in cartesian coordinates
        self._coords = self._coords.to('cart')
        self._species.coordinates = np.array(self._coords, copy=True)

        grad = Calculation(name=f'{self._species.name}_opt_{self.iteration}',
                           molecule=self._species,
                           method=self._method,
                           keywords=self._method.keywords.grad,
                           n_cores=self._ncores)
        grad.run()

        # Update the energy and gradient for the species
        self._species.energy = grad.get_energy()
        self._species.gradient = grad.get_gradients()
        grad.clean_up(force=True, everything=True)

        self._coords.g = self._species.gradient.flatten()

        return None

    def _check_species_and_method(self) -> None:
        """Check the internal species and method have the correct attributes"""

        if self._species is None or self._method is None:
            raise ValueError('Must have a species and a method to run an '
                             f'optimisation. Had: {self._species} and '
                             f'{self._method}')

        if not all(hasattr(self._species, attr) for attr in ('energy', 'name')):
            raise ValueError('Internal species required energy and name '
                             f'attributes but had {self._species}')

    @abstractmethod
    def _step(self) -> None:
        """
        Take a step with this optimiser. Should only act on self._coords
        using the gradient (self._coords.g) and hessians (self._coords.h)
        """

    @abstractmethod
    def _initialise_coords(self) -> None:
        """Initialise self._coords from self._species"""


class SteepestDecent(Optimiser, ABC):

    def __init__(self, maxiter, gtol, etol, step_size=0.2, **kwargs):
        """
        Steepest decent optimiser

        Arguments:
            step_size (float): Size of the step to take. Units of distance
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self.step_size = step_size


class CartesianSDOptimiser(SteepestDecent):

    def _initialise_coords(self) -> None:
        """
        Initialise a set of cartesian coordinates. As a species' coordinates
        are already Cartesian there is nothing special to do
        """
        self._coords = CartesianCoordinates(self._species.coordinates)

    def _step(self) -> None:
        """
        Take a steepest decent step::

        .. math::

            x_{i+1} = x_{i} - d \nabla E

        where d is the step size.
        """
        self._coords -= self.step_size * self._coords.g


class DIC_SD_Optimiser(SteepestDecent):

    def _initialise_coords(self) -> None:
        """Initialise the delocalised internal coordinates"""
        self._coords = CartesianCoordinates(self._species.coordinates).to('dic')

    def _step(self) -> None:
        """Take a step in DICs"""

        self._coords = self._coords.to('dic')
        self._coords -= self.step_size * self._coords.g
