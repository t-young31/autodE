import autode as ade
import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    """Abstract base class for an autodE optimiser"""

    def run(self, max_iterations: int):
        """
        Optimise to convergence

        Arguments:
            max_iterations (int): Maximum number of iterations to perform

        Returns:
            (None):
        """

    def _get_gradient(self, coordinates):
        """
        Calculate the gradient for a set of coordinates

        Args:
            coordinates (np.ndarray): Cartesian coordinates with shape
                        (n_atoms, 3) or (3*n_atoms,)

        Returns:
            (np.ndarray): Gradient shape = (3*n_atoms,)  (Ha / Å)

        Raises:
            (autode.exceptions.CalculationException): If a calculation fails
        """
        self.species.coordinates = coordinates

        # Run the calculation and remove all the files, if it's successful
        calc = ade.Calculation(name='tmp_grad',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.grad,
                               n_cores=ade.Config.n_cores)
        calc.run()

        grad = calc.get_gradients()
        calc.clean_up(force=True, everything=True)

        return grad.flatten()

    def __init__(self, species, method):
        """
        Optimiser initialised from a species and a method used to perform
        gradient and/or Hessian calculations

        Arguments:
            species (autode.species.Species):

            method (autode.wrappers.base.ElectronicStructureMethod):
        """
        self.species = species
        self.method = method


class Coordinates(ABC):
    """Abstract base class for some autodE coordinates"""

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, coords):
        """
        Set coordinates from another set of coordinates which may be Cartesian
        or a different coordinate system. As internal coordinate systems should
        have 3N-6 components checking the shape should be sufficient(?!)
        """

        if len(coords.flatten()) == len(self.x):
            self.x = coords
            self._s = self._cart_to_s(coords)

        elif len(coords.flatten()) == len(self.s):
            self.x = self._s_to_cart(coords)
            self._s = coords

        else:
            raise ValueError("Couldn't set the coordinates: not a valid shape")

    @abstractmethod
    def _s_to_cart(self, x):
        """Convert cartesian coordinates to this coordinate system"""

    @abstractmethod
    def _cart_to_s(self, x):
        """Convert cartesian coordinates to this coordinate system"""

    @property
    def g(self):
        """Gradient in this coordinate system"""
        return self._g_s

    @g.setter
    @abstractmethod
    def g(self, g_x):
        """Set the gradient in a coordinate system (_g_s) from Cartesian"""

    @property
    def H(self):
        """Hessian in this coordinate system"""
        return self._H_s

    @H.setter
    @abstractmethod
    def H(self, H_x):
        """Set the Hessian in a coordinate system (_H_s) from Cartesian"""

    def __init__(self, cartesian_coordinates: np.ndarray):
        """
        Superclass for internal or cartesian coordinates

        Arguments:
            cartesian_coordinates (np.ndarray):
        """
        self.x = cartesian_coordinates.flatten()    # Cartesians (Å)

        self.B = None                               # Wilson B matrix
        self.B_T_inv = None                         # Generalised inverse of B

        self._s = self._cart_to_s(self.x)           # Coordinates (Å)
        self._g_s = None                            # Gradient (Ha / Å)
        self._H_s = None                            # Hessian (Ha / Å^2)


class CartesianCoordinates(Coordinates):

    def _cart_to_s(self, x):
        """Nothing to do for x -> s"""
        return x

    def _s_to_cart(self, s):
        """Nothing to do for s -> x"""
        return s

    @Coordinates.g.setter
    def g(self, g_x):
        """Nothing to do to convert Cartesian gradient"""
        self._g_s = g_x

    @Coordinates.H.setter
    def H(self, H_x):
        """Nothing to do to convert Cartesian Hessian"""
        self._H_s = H_x

    def __int__(self, cartesian_coordinates):
        """Cartesian coordinates"""
        super().__init__(cartesian_coordinates)


class Primitive(ABC):
    """Primitive internal coordinate"""

    @abstractmethod
    def derivative(self, i, k, x):
        """
        Calculate the derivative with respect to a cartesian coordinate

            dq   |
        ---------|
        dx_(i, k)|_x=x0

        Argument:
            i (int): Cartesian index to take the derivative with respect to;
                     0-N for N atoms

            k (int): Cartesian component (x, y, z) to take the derivative with
                     respect to; 0-3

            x (np.ndarray): Cartesian coordinates shape = (N, 3) for N atoms

        Returns:
            (float)
        """
