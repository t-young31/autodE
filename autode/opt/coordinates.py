import numpy as np
from abc import ABC, abstractmethod
from autode.units import (ang, nm, pm, m)
from autode.values import ValueArray


class OptCoordinates(ValueArray, ABC):
    """Coordinates used to perform optimisations"""

    implemented_units = [ang, nm, pm, m]

    @abstractmethod
    def __repr__(self):
        """Representation of these coordinates"""

    def __new__(cls, input_array, units) -> 'OptCoordinates':
        """New instance of these coordinates"""

        arr = super().__new__(cls, input_array, units)
        arr._g, arr._h = None, None
        arr.B = None               # Wilson B matrix
        arr.B_T_inv = None         # Generalised inverse of B

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        for attr in ('units', '_g', '_h', 'B', 'B_T_inv'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return None

    @property
    def g(self) -> np.ndarray:
        """Gradient of the energy: {dE/dx_i}"""
        return self._g

    @property
    def h(self) -> np.ndarray:
        """Second derivatives of the energy: {d^2E/dx_idx_j^2}"""
        return self._h

    @g.setter
    def g(self, value: np.ndarray):
        """Set the gradient of the energy"""
        self._g = value

    @h.setter
    def h(self, value: np.ndarray):
        """Set the second derivatives of the energy"""
        self._h = value

    @property
    @abstractmethod
    def type_str(self) -> str:
        """Type string of these coordinates. Should be compatible with to()"""

    @abstractmethod
    def to(self, *args, **kwargs):
        """Transformation between these coordinates and another type"""

    def __setitem__(self, key, value):
        """
        Set an item or slice in these coordinates. Clears the current
        gradient and Hessian as well as clearing setting the coordinates.
        Does NOT check if the current value is close to the current, thus
        the gradient and hessian shouldn't be cleared.
        """

        self.clear_gradient_and_hessian()
        return super().__setitem__(key, value)

    @abstractmethod
    def _iadd(self, value: np.ndarray) -> 'OptCoordinates':
        """Inplace addition of some coordinates"""

    def __iadd__(self, other: np.ndarray):
        """
        Inplace addition of another set of coordinates. Clears the current
        gradient vector and Hessian matrix.

        Arguments:
            other (np.ndarray): Array to add to the coordinates

        Returns:
            (autode.opt.coordinates.OptCoordinates): Shifted coordinates
        """
        self._iadd(other)
        self.clear_gradient_and_hessian()
        return self

    def __isub__(self, other: np.ndarray):
        """
        Inplace subtraction of another set of coordinates. Clears the current
        gradient vector and Hessian matrix.

        Arguments:
            other (np.ndarray): Array to subtract from the coordinates

        Returns:
            (autode.opt.coordinates.OptCoordinates): Shifted coordinates
        """
        return self.__iadd__(-other)

    def clear_gradient_and_hessian(self) -> None:
        """
        Clear the gradient and Hessian for these coordinates. Called if the
        coordinates have been perturbed, making these derivatives not
        accurate any more for the new coordinates
        """
        self._g, self._h = None, None
        return None
