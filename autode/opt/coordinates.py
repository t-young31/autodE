import numpy as np
from abc import ABC, abstractmethod
from autode.units import (ang, nm, pm, m)
from autode.values import ValueArray


class _OptCoordinates(ValueArray, ABC):
    """Coordinates used to perform optimisations"""

    implemented_units = [ang, nm, pm, m]

    @abstractmethod
    def __repr__(self):
        """Representation of these coordinates"""

    def __new__(cls, input_array, units) -> '_OptCoordinates':
        """New instance of these coordinates"""

        arr = super().__new__(cls, input_array, units)
        arr._g, arr._h = None, None
        arr.B = None               # Wilson B matrix
        arr.B_T_inv = None         # Generalised inverse of B

        return arr

    def __array_finalize__(self, obj: '_OptCoordinates') -> None:
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

    @abstractmethod
    def to(self, *args, **kwargs):
        """Transformation between these coordinates and another type"""
