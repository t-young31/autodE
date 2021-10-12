import numpy as np
from autode.log import logger
from autode.opt.coordinates import _OptCoordinates
from autode.opt.dic import DIC


class CartesianCoordinates(_OptCoordinates):
    """Flat Cartesian coordinates shape = (3 × n_atoms, )"""

    def __repr__(self):
        return f'Cartesian Coordinates({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls, input_array, units='Å') -> 'CartesianCoordinates':
        """New instance of these coordinates"""

        return super().__new__(cls, input_array.flatten(), units=units)

    def __array_finalize__(self, obj) -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        if obj is None:
            return

        for attr in ('B', '_B_T_inv'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return super().__array_finalize__(obj)

    def to(self, value: str) -> _OptCoordinates:
        """
        Allow for the transformation between cartesian and internal coordinates
        e.g. delocalised internal coordinates

        Arguments:
            value (str):

        Retuns:
            (autode.opt.coordinates._OptCoordinates):
        """
        logger.info(f'Transforming Cartesian coordinates to {value}')

        if value in ('dic', 'delocalised internal coordinates'):
            return DIC.from_cartesian(self)

        # ----------- Implement other internals here ------------

        elif any(value in unit.aliases for unit in self.implemented_units):
            return CartesianCoordinates(super().to(units=value).flatten(),
                                        units=value)
        else:
            raise ValueError(f'Cannot convert Cartesian coordinates to {value}')
