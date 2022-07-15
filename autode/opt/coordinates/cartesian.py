import numpy as np

from abc import ABC, abstractmethod
from typing import Optional
from autode.log import logger
from autode.geom import get_rot_mat_euler
from autode.values import ValueArray
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.dic import DIC


def cartesian_coordinates(species: 'autode.species.Species') -> 'CartesianCoordinates':
    """
    Construct an appropriate set of cartesian coordinates for a species.

    ---------------------------------------------------------------------------
    Arguments:
        species: Species for which to generate the coordinates. If it is planar
                 then 2D coordinates will be constructed, otherwise 3D

    Returns:
        CartesianCoordinates: Coordinates
    """
    atoms = species.atoms
    coordinates = species.coordinates

    if species.is_planar() and species.n_atoms > 2:
        rot_mat = _rotation_matrix_into_xy_plane(atoms)
        x = CartesianCoordinates2D(coordinates.dot(rot_mat.T)[:, :2])
        x.rev_rotation_matrix = np.linalg.inv(rot_mat)
        return x

    else:
        return CartesianCoordinates3D(coordinates)


class CartesianCoordinates(OptCoordinates, ABC):  # lgtm [py/missing-equals]
    """Cartesian coordinates"""

    @property
    @abstractmethod
    def num_dimensions(self) -> int:
        """Number of dimensions of space these coordinates occupy"""

    @property
    def n_atoms(self) -> int:
        """Number of atoms that comprise these cartesian coordinates"""

        if self.ndim != 2:
            raise ValueError("Cannot determine the number of atoms with a flat"
                             " array of coordinates")

        return self.shape[0]

    def __repr__(self):
        return f'Cartesian Coordinates({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls, input_array, units='Å') -> 'CartesianCoordinates':
        """New instance of these coordinates"""
        return super().__new__(cls, np.array(input_array), units=units)

    def __array_finalize__(self, obj) -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        return None if obj is None else super().__array_finalize__(obj)

    def _str_is_valid_unit(self, string) -> bool:
        """Is a string a valid unit for these coordinates e.g. nm"""
        return any(string in unit.aliases for unit in self.implemented_units)

    def _update_g_from_cart_g(self,
                              arr: Optional['autode.values.Gradient']
                              ) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient, which for
        Cartesian coordinates there is nothing to be done for.

        -----------------------------------------------------------------------
        Arguments:
            arr: Gradient array
        """
        self.g = None if arr is None else np.array(arr)

    def _update_h_from_cart_h(self,
                              arr: Optional['autode.values.Hessian']
                              ) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms.


        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        self.h = None if arr is None else np.array(arr)

    def iadd(self, value: np.ndarray) -> OptCoordinates:
        return np.ndarray.__iadd__(self, value)

    def to(self, value: str) -> OptCoordinates:
        """
        Transform between cartesian and internal coordinates e.g. delocalised
        internal coordinates or other units

        -----------------------------------------------------------------------
        Arguments:
            value (str): Intended conversion

        Returns:
            (autode.opt.coordinates.OptCoordinates): Transformed coordinates

        Raises:
            (ValueError): If the conversion cannot be performed
        """
        logger.info(f'Transforming Cartesian coordinates to {value}')

        if value.lower() in ('cart', 'cartesian', 'cartesiancoordinates'):
            return self

        elif value.lower() in ('dic', 'delocalised internal coordinates'):
            return DIC.from_cartesian(self)

        # ---------- Implement other internal transformations here -----------

        elif self._str_is_valid_unit(value):
            return self.__class__(ValueArray.to(self, units=value), units=value)
        else:
            raise ValueError(f'Cannot convert Cartesian coordinates to {value}')


class CartesianCoordinates3D(CartesianCoordinates):
    """Cartesian coordinates in 3 dimensions"""

    @property
    def num_dimensions(self) -> int:
        assert self.shape[1] == 3
        return 3


class CartesianCoordinates2D(CartesianCoordinates):
    """Cartesian coordinates in 2 dimensions"""

    def __new__(cls, *args, **kwargs):
        arr = super().__new__(cls, *args, **kwargs)

        arr.rev_rotation_matrix = None   # Rotation matrix from the xy plane

        return arr

    @property
    def num_dimensions(self) -> int:
        assert self.shape[1] == 2
        return 2

    def to_3d(self) -> CartesianCoordinates3D:
        """
        Convert these coordinates to a 3D set by adding a zero z value
        to each coordinate and then rotating with the inverse transformation
        """

        if self.rev_rotation_matrix is None:
            raise RuntimeError("Cannot convert to 3D without an inverse "
                               "transform matrix")

        coords3d = np.zeros(shape=(self.n_atoms, 3))
        coords3d[:, :2] = self[:, :]
        coords3d = coords3d.dot(self.rev_rotation_matrix.T)

        return CartesianCoordinates3D(coords3d, units=self.units)


def _rotation_matrix_into_xy_plane(atoms: 'autode.atoms.Atoms') -> np.ndarray:
    """Rotation matrix to orientate a planar molecule in the xy plane"""
    assert len(atoms) > 2

    normal = np.cross(atoms.nvector(0, 1), atoms.nvector(0, 2))
    normal /= np.linalg.norm(normal)
    z_axis = np.array([0., 0., 1.])

    rot_mat = get_rot_mat_euler(axis=np.cross(normal, z_axis),
                                theta=np.arccos(np.dot(normal, z_axis)))

    return rot_mat
