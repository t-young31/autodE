"""
Delocalised internal coordinate implementation from:
1. https://aip.scitation.org/doi/pdf/10.1063/1.478397
and references cited therein. Also used is
2. https://aip.scitation.org/doi/pdf/10.1063/1.1515483

The notation follows the paper and is briefly
summarised below:

x : Cartesian coordinates
B : Wilson B matrix
G : 'Spectroscopic G matrix'
q : Redundant internal coordinates
s : Non-redundant internal coordinates
U : Transformation matrix q -> s
"""
import numpy as np
from time import time
from autode.opt.internals import PIC, InverseDistances, InternalCoordinates
from autode.log import logger


class DIC(InternalCoordinates):
    """Delocalised internal coordinates"""

    def __repr__(self):
        return f'DIC(n={len(self)})'

    @staticmethod
    def U(primitives: PIC) -> np.ndarray:
        """
        Transform matrix

        Arguments:
            primitives (autode.opt.internals.PIC):

        Returns:
            (np.ndarray): U
        """

        B_q = primitives.B
        G = np.matmul(B_q, B_q.T)

        w, v = np.linalg.eigh(G)  # Eigenvalues and eigenvectors respectively

        # Form a transform matrix from the primitive internals to a set of
        # 3N - 6 non-redundant internals, s
        return v[:, np.where(np.abs(w) > 1E-10)[0]]

    @classmethod
    def from_cartesian(cls,
                       x:             'autode.opt.cartesian.CartesianCoordinates',
                       primitive_type: PIC = InverseDistances):
        """
        Convert cartesian coordinates to primitives then to delocalised
        internal coordinates (DICs), of which there should be 3N-6 for a
        polyatomic system with N atoms

        Arguments:
            x (autode.opt.cartesian.CartesianCoordinates): Cartesian coordinates

            primitive_type (autode.opt.internals.PIC): Primitive internal
                           coordinates, constructable from Cartesian

        Returns:
            (autode.opt.cartesian.CartesianCoordinates):
        """
        logger.info('Converting cartesian coordinates to DIC')
        start_time = time()

        primitives = primitive_type(x)
        U = cls.U(primitives)

        s = cls(input_array=np.matmul(U.T, primitives.q))
        s.B = np.matmul(U.T, primitives.B)
        s.B_T_inv = np.linalg.pinv(s.B)
        s._x = x.copy()
        s.primitive_type = primitive_type

        # Set the internal gradient
        if x.g is not None:
            s.g = np.matmul(s.B_T_inv.T, x.g)

        # and Hessian
        if x.h is not None:
            # NOTE: This is not the full transformation as noted in
            # 10.1063/1.471864 only an approximate Hessian is required(?)
            s.h = np.linalg.multi_dot((s.B_T_inv.T, x.h, s.B_T_inv))

        logger.info(f'Transformed in      ...{time() - start_time:.4f} s')
        return s

    def to(self, value: str) -> 'autode.opt.coordinates.OptCoordinates':
        """
        Convert these DICs to another type of coordinate

        Arguments:
            value (str):

        Returns:
            (autode.opt.coordinates.OptCoordinates): Coordinates
        """

        if value.lower() in ('x', 'cart', 'cartesian'):
            return self._x

        raise ValueError(f'Unknown conversion to {value}')

    def _new_s_from_kwargs(self, kwargs):
        """Determine a new set of DICs from some keyword arguments"""

        if 'new' in kwargs:
            if kwargs['new'].shape != self.shape:
                raise ValueError('To update the internal coordinates a new '
                                 f'set with shape {self.shape} is needed, but '
                                 f'had {kwargs["new"].shape}')
            return kwargs['new']

        elif 'delta' in kwargs:
            return np.array(self, copy=True) + kwargs['delta']
        else:
            raise ValueError('Expecting one of: *new* or *delta* as keyword '
                             f'arguments. Had only {kwargs}')

    def update(self, *args, **kwargs) -> None:
        """
        Set some new internal coordinates and update the Cartesian coordinates

        .. math::

            x^(k+1) = x(k) + ({B^T})^{-1}(k)[s_{new} - s(k)]

        for an iteration k.

        Keyword Arguments:
            new (np.ndarray): New internal coordinates, must be the same shape
                              as the current coordinates

            delta (int | float | np.ndarray): Difference between the current
                                              and new DICs. Must be
                                              broadcastable into self.shape.
        Raises:
            (RuntimeError): If the transformation diverges
        """
        start_time = time()
        s_new = self._new_s_from_kwargs(kwargs)

        # Initialise
        s_k, x_k = np.array(self, copy=True), self._x.copy()
        U = self.U(primitives=self.primitive_type(x_k))

        iteration = 0

        # Converge to an RMS difference of less than a tolerance
        while np.average(s_k - s_new) ** 2 > 1E-10 and iteration < 100:

            x_k = x_k + np.matmul(self.B_T_inv, (s_new - s_k))

            if np.max(np.abs(x_k)) > 1E5:
                raise RuntimeError('Something went very wrong in the back '
                                   'transformation from internal -> carts')

            # Rebuild the primitives from the back-transformer Cartesians
            primitives = self.primitive_type(x_k)
            s_k = np.matmul(U.T, primitives.q)
            B = np.matmul(U.T, primitives.B)
            self.B_T_inv = np.linalg.pinv(B)

            iteration += 1

        logger.info(f'Converged in {iteration} cycles and '
                    f'{time() - start_time:.4f} s')

        self[:] = s_k
        self.clear_gradient_and_hessian()

        self._x = x_k
        self._x.clear_gradient_and_hessian()
        return None

    def _iadd(self, other: np.ndarray):
        """Inplace addition of another set of coordinates"""
        self.update(delta=other)
        return self
