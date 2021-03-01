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
import time
import numpy as np
from autode.opt.base import Coordinates
from autode.opt.internals import InverseDistances
from autode.log import logger


class DIC(Coordinates):
    """Delocalised internal coordinates"""

    def __str__(self):
        return f'DIC(n={len(self.s)})'

    @Coordinates.g.setter
    def g(self, g_x):
        """Set the gradient in internal coordinates. eqn. 7 in ref [1]"""
        self._g_s = np.matmul(self.B_T_inv.T, g_x)

    @Coordinates.H.setter
    def H(self, H_x):
        raise NotImplementedError

    def _cart_to_s(self, x):
        """Convert cartesians to primitives then to DICs"""
        logger.info('Converting cartesian coordinates to DIC')
        start_time = time.time()

        self.primitives = self.primitives_type(x)

        B_q = self.primitives.B(x)
        G = np.matmul(B_q, B_q.T)

        w, v = np.linalg.eigh(G)  # Eigenvalues and eigenvectors respectively

        # Form a transform matrix from the primitive internals to a set of
        # 3N - 6 non-redundant internals, s
        self.U = v[:, np.where(np.abs(w) > 1E-10)[0]]

        _s = np.matmul(self.U.T, self.primitives.q(x))

        self.B = np.matmul(self.U.T, B_q)
        self.B_T_inv = np.linalg.pinv(self.B)

        logger.info(f'Transformed in      ...{time.time() - start_time:.4f} s')
        return _s

    def _s_to_cart(self, s_new):
        """
        Set some new internal coordinates and update the Cartesians

        x(k+1) = x(k) + B^T^{-1}(k)[s_new - s(k)

        for an iteration k

        Arguments:
            s_new (np.ndarray): New internal coordinates used to update
        """
        start_time = time.time()

        # Initialise
        s_k, x_k = self._s.copy(), self.x.copy()
        iteration = 0

        # Converge to an RMS difference of less than a tolerance
        while np.sqrt(np.average(s_k - s_new) ** 2) > 1E-6 and iteration < 100:

            x_k = x_k + np.matmul(self.B_T_inv, (s_new - s_k))

            if np.max(np.abs(x_k)) > 1E5:
                raise RuntimeError('Something went very wrong in the back '
                                   'transformation from internal -> carts')

            # Rebuild the primitives from the back-transformer Cartesians
            s_k = np.matmul(self.U.T, self.primitives.q(x_k))

            B = np.matmul(self.U.T, self.primitives.B(x_k))
            self.B_T_inv = np.linalg.pinv(B)

            iteration += 1

        logger.info(f'Converged in {iteration} cycles and '
                    f'{time.time() - start_time:.4f} s')
        self._s = s_k

        # Set the cartesian array as an Nx3 matrix
        return x_k

    def __init__(self, x, primitives_type=InverseDistances):
        """
        Construct a set of delocalised internal coordinates from a cartesian
        set (x)

        Args:
            x (np.ndarray):  Cartesian coordinates shape = (N, 3) for N atoms

            primitives_type (ade.opt.internal.Primitive):
        """
        self.primitives_type = primitives_type

        self.primitives = None          # (Redundant) internal coordinates
        self.U = None                   # Transform matrix redundant -> non

        # NOTE: call superclass after primitives_type has been set as it's
        # needed in _cart_to_s
        super().__init__(cartesian_coordinates=x)
