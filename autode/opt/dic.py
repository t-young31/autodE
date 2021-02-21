"""
Delocalised internal coordinate implementation from:
https://aip.scitation.org/doi/pdf/10.1063/1.478397
and references cited therein
"""
import time
import numpy as np
from abc import ABC
from autode.log import logger


class Primitive(ABC):
    """Primitive internal coordinate"""

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
        raise NotImplementedError


class InverseDistance(Primitive):

    def derivative(self, i, k, x):
        """See ABC for docstring"""

        if i == self.idx_i:
            return - (x[i, k] - x[self.idx_j, k]) * self.value**3

        if i == self.idx_j:
            return (x[self.idx_i, k] - x[self.idx_j, k]) * self.value**3

        # Atom does not form part of this distance
        return 0

    def __init__(self, x, idx_i, idx_j):
        """
         1 / |x_i - x_j|        for a cartesian coordinates x (shape = (3N, 3))

        Arguments:
            x (np.ndarray): Cartesian coordinates shape = (N, 3) for N atoms
            idx_i (int):
            idx_j (int):
        """

        self.value = 1.0 / np.linalg.norm(x[idx_i] - x[idx_j])
        self.idx_i = int(idx_i)
        self.idx_j = int(idx_j)


class InternalCoordinates(list):

    def __init__(self, x: np.ndarray):
        """
        Set of primitive redundant inverse distances

        Args:
            x (np.ndarray):  Cartesian coordinates shape = (N, 3) for N atoms
        """
        super().__init__()

        if x.ndim == 1:
            assert x.shape[0] % 3 == 0
            self.x = x.reshape((-1, 3))

        if x.ndim == 2:
            n_atoms, n_components = x.shape
            assert n_atoms > 0 and n_components == 3

            self.x = x


class PIC(InternalCoordinates):
    """Primitive internal coordinates"""

    @property
    def B(self):
        """Wilson B matrix"""

        n_atoms, _ = self.x.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):
                for k in range(3):  # x, y, z components

                    B[i, 3 * j + k] = primitive.derivative(j, k, x=self.x)

        return B

    @property
    def q(self):
        """Values of the internals"""
        return np.array([coord.value for coord in self])


class InverseDistances(PIC):
    """1 / r_ij for all unique pairs i,j"""

    def __init__(self, x):
        super().__init__(x=x)

        n_atoms, _ = self.x.shape

        # Add all the unique inverse distances
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(InverseDistance(self.x, i, j))


class DIC(InternalCoordinates):
    """Delocalised internal coordinates"""

    def _cart_to_dic(self, x):
        """Convert cartesians to primitives then to DICs"""
        logger.info('Converting cartesian coordinates to DIC')
        start_time = time.time()

        self.primitives = self.primitives_type(x)

        B_q = self.primitives.B
        print(self.primitives.q, 'PIC')
        print(B_q, 'prim B')
        G = np.matmul(B_q, B_q.T)

        w, v = np.linalg.eigh(G)  # Eigenvalues and eigenvectors respectively

        # Form a transform matrix from the primitive internals to a set of
        # 3N - 6 non-redundant internals, s
        U = v[:, np.where(np.abs(w) > 1E-10)[0]]

        _s = np.matmul(U.T, self.primitives.q)
        print(_s, 'DIC')


        self.B = np.matmul(U.T, B_q)
        self.B_T_inv = np.linalg.pinv(self.B)

        logger.info(f'Transformed in      ...{time.time() - start_time:.4f} s')
        return _s

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, new_s):
        """
        Set some new internal coordinates and update the Cartesians

        x(k+1) = x(k) + B^T^{-1}(k)[s_new - s(k)

        for an iteration k

        Arguments:
            new_s (np.ndarray): New internal coordinates used to update
        """
        logger.info('Setting new internal and cartesian coordinates')
        assert self._s.shape == new_s.shape

        n_atoms, _ = self.x.shape
        s_k, x_k = self._s.copy(), self.x.copy().flatten()
        iteration = 0

        print(s_k, 'init DIC')

        while np.sqrt(np.average(s_k - new_s)**2) > 1E-2:


            # print(self.B_T_inv.shape, (new_s - s_k).shape)

            x_k = x_k + np.matmul(self.B_T_inv, (new_s - s_k))
            s_k = self._cart_to_dic(x=x_k.copy())
            print('s_k is wrong :-(. All fine to B_q')
            print(s_k, '\n')
            return

            # print(x_k)
            if iteration > 1:
                return

            iteration += 1

        logger.info(f'Converged in {iteration} cycles')
        self._s = s_k

        # Set the cartesian array as an Nx3 matrix
        self.x = x_k.reshape((-1, 3))

    def __init__(self, x, primitives_type=InverseDistances):
        """
        Construct a set of delocalised internal coordinates from a cartesian
        set (x)

        Args:
            x (np.ndarray):  Cartesian coordinates shape = (N, 3) for N atoms

            primitives_type:
        """
        super().__init__(x=x)

        self.primitives_type = primitives_type

        # ---------------- set when _cart_to_dic is called -------------
        self.primitives = None         # (Redundant) internal coordinates
        self.B = None                   # Wilson B matrix
        self.B_T_inv = None             # (B^T)^{-1}

        self._s = self._cart_to_dic(x)  # Delocalised internal coordinates
