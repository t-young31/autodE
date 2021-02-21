import numpy as np
from abc import ABC


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

        self.g = None                   # Gradient (dE/dq) in Ha / Å
        self.B_T_inv = None             # (B^T)^{-1}

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
