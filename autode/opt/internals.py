import numpy as np
from autode.opt.base import Primitive


class InverseDistance(Primitive):

    def derivative(self, i, k, x):
        """See ABC for docstring"""
        assert x.ndim == 2

        if i != self.idx_i and i != self.idx_j:
            # Atom does not form part of this distance
            return 0

        value = 1.0 / np.linalg.norm(x[self.idx_i] - x[self.idx_j])

        if i == self.idx_i:
            return - (x[i, k] - x[self.idx_j, k]) * value**3

        if i == self.idx_j:
            return (x[self.idx_i, k] - x[self.idx_j, k]) * value**3

        return 0

    def value(self, x):
        """1 / |x_i - x_j| """
        assert x.ndim == 2
        return 1.0 / np.linalg.norm(x[self.idx_i] - x[self.idx_j])

    def __init__(self, idx_i, idx_j):
        """
         1 / |x_i - x_j|        for a cartesian coordinates x

        Arguments:
            idx_i (int):
            idx_j (int):
        """

        self.idx_i = int(idx_i)
        self.idx_j = int(idx_j)


class PIC(list):
    """Primitive internal coordinates"""

    def B(self, x):
        """Wilson B matrix"""
        _x = x.reshape((-1, 3))

        n_atoms, _ = _x.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):
                for k in range(3):  # x, y, z components

                    B[i, 3 * j + k] = primitive.derivative(j, k, x=_x)

        return B

    def q(self, x):
        """Values of the internals"""
        _x = x.reshape((-1, 3))
        return np.array([primitive.value(_x) for primitive in self])


class InverseDistances(PIC):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    def __init__(self, x):
        super().__init__()

        n_atoms = x.shape[0] // 3

        # Add all the unique inverse distances
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(InverseDistance(i, j))
