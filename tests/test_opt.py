import pytest
import autode as ade
import numpy as np
from autode.opt import dic
from scipy.optimize import minimize

methane = ade.Molecule('methane.xyz')


def test_ic_base():

    flat_x = np.ones(9)
    internals = dic.InternalCoordinates(x=flat_x)
    assert internals.x.shape == (3, 3)

    # Can't auto-reshape something that doesn't fit into a Nx3 matrix
    with pytest.raises(AssertionError):
        flat_x = np.ones(10)
        _ = dic.InternalCoordinates(x=flat_x)


def test_inverse_dist_prim():

    x = methane.coordinates

    primitives = dic.PIC(x)

    for i in range(methane.n_atoms):
        for j in range(i+1, methane.n_atoms):
            primitives.append(dic.InverseDistance(x=x, idx_i=i, idx_j=j))

    assert np.isclose(primitives[0].value,
                      1.0 / np.linalg.norm(x[0] - x[1]))

    # Check the derivative numerically
    dh = 1E-6

    # ------------------------ dq_0 / dx_(0, k) ---------------------------
    for k in range(3):

        dq_dx = primitives[0].derivative(i=0, k=k, x=x)

        x0_shift = np.zeros(3)
        x0_shift[k] += dh

        q = 1.0 / np.linalg.norm(x[0] - x[1])
        q_plus_dh = 1.0 / np.linalg.norm(x[0] + x0_shift - x[1])
        num_dq_dx = (q_plus_dh - q) / dh

        assert np.isclose(dq_dx, num_dq_dx, atol=1E-5)

    # Derivative with respect to an atom not involved in the distance should
    # be close to zero
    assert np.isclose(primitives[0].derivative(i=2, k=0, x=x), 0.0)

    # ------------------------ dq_0 / dx_(1, 0) ---------------------------
    dq_dx = primitives[0].derivative(i=1, k=0, x=x)

    q = 1.0 / np.linalg.norm(x[0] - x[1])
    q_plus_dh = 1.0 / np.linalg.norm(x[0] - (x[1] + np.array([dh, 0.0, 0.0])))
    num_dq_dx = (q_plus_dh - q) / dh

    assert np.isclose(dq_dx, num_dq_dx, atol=1E-5)


def test_inverse_dist_dic():

    coords = dic.DIC(x=methane.coordinates)

    assert coords.x.shape == (methane.n_atoms, 3)

    n_expected = methane.n_atoms * (methane.n_atoms - 1) // 2
    assert len(coords.primitives) == n_expected
    assert coords.primitives.B.shape == (n_expected, 3 * methane.n_atoms)

    # ---------- check some elements of the B matrix ----------------
    assert np.isclose(coords.primitives.B[0, 0],
                      coords.primitives[0].derivative(0, 0, x=coords.x))

    # dq_2/dx_(0,0) is zero as the 0-th element doesn't appear in that distance
    assert np.isclose(coords.primitives.B[-1, 0], 0)

    assert np.isclose(coords.primitives.B[1, 3],
                      coords.primitives[1].derivative(i=1, k=0, x=coords.x))

    # should have 3N - 6 non-redundant internals coordinates
    assert len(coords.s) == 3 * methane.n_atoms - 6

    # all distances should be ~1-3 A, so inverses between 0.3 and 1
    assert 0.4 < np.average(coords.primitives.q) < 1.0

    # should be able to set a slightly perturbed set of internals
    new_s = coords.s.copy()
    new_s[0] += 0.1

    coords.s = new_s
    new_methane = methane.copy()

    for i, coord in enumerate(coords.x):
        new_methane.atoms[i].coord = coord

    new_methane.print_xyz_file()


"""
def test_dic_opt():

    water = ade.Molecule(smiles='O')
    coords = dic.DIC(water.coordinates)

    result = minimize(fun=energy,
                      x0=coords.s,      # s
                      args=(water, method),
                      method='L-BFGS',
                      jac=gradient)             # g^int
"""