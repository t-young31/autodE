import pytest
import autode as ade
import numpy as np
from autode.opt import dic, internal_opt, internals
from autode.geom import are_coords_reasonable
from scipy.optimize import minimize

methane = ade.Molecule(smiles='C')


def test_ic_base():
    """Test base level internal coordinate properties"""

    flat_x = np.ones(9)
    coords = dic.InternalCoordinates(x=flat_x)
    assert coords.x.shape == (3, 3)

    # Can't auto-reshape something that doesn't fit into a Nx3 matrix
    with pytest.raises(AssertionError):
        flat_x = np.ones(10)
        _ = dic.InternalCoordinates(x=flat_x)


def test_inverse_dist_prim():
    """Test inverse distances, derivative etc."""

    x = methane.coordinates

    primitives = internals.PIC(x)

    for i in range(methane.n_atoms):
        for j in range(i+1, methane.n_atoms):
            primitives.append(internals.InverseDistance(x=x, idx_i=i, idx_j=j))

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


def test_inverse_dist_dic_methane():
    """Test inverse distance delocalised internals for methane"""

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
    new_methane.coordinates = coords.x

    assert np.sqrt(np.average(new_methane.coordinates -
                              methane.coordinates)**2) < 1E-1


def test_inverse_dist_dic_ethane():
    """Test a slightly larger example with DIC from inverse distances"""

    ethane = ade.Molecule(smiles='CC')
    # ethane.print_xyz_file()

    coords = dic.DIC(x=ethane.coordinates)

    new_s = coords.s.copy()
    new_s[0] += 0.1

    coords.s = new_s
    new_mol = ethane.copy()
    new_mol.coordinates = coords.x
    # new_mol.print_xyz_file(filename='tmp.xyz')

    rmsd = np.sqrt(np.average(new_mol.coordinates - ethane.coordinates)**2)
    assert rmsd < 1E-1


def test_dic_opt():
    """Test a simple optimisation of methane with DICs"""

    mol = ade.Molecule(smiles='C')
    coords = dic.DIC(mol.coordinates)

    xtb = ade.methods.XTB()

    result = minimize(fun=internal_opt.energy,
                      x0=coords.s,                  # s
                      args=(mol, xtb, coords),
                      method='BFGS',
                      options={'gtol': 1E-3},
                      jac=internal_opt.gradient)             # g^int

    # below have been checked by hand!
    # print(result)
    # mol.print_xyz_file()

    assert result.success
    assert are_coords_reasonable(mol.coordinates)
