import pytest
import shutil
import numpy as np
from autode import Molecule, Atom
from autode.methods import XTB
from autode.opt.internals import InverseDistances
from autode.opt.primitives import InverseDistance
from autode.opt.cartesian import CartesianCoordinates
from autode.opt.optimisers import CartesianSteepestDecent


def methane_mol():
    return Molecule(atoms=[Atom('C',  0.11105, -0.21307,  0.00000),
                           Atom('H',  1.18105, -0.21307,  0.00000),
                           Atom('H', -0.24562, -0.89375,  0.74456),
                           Atom('H', -0.24562, -0.51754, -0.96176),
                           Atom('H', -0.24562,  0.77207,  0.21720)])


def test_primitives():

    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    inv_dist = InverseDistance(0, 1)
    assert np.isclose(inv_dist(x), 0.5)     # 1/2.0 = 0.5 Å-1

    # Check a couple of derivatives by hand
    assert np.isclose(inv_dist.derivative(0, 'x', x=x),
                      2*inv_dist(x)**3)
    assert np.isclose(inv_dist.derivative(1, 'x', x=x),
                      -2*inv_dist(x)**3)

    # Derivatives with respect to zero components
    assert np.isclose(inv_dist.derivative(0, 'y', x=x),
                      0)
    # or those that are not present in the system should be zero
    assert np.isclose(inv_dist.derivative(2, 'x', x=x),
                      0)


def test_cart_to_dic():

    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)
    assert x.ndim == 1

    # Should only have 1 internal coordinate
    pic = InverseDistances(x)
    assert len(pic) == 1

    # Delocalised internals should preserve the single internal coordinate
    dics = x.to('dic')
    assert len(dics) == 1
    # and store the previous catesian coordinates
    assert hasattr(dics, '_x')
    # as a copy, so changing the initial x should not change prev_x
    x += 0.1
    assert not np.allclose(x, dics._x)
    x -= 0.1


def test_simple_dic_to_cart():
    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    # Should be able to transform back to Cartesians
    dic = CartesianCoordinates(arr).to('dic')

    x = dic.to('cartesian')
    assert np.allclose(CartesianCoordinates(arr), x)

    assert np.allclose(np.array([0.6]),
                       dic + 0.1)
    # Updating the DICs should afford cartesian coordinates that are
    # ~1.7 Å apart (1/r = 0.6)
    dic.update(delta=0.1)
    assert dic.shape == (1,)
    assert np.isclose(dic[0], 0.6)

    arr_update = dic.to('cart').reshape((2, 3))

    assert np.isclose(np.linalg.norm(arr_update[0, :] - arr_update[1, :]),
                      1.66666,
                      atol=1E-4)

    # DICs must be updated with either 'new' or 'delta' kwargs
    with pytest.raises(ValueError):
        dic.update(1)

    with pytest.raises(ValueError):
        dic.update(an_undefined_keyword_argument=1)

    with pytest.raises(ValueError):
        dic.update(new=np.array([0, 1]))       # Wrong shape

    with pytest.raises(Exception):
        dic.update(delta=np.array([0, 1]))      # Wrong shape


def test_methane_cart_to_dic():

    x = CartesianCoordinates(methane_mol().coordinates)
    dic = x.to('dic')
    assert len(dic) == 9   # 3N-6 for N=5

    dic.update(delta=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Cartesian coordinates should be close to the starting ones
    assert np.linalg.norm(x - dic.to('cart')) < 0.5


def test_co2_cart_to_dic():

    arr = np.array([[-1.31254, 0.34625, -0.00000],
                    [-0.11672, 0.30964, 0.00000],
                    [1.07904, 0.27311, 0.00000]])

    x = CartesianCoordinates(arr)
    dic = x.to('dic')
    assert len(dic) == 3

    # Applying a shift to the internal coordinates that are close to linear
    # can break the back transformation to Cartesians
    with pytest.raises(RuntimeError):
        dic.update(delta=np.array([0.0, 0.0, 0.1]))


def test_grad_transform_linear():

    k = 1.0
    r0 = 1.0

    def energy(_x):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0)**2

    def grad(_x):
        _x = _x.reshape((-1, 3))
        diff = _x[0, 0] - _x[1, 0]
        r = np.linalg.norm(_x[0] - _x[1])
        return np.array([k * (r - r0) * diff/r,
                         0.0,
                         0.0,
                         - k * (r - r0) * diff/r,
                         0.0,
                         0.0])

    def num_grad(_x, h=1E-8):

        _g = []
        for i in range(len(_x.flatten())):

            x_ph = np.array(_x, copy=True)
            x_ph[i] += h

            g_i = (energy(x_ph) - energy(_x)) / h
            _g.append(g_i)

        return np.array(_g)

    coords = np.array([[0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    assert np.allclose(num_grad(x), grad(x))
    x.g = grad(coords)

    dic = x.to('dic')
    assert dic.shape == (1,)                   # Only a single distance
    assert np.isclose(dic[0], 0.5, atol=1E-6)  # 1/r_012 = 0.5 Å

    assert dic.g.shape == (1,)   # dE/ds_i has only a single component

    # Determined by hand
    assert np.isclose(dic.g[0],
                      -1/0.5**2 * grad(coords)[3])


def test_hess_transform_linear():

    k = 1.0
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    def energy(_x, r0=1):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0)**2

    def hessian(_x):
        _x = _x.reshape((-1, 3))
        delta_x = _x[0, 0] - _x[1, 0]

        r = np.linalg.norm(_x[0] - _x[1])
        h_oo = k*(1 - 1/r + delta_x**2/r**3)

        _h = np.zeros(shape=(6, 6))
        _h[0, 0] = _h[3, 3] = h_oo
        _h[0, 3] = _h[3, 0] = -h_oo

        return _h

    def g_i(_x, i, h=1E-8):
        """Numerical graident"""
        x_ph = np.array(_x, copy=True)
        x_ph[i] += h
        return (energy(x_ph) - energy(_x)) / h

    def h_ij(_x, i, j, h=1E-8):

        x_ph = np.array(_x, copy=True)
        x_ph[j] += h

        return (g_i(x_ph, i=i) - g_i(_x, i=i)) / h

    def num_hess(_x):
        _h = np.zeros(shape=(6, 6))

        for i in range(6):
            for j in range(6):

                _h[i, j] = h_ij(_x, i, j)

        return _h

    assert np.linalg.norm(hessian(x) - num_hess(x)) < 1E-7
    x.h = hessian(coords)

    dic = x.to('dic')
    assert dic.h.shape == (1, 1)         # 1x1 internal Hessian
    assert np.isclose(dic.h[0, 0], k)    # should be ~k


def test_xtb_h2_cart_opt():

    mol = Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.5)])

    # Don't run the calculation without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    CartesianSteepestDecent.optimise(mol, method=XTB(), maxiter=50)
    assert np.isclose(mol.distance(0, 1), 0.777, atol=0.1)
