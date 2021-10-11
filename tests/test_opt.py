import numpy as np
import pytest

from autode.opt.internals import InverseDistances
from autode.opt.primitives import InverseDistance
from autode.opt.cartesian import CartesianCoordinates


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


def test_dic_to_cart():
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
