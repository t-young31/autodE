import numpy as np
from autode.wrappers.base import Method
from autode.species import Molecule
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.trust_region import (CauchyTROptimiser,
                                                DoglegTROptimiser,
                                                CGSteihaugTROptimiser)


def branin_energy(x, y):
    return (y - 0.129 * x ** 2 + 1.6 * x - 6) ** 2 + 6.07 * np.cos(x) + 10




class BraninCauchyTROptimiser(CauchyTROptimiser):

    __test__ = False

    def _update_gradient_and_energy(self) -> None:
        """Update the gradient and energy for the Branin function

        f(x, y) = (y - 0.129x^2 + 1.6x - 6)^2 + 6.07cos(x) + 10
        """
        x, y = self._coords

        self._coords.e = branin_energy(x, y)

        grad = [(2 * (1.6 - 0.258*x) * (y - 0.129*x**2 + 1.6*x - 6)
                 - 6.07*np.sin(x)),
                (2 * (y - 0.129*x**2 + 1.6*x - 6))]

        self._coords.g = np.array(grad)

        h_xx = (2*(1.6-0.258*x)**2
                - 0.516*(-0.129*x**2 + 1.6*x + y - 6)
                - 6.07*np.cos(x))
        h_xy = 2*(1.6 - 0.258*x)
        h_yy = 2
        self._coords.h = np.array([[h_xx, h_xy],
                                   [h_xy, h_yy]])

    def _log_convergence(self) -> None:

        attrs = [self.iteration, self._coords.e, *self._coords, self.rho,
                      self.alpha, np.linalg.norm(self.p), self._g_norm]

        if hasattr(self, 'tau'):
            attrs.append(self.tau)

        for thing in attrs:
            print(f'{round(thing, 3):10.3f}'
                  f'', end=' ')
        print()


class BraninDoglegTROptimiser(DoglegTROptimiser):

    def _update_gradient_and_energy(self) -> None:
        return BraninCauchyTROptimiser._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        return BraninCauchyTROptimiser._log_convergence(self)


class CGSteihaugTROptimiser(CGSteihaugTROptimiser):

    def _update_gradient_and_energy(self) -> None:
        return BraninCauchyTROptimiser._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        return BraninCauchyTROptimiser._log_convergence(self)


def test_branin_minimisation():
    """Uses the example from:
    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
    """

    init_coords = CartesianCoordinates([6.0, 14.0])

    optimiser = BraninCauchyTROptimiser(maxiter=20,
                                        etol=100,  # Some large value
                                        trust_radius=2.0,
                                        coords=init_coords,
                                        gtol=0.01,
                                        max_trust_radius=5.0,
                                        t_1=0.25,
                                        t_2=2.0,
                                        eta_1=0.2,
                                        eta_2=0.25,
                                        eta_3=0.75)

    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged
    assert np.allclose(optimiser._coords,
                       np.array([3.138, 2.252]),
                       atol=0.01)


def test_branin_dogleg_minimisation():

    # Dogleg optimiser doesn't seem to be so efficient - TODO: Check
    optimiser = BraninDoglegTROptimiser(maxiter=1000,
                                        etol=100,  # Some large value
                                        trust_radius=2.0,
                                        coords=CartesianCoordinates([6., 14.]),
                                        gtol=0.01,
                                        max_trust_radius=5.0,
                                        t_1=0.25,
                                        t_2=2.0,
                                        eta_1=0.2,
                                        eta_2=0.25,
                                        eta_3=0.75)

    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords,
                       np.array([3.138, 2.252]),
                       atol=0.02)


def test_branin_cg_minimisation():

    optimiser = CGSteihaugTROptimiser(maxiter=1000,
                                      etol=100,  # Some large value
                                      trust_radius=2.0,
                                      coords=CartesianCoordinates([6., 14.]),
                                      gtol=0.01,
                                      max_trust_radius=5.0,
                                      t_1=0.25,
                                      t_2=2.0,
                                      eta_1=0.2,
                                      eta_2=0.25,
                                      eta_3=0.75)

    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords,
                       np.array([3.138, 2.252]),
                       atol=0.02)
