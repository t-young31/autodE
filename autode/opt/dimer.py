"""
Dimer method for finding transition states given two points on the PES.
Notation follows
1. https://aip.scitation.org/doi/10.1063/1.2815812
based on
2. https://aip.scitation.org/doi/10.1063/1.2104507
3. https://aip.scitation.org/doi/10.1063/1.480097

x : Cartesian coordinates
g : gradient in cartesian coordinates
"""
import autode as ade
import numpy as np
from autode.log import logger


class Dimer:
    """Dimer spanning two points on the PES with a TS at the midpoint"""

    def _gradient(self, coordinates):
        """
        Calculate the gradient for a set of coordinates

        Args:
            coordinates (np.ndarray): Cartesian coordinates with shape
                        (n_atoms, 3) or (3*n_atoms,)

        Returns:
            (np.ndarray): Gradient shape = (3*n_atoms,)  (Ha / Å)

        Raises:
            (autode.exceptions.CalculationException): If a calculation fails
        """
        self._species.coordinates = coordinates

        # Run the calculation and remove all the files, if it's successful
        calc = ade.Calculation(name='tmp_grad',
                               molecule=self._species,
                               method=self.method,
                               keywords=self.method.keywords.grad,
                               n_cores=ade.Config.n_cores)
        calc.run()

        grad = calc.get_gradients()
        calc.clean_up(force=True, everything=True)

        return grad.flatten()

    @property
    def tau(self):
        """τ = (x1 - x2)/2"""
        return (self.x1 - self.x2) / 2.0

    @property
    def tau_hat(self):
        """^τ = τ / |τ|"""
        tau = self.tau
        return tau / np.linalg.norm(tau)

    @property
    def delta(self):
        """Distance between the dimer point, Δ"""
        return np.linalg.norm(self.x1 - self.x2) / 2.0

    @property
    def theta(self):
        """Rotation direction Θ, calculated using steepest descent"""
        tau_hat = self.tau_hat
        f_r = (-2.0 * (self.g1 - self.g0)
               + 2.0 * (np.dot((self.g1 - self.g0), tau_hat)) * tau_hat)

        return f_r / np.linalg.norm(f_r)

    @property
    def c(self):
        """Curvature of the PES, C_τ.  eqn. 4 in ref [1]"""
        return np.dot((self.g1 - self.g0), self.tau_hat) / self.delta

    @property
    def dc_dphi(self):
        """dC_τ/dϕ eqn. 6 in ref [1] """
        return 2.0 * np.dot((self.g1 - self.g0), self.theta) / self.delta

    def rotate_coords(self, phi, update_g1=False):
        """
        Rotate the dimer by an angle phi around the midpoint.
        eqn. 13 in ref. [2]

        Arguments:
            phi (float): Rotation angle in radians (ϕ)

        Keyword Arguments:
            update_g1 (bool): Update the gradient on point 1 after the rotation
        """
        delta, tau_hat, theta = self.delta, self.tau_hat, self.theta

        self.x1 = self.x0 + delta * (tau_hat * np.cos(phi) + theta * np.sin(phi))
        self.x2 = self.x0 - delta * (tau_hat * np.cos(phi) + theta * np.sin(phi))

        if update_g1:
            self.g1 = self._gradient(coordinates=self.x1)

        return None

    def rotate(self):
        """Do a single steepest descent rotation of the dimer"""

        # Curvature at ϕ=0 i.e. no rotation
        c_phi0 = self.c

        # and derivative with respect to the rotation evaluated at ϕ=0
        dc_dphi0 = self.dc_dphi

        # test point for rotation. eqn. 5 in ref [1]
        phi_1 = -0.5 * np.arctan(dc_dphi0 / (2.0 * np.linalg.norm(c_phi0)))

        # rotate to the test point and update the gradient
        self.rotate_coords(phi=phi_1, update_g1=True)

        b1 = 0.5 * dc_dphi0  # eqn. 8 from ref. [1]

        a1 = ((c_phi0 - self.c + b1 * np.sin(2 * phi_1)) # eqn. 9 from ref. [1]
              / (1 - 2.0 * np.cos(2.0 * phi_1)))

        a0 = 2.0 * (c_phi0 - a1)  # eqn. 10 from ref. [1]

        phi_min = 0.5 * np.arctan(b1 / a1)
        c_min = 0.5 * a0 + a1 * np.cos(2.0 * phi_min) + b1 * np.sin(
            2.0 * phi_min)

        if c_min > c_phi0:
            logger.info('Optimised curvature was larger than the initial, '
                        'adding π/2')
            c_min += np.pi / 2.0

        self.rotate_coords(phi=phi_min, update_g1=True)

        self.iterations.append(DimerIteration(phi=phi_min, d=0, dimer=self))

        return None

    def optimise_rotation(self, phi_tol=1E-2, max_iterations=50):
        """Rotate the dimer optimally

        Keyword Arguments:
            phi_tol (float): Tolerance below which rotation is not performed
            max_iterations (int): Maximum number of rotation steps to perform
        """
        logger.info(f'Minimising dimer rotation up to {phi_tol}')
        iteration, phi = 0, np.inf

        while iteration < max_iterations and phi > phi_tol:
            self.rotate()

            phi = self.iterations[-1].phi
            logger.info(f'Iteration={iteration}   ϕ={phi:.4f}')

            iteration += 1

        return None

    def __init__(self, species_1, species_2, species_mid, method):
        """
        Initialise a dimer from three species, one either side of a peak and a
         midpoint (species_mid)

        Arguments:
            species_1: (ade.species.Species)

            species_2: (ade.species.Species)

            species_mid: (ade.species.Species)

            method (autode.wrappers.base.ElectronicStructureMethod):
        """
        self.method = method  # Method for gradient evaluations

        # Temporary species used to perform gradient calculations
        self._species = species_1.copy()

        # Note the notation follows [1] and is not necessarily the most clear..
        self.x0 = species_mid.coordinates.flatten()
        self.x1 = species_1.coordinates.flatten()
        self.x2 = species_2.coordinates.flatten()

        # Run two initial gradient evaluations
        self.g0 = self._gradient(coordinates=self.x0)
        self.g1 = self._gradient(coordinates=self.x1)

        self.iterations = [DimerIteration(phi=0, d=0, dimer=self)]


class DimerIteration:
    """Single iteration of a TS dimer"""

    def __init__(self, phi, d, dimer):
        """
        Initialise from a rotation angle, a distance and the whole dimer

        Arguments:
            phi (float): Rotation with respect to the previous iteration
            d (float): Translation distance with respect to the previous
                       iteration
            dimer (autode.opt.dimer.Dimer):
        """
        self.phi = phi
        self.d = d

        self.x0 = dimer.x0.copy()
        self.x1 = dimer.x1.copy()
        self.x2 = dimer.x2.copy()

        self.g0 = dimer.g0.copy()
        self.g1 = dimer.g1.copy()
