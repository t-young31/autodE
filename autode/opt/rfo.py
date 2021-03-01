"""
Rational function optimisation method (RFO) with BFGS updating
algorithm follows the notation from
1. https://aip.scitation.org/doi/10.1063/1.1515483

-------------------------------------------------------
x : cartesian coordinates
H : Hessian matrix 3N x 3N for N atoms
g : gradient
"""
import numpy as np
from autode.log import logger
from autode.opt.base import Optimiser, CartesianCoordinates
from scipy.optimize._hessian_update_strategy import BFGS as BFGS_updater


class RFOptimiser(Optimiser):

    def _gradient(self):
        """Calculate and set the gradient"""
        self.coords.g = super()._get_gradient(coordinates=self.coords.x)
        return None

    def _estimate_hessian(self):
        """Estimate a Hessian matrix"""
        import autode as ade
        self.species.coordinates = self.coords.x

        calc = ade.Calculation(name=f'tmp_hess',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.hess,
                               n_cores=ade.Config.n_cores)
        calc.run()
        self.species.energy = calc.get_energy()
        self.coords.H = calc.get_hessian()
        self.bfgs_updater.B = self.coords.H.copy()

        self.coords.g = calc.get_gradients().flatten()
        calc.clean_up(force=True, everything=True)

        # TODO: an estimation here
        return None

    def _update_hessian_bfgs(self, dg_i, ds_i):
        """
        Update the Hessian based on the Bofill scheme. eqn. 42-47 in ref [1]

        Arguments:
            dg_i (np.ndarray): Difference between the current gradient and the
                               previous iteration. Δg_i, shape = (3*n_atoms,)

            ds_i (np.ndarray): Shift of the coordinates. Δx_i ,
                               shape = (3*n_atoms,)
        """

        logger.info('Updating the Hessian with the BFGS scheme')

        self.bfgs_updater.update(delta_grad=dg_i, delta_x=ds_i)
        self.coords.H = self.bfgs_updater.B.copy()

        return None

    def step(self, max_step=0.05):
        """
        Do a single RFO step

        Raises:
            (autode.exceptions.OptimisationFailed):
        """
        h_n, _ = self.coords.H.shape

        # Form the augmented Hessian, structure from ref [1], eqn. (56)
        aug_H = np.zeros(shape=(h_n+1, h_n+1))

        aug_H[:h_n, :h_n] = self.coords.H
        aug_H[-1, :h_n] = self.coords.g
        aug_H[:h_n, -1] = self.coords.g

        aug_H_lmda, aug_H_v = np.linalg.eigh(aug_H)
        # A RF step uses the eigenvector corresponding to the lowest non zero
        # eigenvalue
        mode = np.where(np.abs(aug_H_lmda) > 1E-8)[0][0]
        logger.info(f'Stepping along mode: {mode}')

        # and the step scaled by the final element of the eigenvector
        delta_s = aug_H_v[:-1, mode] / aug_H_v[-1, mode]

        if np.max(np.abs(delta_s)) > max_step:
            logger.warning(f'Maximum component of the step '
                           f'{np.max(np.abs(delta_s)):.4} Å > {max_step:.4f} '
                           f'Å. Scaling down')
            delta_s *= max_step / np.max(np.abs(delta_s))

        self.coords.s = self.coords.s + delta_s

        g_i = self.coords.g.copy()   # Current gradient
        self._gradient()
        self._update_hessian_bfgs(dg_i=self.coords.g - g_i, ds_i=delta_s)

        return None

    def run(self, max_iterations=100):
        """
        Optimise to a minimum

        Raises:
            (autode.exceptions.OptimisationFailed):
        """
        logger.info('Using RFO to optimise')
        self._gradient()
        self._estimate_hessian()

        iteration = 0
        while (np.linalg.norm(self.coords.g) > self.g_tol
               and iteration < max_iterations):

            self.step()

            logger.info(f'Iteration {iteration}  '
                        f'|g| = {np.linalg.norm(self.coords.g):.4f} Ha / Å')
            iteration += 1

        return None

    def __init__(self, species, method,
                 coordinate_type=CartesianCoordinates,
                 g_tol=1E-3):
        """
        Rational function optimiser with BFGS update of a model Hessian

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            g_tol (float): Tolerance on |g| (Ha Å^-1)
        """
        super().__init__(species=species.copy(), method=method)

        self.coords = coordinate_type(species.coordinates)
        self.bfgs_updater = BFGS_updater(exception_strategy='damp_update')
        self.bfgs_updater.initialize(n=len(self.coords.s),
                                     approx_type='hess')

        self.g_tol = g_tol
