"""
Partitioned rational function optimisation method (P-RFO) TS optimisation
algorithm follows the notation from
1. https://aip.scitation.org/doi/10.1063/1.1515483

-------------------------------------------------------
x : cartesian coordinates
H : Hessian matrix 3N x 3N for N atoms
v : eigenvectors of the Hessian matrix (columns)
g : gradient
lmda : λ, eigenvectors of the Hessian matrix
"""
import autode as ade
import numpy as np
import autode.exceptions as ex
from autode.log import logger
from autode.opt.base import Optimiser
from autode.opt.base import CartesianCoordinates


class PRFOptimser(Optimiser):

    def _hessian(self):
        """Calculate and set the Hessian and gradient"""
        self.species.coordinates = self.coords.x

        calc = ade.Calculation(name=f'tmp_hess',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.hess,
                               n_cores=ade.Config.n_cores)
        calc.run()
        self.species.energy = calc.get_energy()
        self.coords.H = calc.get_hessian()
        self.coords.g = calc.get_gradients().flatten()
        calc.clean_up(force=True, everything=True)

        return None

    def _gradient(self):
        """Calculate and set the gradient"""
        self.coords.g = super()._get_gradient(coordinates=self.coords.x)
        return None

    def _update_hessian_bofill(self, dg_i, dx_i, min_update_tol=1E-6):
        """
        Update the Hessian based on the Bofill scheme. eqn. 42-47 in ref [1]

        Arguments:
            dg_i (np.ndarray): Difference between the current gradient and the
                               previous iteration. Δg_i, shape = (3*n_atoms,)

            dx_i (np.ndarray): Shift of the coordinates. Δx_i ,
                               shape = (3*n_atoms,)

        Keyword Arguments:
            min_update_tol (float): Threshold on |Δg - HΔx| below which the
                                    Hessian will not be updated, to prevent
                                    dividing by zero
        """
        logger.info('Updating the Hessian with the Bofill scheme')

        # from ref. [1] the approximate Hessian (G) is self.H
        G_i_1 = self.coords.H                       # G_{i-1}
        dE_i = dg_i - np.matmul(G_i_1, dx_i).T     # ΔE_i = Δg_i - G_{i-1}Δx_i

        if np.linalg.norm(dE_i) < min_update_tol:
            logger.warning(f'|Δg_i - G_i-1Δx_i| < {min_update_tol:.4f} '
                           f'not updating the Hessian')
            return None

        # G_i^MS eqn. 42 from ref. [1]
        G_i_MS = G_i_1 + np.outer(dE_i, dE_i) / np.dot(dE_i, dx_i)

        # G_i^PBS eqn. 43 from ref. [1]
        dxTdg = np.dot(dx_i, dg_i)
        G_i_PSB = (G_i_1
                   + ((np.outer(dE_i, dx_i) + np.outer(dx_i, dE_i))
                      / np.dot(dx_i, dx_i))
                   - (((dxTdg - np.linalg.multi_dot((dx_i, G_i_1, dx_i)))
                      * np.outer(dx_i, dx_i))
                      / np.dot(dx_i, dx_i)**2)
                   )

        # ϕ from eqn. 46 from ref [1]
        phi_bofill = 1.0 - (np.dot(dx_i, dE_i)**2
                            / (np.dot(dx_i, dx_i)*np.dot(dE_i, dE_i)))

        logger.info(f'ϕ_Bofill = {phi_bofill:.3f}')

        # Setting the Hessian in internal coordinates
        self.coords._H_s = (1.0 - phi_bofill) * G_i_MS + phi_bofill * G_i_PSB
        return None

    def step(self, calc_hessian=False, max_step=0.05, im_mode=0):
        """
        Do a single PRFO step

        Keyword Arguments:
            calc_hessian (bool):
            max_step (float):
            im_mode (int):

        Raises:
            (autode.exceptions.OptimisationFailed):
        """
        # Eigenvalues (\tilde{H}_kk in ref [1]) and eigenvectors (V in ref [1])
        # of the Hessian matrix
        lmda, v = np.linalg.eigh(self.coords.H)

        if np.min(lmda) > 0:
            logger.warning('Hessian had no negative eigenvalues, cannot '
                           'follow to a TS')
            raise ex.OptimisationFailed

        logger.info(f'Maximising along mode {int(im_mode)} with '
                    f'λ={lmda[im_mode]:.4f}')

        # Gradient in the eigenbasis of the Hessian. egn 49 in ref. 50
        g_tilde = np.matmul(v.T, self.coords.g)

        # Initialised step in the Hessian eigenbasis
        s_tilde = np.zeros_like(g_tilde)

        # For a step in Cartesian coordinates the Hessian will have zero
        # eigenvalues for translation/rotation - keep track of them
        non_zero_lmda = np.where(np.abs(lmda) > 1E-8)[0]

        # Augmented Hessian 1 along the imaginary mode to maximise, with the
        # form (see eqn. 59 in ref [1]):
        #  (\tilde{H}_11  \tilde{g}_1) (\tilde{s}_1)  =      (\tilde{s}_1)
        #  (                         ) (           )  =  ν_R (           )
        #  (\tilde{g}_1        0     ) (    1      )         (     1     )
        #
        aug1 = np.array([[lmda[im_mode], g_tilde[im_mode]],
                         [g_tilde[im_mode], 0.0]])
        _, aug1_v = np.linalg.eigh(aug1)

        # component of the step along the imaginary mode is the first element
        # of the eigenvector with the largest eigenvalue (1), scaled by the
        # final element
        s_tilde[im_mode] = aug1_v[0, 1] / aug1_v[1, 1]

        # Augmented Hessian along all other modes with non-zero eigenvalues,
        # that are also not the imaginary mode to be followed
        non_mode_lmda = np.delete(non_zero_lmda, [im_mode])

        # see eqn. 60 in ref. [1] for the structure of this matrix!
        augn = np.diag(np.concatenate((lmda[non_mode_lmda], np.zeros(1))))
        augn[:-1, -1] = g_tilde[non_mode_lmda]
        augn[-1, :-1] = g_tilde[non_mode_lmda]

        _, augn_v = np.linalg.eigh(augn)

        # The step along all other components is then the all but the final
        # component of the eigenvector with the smallest eigenvalue (0)
        s_tilde[non_mode_lmda] = augn_v[:-1, 0] / augn_v[-1, 0]

        # Transform back from the eigenbasis with eqn. 52 in ref [1]
        delta_s = np.matmul(v, s_tilde)

        if np.max(delta_s) > 100 * max_step:
            raise ex.OptimisationFailed('About to perform a huge unreasonable '
                                        'step!')

        if np.max(delta_s) > max_step:
            logger.warning(f'Maximum step size = {np.max(delta_s):.4f} Å '
                           f'was above the maximum allowed {max_step:.4f} Å '
                           f'will scale down')
            delta_s *= max_step / np.max(delta_s)

        logger.info(f'Maximum step size = {np.max(delta_s):.4f} Å')
        self.coords.s = self.coords.s + delta_s

        if calc_hessian:
            self._hessian()

        else:
            g_i = self.coords.g
            self._gradient()
            self._update_hessian_bofill(dg_i=self.coords.g - g_i, dx_i=delta_s)

        return None

    def run(self, max_iterations=100):
        """
        Optimise to a TS

        Raises:
            (autode.exceptions.OptimisationFailed):
        """
        logger.info('Using PRFO to optimise to a TS. Calculating an initial '
                    'Hessian...')
        self._hessian()

        iteration = 0
        while (np.linalg.norm(self.coords.g) > self.g_tol
               and iteration < max_iterations):

            if iteration % self.recalc_hess_every == 0:
                logger.info('Performing a PRFO step & calculating the Hessian')
                self.step(calc_hessian=True)

            else:
                logger.info('Performing a PRFO step')
                self.step()

            logger.info(f'Iteration {iteration}  '
                        f'|g| = {np.linalg.norm(self.coords.g):.4f} Ha / Å')
            iteration += 1

        return None

    def __init__(self, species, method,
                 coordinate_type=CartesianCoordinates,
                 g_tol=1E-3,
                 recalc_hess_every=None):
        """
        Partitioned rational function optimiser initialised from a good guess
        of a TS, with at least one imaginary mode

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            g_tol (float): Tolerance on |g| (Ha Å^-1)

            recalc_hess_every (None | int): Recalculate the Hessian every this
                              number of steps, if None will never update the
                              Hessian
        """
        super().__init__(species=species.copy(), method=method)

        self.coords = coordinate_type(species.coordinates)

        self.g_tol = g_tol

        # Use a large integer if the Hessian is not recalculated at all, to
        # allow for modulo division
        self.recalc_hess_every = (999 if recalc_hess_every is None
                                  else int(recalc_hess_every))
