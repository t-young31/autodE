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


class PRFOptimser(Optimiser):

    def _hessian(self):
        """Calculate and set the Hessian and gradient"""
        self.species.coordinates = self.x

        calc = ade.Calculation(name=f'tmp_hess',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.hess,
                               n_cores=ade.Config.n_cores)
        calc.run()
        self.H = calc.get_hessian()
        self.g = calc.get_gradients().flatten()
        calc.clean_up(force=True, everything=True)

        return None

    def _gradient(self):
        """Calculate and set the gradient"""
        self.g = super()._get_gradient(coordinates=self.x)
        return None

    def _update_hessian_bofill(self, f_j, dx_j):
        """
        Update the Hessian based on the Bofill scheme. eqn. 8, 9 in ref [1]

        Arguments:
            f_j (np.ndarray): Force on the previous iteration.
                              shape = (3*n_atoms,)

            dx_j (np.ndarray): Shift of the coordinates. shape = (3*n_atoms,)
        """
        raise NotImplementedError
        logger.info('Updating the Hessian with the Bofill scheme')

        h_j = self.H                 # H_j      current Hessian
        f_jp1 = -self.g              # f_j+1
        df_jp1 = f_j - f_jp1         # Δf_j+1 = f_j - f_j+1

        df_hdx = df_jp1 - np.dot(h_j, dx_j)  # Δf_j+1 - H_j Δx_j

        delta_h_sr1 = np.outer(df_hdx, df_hdx) / np.dot(df_hdx, dx_j)

        delta_h_powell = ((np.outer(df_hdx, dx_j.T) + np.outer(dx_j, df_hdx.T))
                          / np.outer(dx_j.T, dx_j)
                          -
                          ((np.outer(df_hdx.T, dx_j) * np.outer(dx_j, dx_j.T))
                          / (np.dot(dx_j.T, dx_j)**2)))

        phi_bofill = (np.dot(df_hdx.T, dx_j)**2
                      / (np.dot(df_hdx.T, df_hdx) * np.dot(dx_j.T, dx_j)))

        logger.info(f'ϕ_Bofill = {phi_bofill:.3f}')
        delta_h = phi_bofill * delta_h_sr1 + (1 - phi_bofill) * delta_h_powell

        self.H = h_j + delta_h
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
        lmda, v = np.linalg.eigh(self.H)

        if np.min(lmda) > 0:
            logger.warning('Hessian had no negative eigenvalues, cannot '
                           'follow to a TS')
            raise ex.OptimisationFailed

        logger.info(f'Maximising along mode {int(im_mode)} with '
                    f'λ={lmda[im_mode]:.4f}')

        # Gradient in the eigenbasis of the Hessian. egn 49 in ref. 50
        g_tilde = np.matmul(v.T, self.g)

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
        delta_x = np.matmul(v, s_tilde)

        if np.max(delta_x) > 100 * max_step:
            raise ex.OptimisationFailed('About to perform a huge unreasonable '
                                        'step!')

        if np.max(delta_x) > max_step:
            logger.warning(f'Maximum step size = {np.max(delta_x):.4f} Å '
                           f'was above the maximum allowed {max_step:.4f} Å '
                           f'will scale down')
            delta_x *= max_step / np.max(delta_x)

        logger.info(f'Maximum step size = {np.max(delta_x):.4f} Å')
        self.x += delta_x

        if calc_hessian:
            self._hessian()

        else:
            f = -self.g
            self._gradient()
            self._update_hessian_bofill(f_j=f, dx_j=delta_x)

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
        while (np.linalg.norm(self.g) > self.g_tol
               and iteration < max_iterations):

            if iteration % self.recalc_hess_every == 0:
                logger.info('Performing a PRFO step & calculating the Hessian')
                self.step(calc_hessian=True)

            else:
                logger.info('Performing a PRFO step')
                self.step()

            logger.info(f'Iteration {iteration}  '
                        f'|g| = {np.linalg.norm(self.g):.4f} Ha / Å')
            iteration += 1

        return None

    def __init__(self, species, method, g_tol=1E-3, recalc_hess_every=None):
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

        self.x = species.coordinates.flatten()
        self.H = None
        self.g = None

        self.g_tol = g_tol

        # Use a large integer if the Hessian is not recalculated at all, to
        # allow for modulo division
        self.recalc_hess_every = (999 if recalc_hess_every is None
                                  else int(recalc_hess_every))
