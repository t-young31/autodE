"""
Partitioned rational function optimisation method (P-RFO) TS optimisation
algorithm follows the notation from
1. https://aip.scitation.org/doi/10.1063/1.2104507

-------------------------------------------------------
x : cartesian coordinates
H : Hessian matrix 3N x 3N for N atoms
u : eigenvector of the Hessian matrix # TODO: mass weighted?
f : -g i.e. negative of the gradient
lmda : λ, eigenvectors of the Hessian matrix
gamma : shift factor
"""
import autode as ade
import numpy as np
import autode.exceptions as ex
from autode.log import logger
from autode.opt.base import Optimiser


class PRFOptimser(Optimiser):

    def _hessian(self):
        """Calculate and set the Hessian and gradient"""

        calc = ade.Calculation(name=f'tmp_hess',
                               molecule=self.species,
                               method=self.method,
                               keywords=self.method.keywords.hess,
                               n_cores=ade.Config.n_cores)
        calc.run()
        self.H = calc.get_hessian()
        self.g = calc.get_gradients().flatten()
        calc.clean_up(force=True, everything=True)

        return

    def _gradient(self):
        """Calculate and set the gradient"""
        self.g = super()._get_gradient(coordinates=self.species.coordinates)
        return None

    def _update_hessian_bofill(self, f_j, dx_j):
        """
        Update the Hessian based on the Bofill scheme. eqn. 8, 9 in ref [1]
        """
        h_j = self.H                 # H_j      current Hessian
        f_jp1 = -self.g              # f_j+1
        df_jp1 = f_j - f_jp1         # Δf_j+1 = f_j - f_j+1

        df_hdx = df_jp1 - np.dot(h_j, dx_j)  # Δf_j+1 - H_j Δx_j

        delta_h_sr1 = np.outer(df_hdx, df_hdx) / np.outer(df_hdx, dx_j)

        delta_h_powell = ((np.outer(df_hdx, dx_j.T) + np.outer(dx_j, df_hdx.T))
                          / np.outer(dx_j.T, dx_j)
                          -
                          ((np.outer(df_hdx.T, dx_j) * np.outer(dx_j, dx_j.T))
                          / (np.dot(dx_j.T, dx_j)**2)))

        phi_bofill = (np.dot(df_hdx.T, dx_j)**2
                      / (np.dot(df_hdx.T, df_hdx) * np.dot(dx_j.T, dx_j)))

        delta_h = phi_bofill * delta_h_sr1 + (1 - phi_bofill) * delta_h_powell

        self.H = h_j + delta_h
        return None

    def step(self, calc_gradient=True, calc_hessian=False, max_step=0.05):
        """
        Do a single PRFO step

        Raises:
            (autode.exceptions.OptimisationFailed):
        """
        lmda, u = np.linalg.eigh(self.H)  # Eigenvalues and eigenvectors
        f = -self.g  # force

        if np.min(lmda) > 0:
            logger.warning('Hessian had no negative eigenvalues, cannot '
                           'follow the to a TS')
            raise ex.OptimisationFailed

        # TODO: choose mode (k) with the maximum overlap with bond rearr
        k = 0
        logger.info(f'Following mode 0 with λ={lmda[k]:.3f}')

        u_f = np.array([np.dot(u_i, f) for lmda_i, u_i in zip(lmda, u)])

        # matrix to determine the shift on the lowest eigenvector
        m_p = np.array([[lmda[k], -u_f[k]],
                        [-u_f[k],  0.0]])
        # largest eigenvalue of the above array. eqn. 5 in ref. [1]
        gamma_p = np.linalg.eigvalsh(m_p)[-1]

        # Initialise a blank array to shift the coordinates by
        delta_x = np.zeros(3*self.species.n_atoms)

        # eqn. 7 from ref [1]
        delta_x += (np.dot(u[k], f) * u[k]
                    / (lmda[k] - gamma_p))

        not_k_idx = [idx for idx in range(len(lmda)) if idx != k]

        # diagonals including a trailing zero
        m_n = np.diag(np.concatenate((lmda[not_k_idx], np.zeros(1))))
        m_n[-1, :-1] = -u_f[not_k_idx]     # final row
        m_n[:-1, -1] = -u_f[not_k_idx]     # final column 

        # smallest eigenvalue of the m_n array, which should be the first
        gamma_n = np.linalg.eigvalsh(m_n)[0]

        # remaining part of eqn. 7 from ref [1]
        for i in not_k_idx:
            delta_x += u_f[i] * u[i] / (lmda[i] - gamma_n)

        if np.max(delta_x) > max_step:
            logger.warning(f'Maximum step size = {np.max(delta_x):.4f} Å '
                           f'was above the maximum allowed {max_step:.4f} Å '
                           f'will scale down')
            delta_x *= max_step / np.max(delta_x)

        logger.info(f'Maximum step size = {np.max(delta_x):.4f} Å')

        new_coords = self.species.coordinates + delta_x.reshape(-1, 3)
        self.species.coordinates = new_coords

        if calc_hessian:
            self._hessian()

        elif calc_gradient:
            self._gradient()
            self._update_hessian_bofill(f_j=f, dx_j=delta_x)

        else:
            logger.warning('Did not calculate a gradient after the step')

        return None

    def run(self):
        """
        Optimise to a TS

        Raises:
            (autode.exceptions.OptimisationFailed):
        """

        raise NotImplementedError

    def __init__(self, species, method, g_tol=1E-3):
        """
        Partitioned rational function optimiser initialised from a good guess
        of a TS, with at least one imaginary mode

        Arguments:
            species (autode.species.Species):
        """
        super().__init__(species=species.copy(), method=method)

        self.H = None
        self.g = None
        self.g_tol = g_tol

        # Calculate an initial Hessian
        self._hessian()
