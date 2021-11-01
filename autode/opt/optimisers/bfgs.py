import numpy as np
from abc import ABC
from typing import Type, Optional
from autode.log import logger
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.optimisers.line_search import (LineSearchOptimiser,
                                               ArmijoLineSearch)


class BFGSOptimiser(NDOptimiser, ABC):

    def __init__(self,
                 maxiter:          int,
                 gtol:             'autode.values.GradientNorm',
                 etol:             'autode.values.PotentialEnergy',
                 init_alpha:       float = 1.0,
                 line_search_type: Type[LineSearchOptimiser] = ArmijoLineSearch,
                 **kwargs):
        """
        Broyden–Fletcher–Goldfarb–Shanno optimiser. Implementation taken
        from: https://tinyurl.com/526yymsw

        ----------------------------------------------------------------------
        Arguments:
            init_alpha (float): Length of the initial step to take in the line
                               search. Units of distance

        See Also:

            :py:meth:`NDOptimiser <autode.opt.optimisers.base.NDOptimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self._line_search_type = line_search_type
        self._alpha = init_alpha

    def _step(self) -> None:
        r"""
        Perform a BFGS step. Requires an initial guess of the Hessian matrix
        i.e. (self._coords.h must be defined). Steps follow:

        1. Determine the inverse Hessian:

        :py:meth:`h_inv <autode.opt.optimisers.bfgs.BFGSOptimiser._update_h_inv>`

        2. Determine the search direction with:

        .. math::

             \boldsymbol{p}_k = - H_k^{-1} \nabla E

        where H is the Hessian matrix, p is the search direction and
        :math:`\nabla E` is the gradient of the energy with respect to the
        coordinates. On the first iteration :math:`H_0` is either the true
        or exact Hessian.

        3. Performing a (in)exact line search to obtain a suitable step size

        .. math::

            \alpha_k = \text{arg min} E(X_k + \alpha \boldsymbol{p}_k)

        and setting :math:`s_k = \alpha \boldsymbol{p}_k`, and updating the
        positions accordingly (:math:`X_{k+1} = X_{k} + s_k`)
        """
        self._update_h_inv()

        p = np.matmul(self._coords.h_inv, -self._coords.g)

        logger.info('Performing a line search')
        ls = self._line_search_type(direction=p,
                                    init_alpha=self._alpha,
                                    coords=self._coords.copy())

        ls.run(self._species, self._method, n_cores=self._n_cores)

        self._coords = self._coords + ls.alpha * p
        return None

    def _update_h_inv(self) -> None:
        """
        Update the inverse of the Hessian matrix :math:`H^{-1}` for the
        current set of coordinates. If the first iteration then use the true
        inverse of the (estimated) Hessian, otherwise update the inverse
        """

        if self.iteration == 0:
            logger.info('First iteration so using exact inverse, H^-1')
            return

        coords_l, coords_k = self._coords, self._history.penultimate

        y_k = (coords_l.g - coords_k.g)
        s_k = (coords_l - coords_k)
        h_inv_k = coords_k.h_inv

        if np.linalg.norm(s_k) < 1E-10:
            logger.warning('No update needed - little shift to coordinates:'
                           f'|x_k - x_k-1| = {np.linalg.norm(s_k)}')
            return

        if np.dot(y_k, s_k) < 0:
            logger.warning('Secant condition not satisfied. Skipping H update')
            coords_l.h_inv = h_inv_k
            return

        coords_l.h_inv = self._sherman_morrison_h_inv(h_inv_k, s_k, y_k)
        return None

    def _sherman_morrison_h_inv(self, h_inv_k, s_k, y_k):
        r"""
        Sherman–Morrison inverse matrix update

        .. math::

            H_l^{-1} = H_k^{-1} +
                       \frac{(s_k^Ty_k + y_k^T H_k^{-1} y_k) s_k^T s_k}
                            {s_k^T y_k} -
                        \frac{H_k^{-1} y_k s_k^T + s_k y_k^T H_k^{-1}}
                             {s_k^T y_k}

        where :math:`k = l - 1,\; s_k = x_l - x_k,\; \boldsymbol{y}_l =
        \nabla E_l - \nabla E_k`.

        ----------------------------------------------------------------------
        Arguments:
              h_inv_k (np.ndarray): Inverse Hessian shape = (N, N)

              s_k (np.ndarray): Coordinate shift. shape = (N,)

              y_k (np.ndarray): Gradient shift. shape = (N, )
        """
        logger.info('Updating H^(-1) with Sherman–Morrison formula')

        s_y = np.dot(s_k, y_k)
        y_h_inv_y = np.dot(y_k, np.matmul(h_inv_k, y_k))
        s_s = np.outer(s_k, s_k)
        h_inv_y_s = np.matmul(h_inv_k, np.outer(y_k, s_k))
        s_y_h_inv = np.outer(s_k, np.matmul(y_k, h_inv_k))

        h_inv_l = (h_inv_k
                   + (s_y + y_h_inv_y)/(s_y**2) * s_s
                   - (h_inv_y_s + s_y_h_inv)/ s_y)

        return h_inv_l
