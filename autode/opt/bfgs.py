import numpy as np
from abc import ABC
from autode.opt.optimisers import NDOptimiser
from autode.opt.line_search import ArmijoLineSearch


class BFGSOptimiser(NDOptimiser, ABC):

    def __init__(self, maxiter, gtol, etol, step_size=0.2, **kwargs):
        """
        Broyden–Fletcher–Goldfarb–Shanno optimiser. Implementation taken
        from: https://tinyurl.com/526yymsw

        ----------------------------------------------------------------------
        Arguments:
            step_size (float): Length of the initial step to take in the line
                               search. Units of distance

        See Also:

            :py:meth:`NDOptimiser <autode.opt.optimisers.NDOptimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self._init_alpha = step_size

    def _step(self) -> None:
        r"""
        Perform a BFGS step by for each iteration (k) by:


        1. If beyond the first iteration (k > 0):

        .. math::
            H_0 = I_n



        1. Solving

        .. math::

            H_k \boldsymbol{p}_k = - \nabla E

        where H is the Hessian matrix, p is the search direction and
        :math:`\nabla E` is the gradient of the energy with respect to the
        coordinates. On the first iteration :math:`H_0` is either the true
        or exact Hessian.

        2. Performing a (in)exact line search to obtain a suitable step size

        .. math::

            \alpha_k = \text{arg min} E(X_k + \alpha \boldsymbol{p}_k)

        and setting :math:`s_k = \alpha \boldsymbol{p}_k`, and updating the
        positions accordingly (:math:`X_{k+1} = X_{k} + s_k`)

        """
        p = np.matmul(np.linalg.inv(self._coords.h), -self._coords.g)

        ls = ArmijoLineSearch(direction=p, init_alpha=self._init_alpha)
        ls.run(self._species, self._method, n_cores=self._n_cores)

        s = ls.alpha * p

        return None
