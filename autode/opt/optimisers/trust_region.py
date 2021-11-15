r"""
Trust region methods for performing optimisations, in contrast to line
search methods the direction within the 1D problem is optimised rather than
the distance in a particular direction. The sub-problem is:

.. math::

    \min(m_k(p)) = E_k + g_k^T p + \frac{1}{2}p^T H_k p
    \quad : \quad ||p|| \le \alpha_k

where :math:`p` is the search direction, :math:`E_k` is the energy at an
iteration :math:`k`, :math:`g_k` is the gradient and :math:`H` is the Hessian
and :math:`alpha_k` is the trust radius at step k. Notation follows
https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
with :math:`\Delta \equiv \alpha`
"""
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from autode.log import logger
from autode.opt.optimisers.base import NDOptimiser
from autode.opt import CartesianCoordinates


class TrustRegionOptimiser(NDOptimiser, ABC):

    def __init__(self,
                 maxiter:          int,
                 trust_radius:     float,
                 coords:           Optional['autode.opt.OptCoordinates'] = None,
                 max_trust_radius: Optional[float] = None,
                 eta_1:            float = 0.1,
                 eta_2:            float = 0.25,
                 eta_3:            float = 0.75,
                 t_1:              float = 0.25,
                 t_2:              float = 2.0):
        """Trust radius optimiser"""

        super().__init__(maxiter=maxiter, coords=coords)
        self.alpha = trust_radius
        self.alpha_max = (max_trust_radius if max_trust_radius is not None
                          else 10 * trust_radius)

        # Parameters for the TR optimiser
        self._eta = _Eta(eta_1, eta_2, eta_3)
        self._t = _T(t_1, t_2)

        self.rho: Optional[float] = None        # Actual vs. predicted change
        self.m:   Optional[float] = None        # Energy estimate
        self.p:   Optional[np.ndarray] = None   # Direction

    @classmethod
    def optimise(cls,
                 species:     'autode.species.Species',
                 method:      'autode.wrappers.base.Method',
                 n_cores:      Optional[int] = None,
                 coords:       Optional['autode.opt.OptCoordinates'] = None,
                 maxiter:      int = 5,
                 trust_radius: float = 1.0,
                 **kwargs
                 ) -> None:
        """
        Construct and optimiser using a trust region optimiser
        """

        optimiser = cls(maxiter=maxiter,
                        trust_radius=trust_radius,
                        coords=coords)

        optimiser.run(species, method, n_cores=n_cores)

        return None

    def _step(self) -> None:
        """
        Perform a TR step based on a solution to the 'sub-problem' to find a
        direction which to step in, the distance for which is fixed by the
        current trust region (alpha)

        TODO: Math
        """
        self._solve_subproblem()

        if self.iteration == 0:
            # First iteration, so take a normal step
            self._coords = self._coords + self.p
            return

        if self.rho < self._eta[2]:
            self.alpha *= self._t[1]

        else:
            if self.rho > self._eta[3] and self._step_was_close_to_max:
                self.alpha = min(self._t[2] * self.alpha, self.alpha_max)

            else:
                pass  # No updated required: α_k+1 = α_k

        if self.rho > self._eta[1]:
            self._coords = self._coords + self.p

        else:
            logger.warning('Trust radius step did not result in a satisfactory'
                           ' reduction. Not taking a step')
            self._coords = self._coords.copy()

        return None

    @property
    def _step_was_close_to_max(self) -> bool:
        """
        Is the current step close to the maximum allowed?

        -----------------------------------------------------------------------
        Returns:
            (bool): |p| ~ α_max
        """
        return np.allclose(np.linalg.norm(self.p), self.alpha_max)

    @abstractmethod
    def _solve_subproblem(self) -> None:
        """Solve the TR 'subproblem' for the ideal step to use"""

    @abstractmethod
    def _update_hessian(self) -> None:
        """Solve the TR 'subproblem' for the ideal step to use"""

    def _update_gradient_and_energy(self) -> None:
        """
        Update the gradient and energy, along with a couple of other
        properties, derived from the energy and gradient
        """
        super()._update_gradient_and_energy()
        self._update_hessian()

        if self.iteration > 1:
            self.rho = ((self._history.penultimate.e - self._coords.e)
                        / (self._history.penultimate.e - self.m))

        self.m = (self._coords.e
                  + np.dot(self._coords.g, self.p)
                  + 0.5 * np.dot(self.p, np.matmul(self._coords.h, self.p)))

        return None


class CauchyTROptimiser(TrustRegionOptimiser):
    """Most simple trust-radius optimiser, solving the subproblem with a
    cauchy point calculation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau: Optional[float] = None

    def _initialise_run(self) -> None:
        """Initialise a TR optimiser, so it can take the first step"""

        if self._coords is None:
            self._coords = CartesianCoordinates(self._species.coordinates)

        self._update_gradient_and_energy()
        self._solve_subproblem()
        return None

    def _update_hessian(self) -> None:
        """Hessian is always the identity matrix"""
        self._coords.h = np.eye(len(self._coords))
        return None

    def _solve_subproblem(self) -> None:
        r"""
        Solve for the optimum direction by a Cauchy point calculation

        .. math::

            \tau =
            \begin{cases}
            1 \qquad \text{ if } g^T H g <= 0\\
            \min\left( \frac{|g|^3}{\alpha g^T H g}, 1\right)
            \qquad \text{otherwise}
            \end{cases}

        and

        .. math::

            p = -\tau \frac{\alpha}{|g|} g

        """
        g, h = self._coords.g, self._coords.h
        g_h_g = np.dot(g, np.matmul(h, g))

        if g_h_g <= 0:
            self.tau = 1.0
        else:
            self.tau = min((np.linalg.norm(g)**3 / (self.alpha * g_h_g), 1.0))

        self.p = -self.tau * (self.alpha / np.linalg.norm(g)) * g

        return None


class _ParametersIndexedFromOne:

    def __getitem__(self, item):
        """Internal array is indexed from 1"""
        return self._arr[item - 1]

    def __init__(self, *args):
        """Scalar float arguments"""
        self._arr = [float(arg) for arg in args]


class _Eta(_ParametersIndexedFromOne):
    """η parameters in the TR optimisers"""

    def __init__(self, p1, p2, p3):
        super().__init__(p1, p2, p3)


class _T(_ParametersIndexedFromOne):
    """t parameters in the TR optimisers"""

    def __init__(self, p1, p2):
        super().__init__(p1, p2)
