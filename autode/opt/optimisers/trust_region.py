r"""
Trust region methods for performing 1D optimisations, in contrast to line
search methods the direction is optimised rather than the distance in a
particular direction. The sub-problem is:

.. math::

    \min(m_k(p)) = E_k + g_k^T p + \frac{1}{2}p^T H_k p
    \quad : \quad ||p|| \le \Delta_k

where :math:`p` is the search direction, :math:`E_k` is the energy at an
iteration :math:`k`, :math:`g_k` is the gradient and :math:`H` is the Hessian.
"""
from typing import Optional
from autode.opt.optimisers.base import Optimiser


class TrustRegionOptimiser(Optimiser):

    @classmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 n_cores:  Optional[int] = None,
                 coords:   Optional['OptCoordinates'] = None,
                 **kwargs):
        pass

    def _step(self) -> None:
        pass

    def _initialise_run(self) -> None:
        pass

    @property
    def converged(self) -> bool:
        pass