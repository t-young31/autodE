import autode as ade
import numpy as np
from autode.opt.prfo_opt import PRFOptimser


class PRFOptimser2D(PRFOptimser):
    """Partitioned rational function optimisation on a 2D PES that can be
    easily visualised using the surface

    E = x^2 - y^2

    (as  Dimer2D)
    """
    def _hessian(self):
        """E = x^2 - y^2   -->   (d^2E/dx2)_y = 2  ; (d^2E/dy^2)_x = -2
        d^2E/dxdy = d^2E/dydx  = 0
        """
        self.H = np.array([[2.0, 0.0],
                           [0.0, -2.0]])
        self._gradient()
        return None

    def _gradient(self):
        """E = x^2 - y^2   -->   (dE/dx)_y = 2x  ; (dE/dy)_x = -2y"""
        x, y = self.x
        self.g = np.array([2.0*x, -2.0*y])
        return None

    def __init__(self, coords, recalc_freq=None):
        # Initialise the optimiser with a temporary species and override coords
        super().__init__(species=ade.Molecule(smiles='O'),
                         method=None,
                         recalc_hess_every=recalc_freq)

        self.x = coords


def test_prfo_2d():

    optimiser = PRFOptimser2D(coords=np.array([-0.5, -0.2]),
                              recalc_freq=1)

    # Optimising on a test 2D surface should optimise to the 1st order
    # saddle point at (0, 0)
    optimiser.run()

    assert np.linalg.norm(optimiser.x - np.zeros(2)) < 1E-3

    # should also work with hessian updates on every iteration
    optimiser.recalc_hess_every = 1
    optimiser.x = np.array([-0.5, -0.2])
    optimiser.run()

    assert np.linalg.norm(optimiser.x - np.zeros(2)) < 1E-3

    # Should also be able to optimise far from the TS
    optimiser.x = np.array([-2.5, 1.5])
    optimiser.run(max_iterations=100)

    assert np.linalg.norm(optimiser.x - np.zeros(2)) < 1E-3


def test_prfo_sn2():

    sn2_tsg = ade.Molecule('sn2_ts_guess.xyz', charge=-1, solvent_name='water')
    optimiser = PRFOptimser(species=sn2_tsg,
                            method=ade.methods.XTB(),
                            recalc_hess_every=1,
                            g_tol=1E-2)

    optimiser.run(max_iterations=100)

    optimiser.species.print_xyz_file(filename='tmp.xyz')
