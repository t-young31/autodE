import shutil
import autode as ade
import numpy as np
from autode.atoms import Atom
from autode.opt.prfo import PRFOptimser
from autode.opt.base import CartesianCoordinates
from autode.utils import work_in_tmp_dir


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
        self.coords.H = np.array([[2.0, 0.0],
                                  [0.0, -2.0]])
        self._gradient()
        return None

    def _gradient(self):
        """E = x^2 - y^2   -->   (dE/dx)_y = 2x  ; (dE/dy)_x = -2y"""
        x, y = self.coords.x
        self.coords.g = np.array([2.0*x, -2.0*y])
        return None

    def __init__(self, coords, recalc_freq=None):
        # Initialise the optimiser with a temporary species and override coords
        super().__init__(species=ade.Molecule(smiles='O'),
                         method=None,
                         recalc_hess_every=recalc_freq)

        self.coords = CartesianCoordinates(coords)


def test_prfo_2d():
    """Check PRFO on a test 2D surface"""

    optimiser = PRFOptimser2D(coords=np.array([-0.5, -0.2]),
                              recalc_freq=None)

    # Optimising on a test 2D surface should optimise to the 1st order
    # saddle point at (0, 0)
    optimiser.run()

    assert np.linalg.norm(optimiser.coords.x - np.zeros(2)) < 1E-3

    # should also work with hessian updates on every iteration
    optimiser.recalc_hess_every = 1
    optimiser.coords.x = np.array([-0.5, -0.2])
    optimiser.run()

    assert np.linalg.norm(optimiser.coords.x - np.zeros(2)) < 1E-3

    # Should also be able to optimise far from the TS
    optimiser.coords.x = np.array([-2.5, 1.5])
    optimiser.run(max_iterations=100)

    assert np.linalg.norm(optimiser.coords.x - np.zeros(2)) < 1E-3


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_prfo_sn2():
    """Check SN2 TS optimisation in the default coordinate system"""

    # Adaptive path guess with dr_min = 0.05 Å and dr_max = 0.3 Å
    sn2_tsg = ade.Molecule(name='ts_guess', charge=-1, solvent_name='water',
                           atoms=[Atom('F', -2.79848, -0.05881,  0.06084),
                                  Atom('Cl', 1.67378,  0.04738, -0.02793),
                                  Atom('C', -0.47153, -0.00340,  0.01543),
                                  Atom('H', -0.66251, -0.59179, -0.85935),
                                  Atom('H', -0.62843, -0.47591,  0.96434),
                                  Atom('H', -0.68434,  1.04500, -0.04650)])

    assert np.isclose(sn2_tsg.angle(3, 2, 4), 117, atol=1)

    # Don't run the calculation without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    # should be able to optimise to a TS without updating the Hessian for such
    # a simple molecule by only calculating an initial Hessian
    optimiser = PRFOptimser(species=sn2_tsg,
                            method=ade.methods.XTB(),
                            recalc_hess_every=None,
                            g_tol=1E-3)

    optimiser.run(max_iterations=100)

    assert np.linalg.norm(optimiser.coords.g) < 1E-2
    # Carbon should be trigonal planar with approx 120 °
    assert np.isclose(optimiser.species.angle(3, 2, 4), 120, atol=2)

    # optimiser.species.print_xyz_file(filename='tmp.xyz')
