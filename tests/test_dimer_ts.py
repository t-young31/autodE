import autode as ade
from autode.opt.dimer import Dimer


def test_dimer_ts_simple():
    """Test a 'dimer' transition state search for an SN2 reaction"""

    dimer = Dimer(species_1=ade.Molecule('sn2_p1.xyz'),
                  species_2=ade.Molecule('sn2_p2.xyz'),
                  species_mid=ade.Molecule('sn2_midpoint.xyz'),
                  method=ade.methods.XTB())

    for _ in range(10):
        dimer.optimise_rotation(phi_tol=0.08)
        dimer.translate(init_step_size=0.1)

    assert len(dimer.iterations) > 1
    assert dimer.iterations[-1].phi < 0.1

    final_point = ade.Molecule('sn2_p1.xyz')
    final_point.coordinates = dimer.x1
    # TODO: remove this test print
    final_point.print_xyz_file(filename='tmp.xyz')

    dimer.iterations.print_xyz_file(species=dimer._species,
                                    point=1)
