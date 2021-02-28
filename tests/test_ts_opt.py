import autode as ade
from autode.opt.prfo_opt import PRFOptimser


def test_prfo_sn2():

    sn2_tsg = ade.Molecule('sn2_ts_guess.xyz', charge=-1, solvent_name='water')
    optimiser = PRFOptimser(species=sn2_tsg, method=ade.methods.XTB())

    for _ in range(5):
        optimiser.step(calc_hessian=True)

    optimiser.species.print_xyz_file(filename='tmp.xyz')
