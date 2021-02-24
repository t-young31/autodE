import autode as ade


def test_dimer_ts():
    """Test a 'dimer' transition state search for an SN2 reaction"""

    species_1 = ade.Molecule('sn2_p1.xyz')
    species_2 = ade.Molecule('sn2_p2.xyz')

    species_0 = ade.Molecule('sn2_midpoint.xyz')




