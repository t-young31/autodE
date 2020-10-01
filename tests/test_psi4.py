from autode.wrappers.PSI4 import PSI4
from autode.atoms import Atom
from autode.calculation import Calculation
from autode.calculation import execute_calc
from autode.species.molecule import Molecule
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoInputError
from autode.exceptions import UnsuppportedCalculationInput
from autode.wrappers.keywords import SinglePointKeywords, OptKeywords
from . import testutils
import pytest
import numpy as np

import os
here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = PSI4()
method.available = True

sp_keywords = SinglePointKeywords(['PBE0-D3BJ', 'def2-TZVP'])
opt_keywords = OptKeywords(['PBE0-D3BJ', 'def2-SVP'])


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'psi4.zip'))
def test_psi4_opt_calculation():

    methylchloride = Molecule(name='CH3Cl',
                              smiles='[H]C([H])(Cl)[H]',
                              solvent_name='water')

    calc = Calculation(name='opt', molecule=methylchloride, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('opt_psi4.inp') is True
    assert os.path.exists('opt_orca.out') is True
    assert len(calc.get_final_atoms()) == 5
    assert -499.735 < calc.get_energy() < -499.730
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.get_imaginary_freqs() == []
    assert calc.input.filename == 'opt_psi4.inp'
    assert calc.output.filename == 'opt_psi4.out'
    assert calc.terminated_normally()

    assert calc.optimisation_converged()

    calc = Calculation(name='opt', molecule=methylchloride, method=method,
                       keywords=opt_keywords)

    # If the calculation is not run with calc.run() then there should be no
    # input and the calc should raise that there is no input

    with pytest.raises(NoInputError):
        execute_calc(calc)


def test_calc_bad_mol():

    class Mol:
        pass

    mol = Mol()

    with pytest.raises(AssertionError):
        Calculation(name='bad_mol_object', molecule=mol, method=method,
                    keywords=opt_keywords)

    mol.atoms = None
    mol.mult = 1
    mol.n_atoms = 0
    mol.charge = 0
    mol.solvent = None

    with pytest.raises(NoInputError):
        Calculation(name='no_atoms_mol', molecule=mol, method=method,
                    keywords=opt_keywords)


def test_bad_psi4_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method,
                       keywords=opt_keywords)
    calc.output.file_lines = []
    calc.output.rev_file_lines = []

    assert calc.get_energy() is None
    with pytest.raises(AtomsNotFound):
        calc.get_final_atoms()

    with pytest.raises(NoInputError):
        calc.execute_calculation()

    calc.output_file_lines = None
    assert calc.terminated_normally() is False


def test_solvation():
    """Solvation not implemented for psi4"""

    methane = Molecule(name='solvated_methane', smiles='C',
                       solvent_name='water')

    with pytest.raises(UnsuppportedCalculationInput):

        # Should raise an unsupported calculation type, the only
        # "supported" implicit solvation type is 'not_supported'
        method.implicit_solvation_type = 'xxx'
        calc = Calculation(name='broken_solvation', molecule=methane,
                           method=method, keywords=sp_keywords)
        calc.run()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'psi4.zip'))
def test_gradients():

    h2 = Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.0)])
    calc = Calculation(name='h2_grad', molecule=h2,
                       method=method,
                       keywords=method.keywords.grad())
    calc.run()
    h2.energy = calc.get_energy()

    delta_r = 1E-8

    # Energy of a finite difference approximation
    h2_disp = Molecule(name='h2_disp',
                       atoms=[Atom('H'), Atom('H', x=1.0 + delta_r)])
    calc = Calculation(name='h2_disp', molecule=h2_disp,
                       method=method,
                       keywords=method.keywords.grad)
    calc.run()
    h2_disp.energy = calc.get_energy()

    delta_energy = h2_disp.energy - h2.energy   # Ha
    grad = delta_energy / delta_r               # Ha A^-1

    calc = Calculation(name='h2_grad', molecule=h2,
                       method=method,
                       keywords=method.keywords.grad)
    calc.run()

    diff = calc.get_gradients()[1, 0] - grad    # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1E-3
