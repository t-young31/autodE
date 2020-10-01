import numpy as np
from autode.utils import run_external
from autode.wrappers.base import ElectronicStructureMethod
from autode.exceptions import UnsuppportedCalculationInput
from autode.atoms import Atom
from autode.config import Config
from autode.log import logger
from autode.utils import work_in_tmp_dir
from autode.exceptions import AtomsNotFound
import os


def add_solvent_keyword(calc_input, implicit_solv_type):
    """Will raine UnsupportedCalculationInput as solvent is not
       implemented for psi4"""

    if implicit_solv_type.lower() not in [calc_input.keywords.sp]:
        raise UnsuppportedCalculationInput

    return


def get_keywords(calc_input, molecule, implicit_solvation_type):
    """Modify the keywords for this calculation with the solvent + fix for
    single atom optimisation calls"""

    keywords = calc_input.keywords.copy()

    for keyword in keywords:
        if 'opt' in keyword.lower() and molecule.n_atoms == 1:
            logger.warning('Can\'t optimise a single atom')
            keywords.remove(keyword)  # ORCA defaults to a SP calc

    if calc_input.solvent is not None:
        add_solvent_keyword(calc_input, implicit_solvation_type)

    return keywords


class PSI4(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        keywords = get_keywords(calc.input, molecule,
                                self.implicit_solvation_type)

        with open(calc.input.filename, 'w') as inp_file:
            print(f'# {calc.name}, invoked by autodE \n', file=inp_file)
            print(f'# keywords requested: {keywords}', file=inp_file)

            print(f'molecule  {molecule.name} ', '{\n')
            print(molecule.charge, molecule.mult, file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                      file=inp_file)
            print('}\n')
            
            print(f'set basis {calc.input.keywords.sp[1]}')
            print(f'energy({calc.input.keywords.sp[0]})')

        return None

    def get_input_filename(self, calc):
        """Input and output files can have arbitrary extensions for PSI4,
        we will use .inp for input and .out for output. PSI4 is invoked by:
        psi4 input_file.inp output_file.out"""
        return f'{calc.name}.inp'

    def get_output_filename(self, calc):
        return f'{calc.name}.out'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.inp', '.out', '.dat'))
        def execute_psi4():
            logger.info(f'Setting the number of OMP threads to {calc.n_cores}')
            os.environ['OMP_NUM_THREADS'] = str(calc.n_cores)
            run_external(params=[calc.method.path, calc.input.filename],
                         output_filename=calc.output.filename)

        execute_psi4()
        raise None

    def calculation_terminated_normally(self, calc):
        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if 'Psi4 exiting successfully' in line:
                logger.info('PSI4 exited successfully.')
                return True
            elif 'Psi4 encountered an error.' in line:
                logger.info('PSI4 encountered an error.')
                return False
            elif n_line > 10:
                # Message saying whether the job was successful is
                # usually in the last few lines in PSI4.
                logger.info('Do not know whether PSI4 exited successfully.')
                return False
        return False

    def get_energy(self, calc):
        """Output is in Eh."""
        for line in reversed(calc.output.file_lines):
            if '@' in line and 'Final Energy' in line:
                return float(line.split()[-1])
        logger.info('Could not find energy from psi4.')
        return None

    def get_zero_point_energy(self, calc):
        """frequency() has to be used in the psi4 input file in order
        for this function to work. Output is in Eh."""
        for line in reversed(calc.output.file_lines):
            if 'Total ZPE' in line and 'Electronic energy' in line:
                return float(line.split()[-2])
        logger.info('Could not find zero point energy from psi4.')
        return None

    def get_enthalpy(self, calc):
        """frequency() has to be used in the psi4 input file in order
        for this function to work. Output is in Eh."""
        for line in reversed(calc.output.file_lines):
            if 'Total H' in line and 'Enthalpy' in line:
                return float(line.split()[-2])
        logger.info('Could not find enthalpy in the psi4 output file.')
        raise None

    def get_free_energy(self, calc):
        """frequency() has to be used in the psi4 input file in order
        for this function to work. Output is in Eh."""
        for line in reversed(calc.output.file_lines):
            if 'Total G' in line and 'Free enthalpy' in line:
                return float(line.split()[-2])
        logger.info('Could not find Gibbs free energy from psi4.')
        return None

    def optimisation_converged(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'Energy and wave function converged.' in line:
                return True
            # include a block to return False if a keyword is found that
            # says energy did not converge - I do not know the keyword yet
        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imaginary_freqs(self, calc):
        """frequencies() needs to be used in the psi4 input file."""
        imaginary_frequencies = []

        for line in calc.output.file_lines:
            if 'post-proj' in line and 'i' in line and '[cm^-1]' in line:
                imaginary_frequencies.append(line.split()[3])

        if len(imaginary_frequencies) > 0:
            logger.info(f'Found imaginary frequencies {imaginary_frequencies}')
            return imaginary_frequencies

        else:
            logger.info('Could not find any imaginary frequencies.')
            return None

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def get_final_atoms(self, calc):
        atoms = []
        optimized = False
        section = False
        for line in calc.output.file_lines:
            if 'Optimization is complete!' in line:
                optimized = True
                continue

            if optimized and 'Cartesian geometry  (in Angstrom)' in line:
                section = True
                continue

            if optimized and section and len(line.split()) == 4:
                try:
                    atom_label, x, y, z = line.split()
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))
                except ValueError:
                    pass

            if optimized and 'Saving final (previous) structure.' in line:
                return atoms

        logger.info('Could not get the optimised geometry from psi4.')
        return AtomsNotFound

    def get_atomic_charges(self, calc):
        raise NotImplementedError

    def get_gradients(self, calc):
        """optimise() needs to be used in the psi4 input file."""
        gradients = []
        gradient_section = False
        line_in_section = 0

        for line in calc.output.file_lines:
            if '-Total Gradient:' in line:
                gradient_section = True

            if gradient_section and '*** tstop() called on' in line:
                gradient_section = False
                line_in_section = 0

            if gradient_section:
                line_in_section += 1

            if line_in_section > 3 and line_in_section < (calc.molecule.n_atoms+4):
                x, y, z = line.split()[1:]
                gradients.append(np.array([float(x), float(y), float(z)]))

        return np.array(gradients)

    def __init__(self):
        super().__init__(name='psi4', path=Config.PSI4.path,
                         keywords_set=Config.PSI4.keywords,
                         implicit_solvation_type=Config.PSI4.implicit_solvation_type)


psi4 = PSI4()
