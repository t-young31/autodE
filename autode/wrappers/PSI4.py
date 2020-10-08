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

# functionals (http://www.psicode.org/psi4manual/master/dft_byfunctional.html) and
# basis sets (http://www.psicode.org/psi4manual/master/basissets_tables.html#apdx-basistables)
# that are supported are listed in the psi4 documentation
psi4_functionals = ['b1lyp', 'b1lyp-d3bj', 'b1pw91', 'b1wc', 'b2gpplyp',
                    'b2gpplyp-d3bj', 'b2gpplyp-nl', 'b2plyp', 'b2plyp-d3bj',
                    'b2plyp-d3mbj', 'b2plyp-nl', 'b3lyp', 'b3lyp-d3bj',
                    'b3lyp-d3mbj', 'b3lyp-nl', 'b3lyp5', 'b3lyps', 'b3p86',
                    'b3p86-d3bj', 'b3pw91', 'b3pw91-d3bj', 'b3pw91-nl',
                    'b5050lyp', 'b86b95', 'b86bpbe', 'b88b95', 'b88b95-d3bj',
                    'b97-0', 'b97-1', 'b97-1-d3bj', 'b97-1p', 'b97-2', 'b97-2-d3bj',
                    'b97-3', 'b97-d', 'b97-d3bj', 'b97-d3mbj', 'b97-gga1', 'b97-k',
                    'b97m-d3bj', 'b97m-v', 'bb1k', 'bhandh', 'bhandhlyp', 'blyp',
                    'blyp-d3bj', 'blyp-d3mbj', 'blyp-nl', 'bmk', 'bmk-d3bj', 'bop',
                    'bop-d3bj', 'bp86', 'bp86-d3bj', 'bp86-d3mbj', 'bp86-nl',
                    'cam-b3lyp', 'cam-b3lyp-d3bj', 'cap0', 'core-dsd-blyp',
                    'core-dsd-blyp-d3bj', 'dldf', 'dldf+d09', 'dldf+d10', 'dsd-blyp',
                    'dsd-blyp-d3bj', 'dsd-blyp-nl', 'dsd-pbeb95', 'dsd-pbeb95-d3bj',
                    'dsd-pbeb95-nl', 'dsd-pbep86', 'dsd-pbep86-d3bj', 'dsd-pbep86-nl',
                    'dsd-pbepbe', 'dsd-pbepbe-d3bj', 'dsd-pbepbe-nl', 'edf1', 'edf2',
                    'ft97', 'gam', 'hcth120', 'hcthteams120-d3bj', 'hcth147', 'hcth407',
                    'hcth407-d3bj', 'hcth407p', 'hcth93', 'hcthp14', 'hcthp76', 'hf',
                    'hf+d', 'hf-d3bj', 'hf-nl', 'hf3c', 'hjs-b88', 'hjs-b97x', 'hjs-pbe',
                    'hjs-pbe-sol', 'hpbeint', 'hse03', 'hse03-d3bj', 'hse06', 'hse06-d3bj',
                    'ksdt', 'kt2', 'lc-vv10', 'lrc-wpbe', 'lrc-wpbeh', 'm05', 'm05-2x',
                    'm06', 'm06-2x', 'm06-hf', 'm06-l', 'm08-hx', 'm08-so', 'm11',
                    'm11-l-d3bj', 'm11-l', 'm11-l-d3bj', 'mb3lyp-rc04', 'mgga_ms0',
                    'mgga_ms1', 'mgga_ms2', 'mgga_ms2h', 'mgga_mvs', 'mgga_mvsh', 'mn12-l',
                    'mn12-l-d3bj', 'mn12-sx', 'mn12-sx-d3bj', 'mn15', 'mn15-d3bj', 'mn15-l',
                    'mohlyp2', 'mohlyp2', 'mp2d', 'mp2mp2', 'mpw1b95', 'mpw1b95-d3bj',
                    'mpw1k', 'mpw1lyp', 'mpw1pbe', 'mpw1pw', 'mpw1pw-d3bj', 'mpw3lyp',
                    'mpw3pw', 'mpwb1k', 'mpwb1k-d3bj', 'mpwlyp1m', 'mpwlyp1w', 'mpwpw',
                    'n12', 'n12-d3bj', 'n12-sx', 'n12-sx-d3bj', 'o3lyp', 'o3lyp-d3bj',
                    'oblyp-d', 'op-pbe', 'opbe-d', 'opwlyp-d', 'otpss-d', 'pbe', 'pbe-d3bj',
                    'pbe-d3mbj', 'pbe-nl', 'pbe0', 'pbe0-13', 'pbe0-2', 'pbe0-d3bj',
                    'pbe0-d3mbj', 'pbe0-dh', 'pbe0-dh-d3bj', 'pbe0-nl', 'pbe1w', 'pbe50',
                    'pbeh3c', 'pbelyp1w', 'pkzb', 'ptpss', 'ptpss-d3bj', 'pw6b95',
                    'pw6b95-d3bj', 'pw86b95', 'pw86pbe', 'pw91', 'pw91-d3bj', 'pwb6k',
                    'pwb6k-d3bj', 'pwpb95', 'pwpb95-d3bj', 'pwpb95-nl', 'revb3lyp',
                    'revm06-l', 'revpbe', 'revpbe-d3bj', 'revpbe-nl', 'revpbe0',
                    'revpbe0-d3bj', 'revpbe0-nl', 'revscan', 'revscan0', 'revtpss',
                    'revtpss-d3bj', 'revtpssh', 'revtpssh-d3bj', 'rpbe', 'rpbe-d3bj',
                    'sb98-1a', 'sb98-1b', 'sb98-1c', 'sb98-2a', 'sb98-2b', 'sb98-2c',
                    'scan', 'scan-d3bj', 'scan0', 'sogga', 'sogga11', 'sogga11-x',
                    'sogga11-x-d3bj', 'svwn', 'teter93', 'th-fc', 'th-fcfo', 'th-fco',
                    'th-fl', 'th1', 'th2', 'th3', 'th4', 'tpss', 'tpss-d3bj', 'tpss-nl',
                    'tpssh', 'tpssh-d3bj', 'tpssh-nl', 'tpsslyp1w', 'tuned-cam-b3lyp', 'vsxc',
                    'vv10', 'wb97', 'wb97m-d3bj', 'wb97m-v', 'wb97x', 'wb97x-d', 'wb97x-d3bj',
                    'wb97x-v', 'wpbe', 'wpbe-d3bj', 'wpbe-d3mbj', 'wpbe0', 'x1b95', 'x3lyp',
                    'x3lyp-d3bj', 'xb1k', 'xlyp', 'xlyp-d3bj', 'zlp']

psi4_basis_sets = ['sto-3g', '3-21g', '6-31g', '6-31g(d)', '6-31g(d,p)', '6-311g',
                   '6-311g(d)', '6-311g(d,p)', '6-311g(2d)', '6-311g(2d,p)',
                   '6-311g(2d,2p)', '6-311g(2df)', '6-311g(2df,p)', '6-311g(2df,2pd)',
                   '6-311g(3df)', '6-311g(3df,p)', '6-311g(3df,2p)', '6-311g(3df,2pd)',
                   '6-311g(3df,3pd)', 'cc-pvxz', 'cc-pv(x+d)z', 'cc-pcvxz', 'cc-pcv(x+d)z',
                   'cc-pwcvxz', 'cc-pwcv(x+d)z', 'cc-pvxz-dk', 'cc-pv(x+d)z-dk',
                   'cc-pcvxz-dk', 'cc-pcv(x+d)z-dk', 'cc-pwcvxz-dk', 'cc-pwcv(x+d)z-dk',
                   'cc-pvxz-f12', 'cc-pvxz-jkfit', 'cc-pv(x+d)z-jkfit', 'cc-pcvxz-jkfit',
                   'cc-pcv(x+d)z-jkfit', 'cc-pwcvxz-jkfit', 'cc-pwcv(x+d)z-jkfit',
                   'cc-pvxz-ri', 'cc-pv(x+d)z-ri', 'cc-pcvxz-ri', 'cc-pcv(x+d)z-ri',
                   'cc-pwcvxz-ri', 'cc-pwcv(x+d)z-ri', 'cc-pvxz-dual', 'cc-pv(x+d)z-dual',
                   'cc-pcvxz-dual', 'cc-pcv(x+d)z-dual', 'cc-pwcvxz-dual',
                   'cc-pwcv(x+d)z-dual', 'pcseg-n', 'aug-pcseg-n', 'pcsseg-n', 'aug-pcsseg-n',
                   'nzapa-nr', 'dzp', 'tz2p', 'tz2pf', 'sadlej-lpol-ds', 'sadlej-lpol-dl',
                   'sadlej-lpol-fs', 'sadlej-lpol-fl']


class PSI4(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        with open(calc.input.filename, 'w') as inp_file:
            print(f'# {calc.name}, invoked by autodE \n', file=inp_file)

            print(f'molecule  {molecule.name} ', '{\n', file=inp_file)
            print(molecule.charge, molecule.mult, file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                      file=inp_file)
            print('}\n', file=inp_file)

            for keyword in calc.input.keywords:
                if keyword.lower() in psi4_basis_sets:
                    print(f'set basis {keyword}', file=inp_file)
                if keyword.lower() in psi4_functionals and isinstance(calc.input.keywords, SinglePointKeywords):
                    print(f'energy({keyword})', file=inp_file)
                if keyword.lower() in psi4_functionals and isinstance(calc.input.keywords, OptKeywords):
                    print(f'optimize({keyword})', file=inp_file)

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
        return None

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
