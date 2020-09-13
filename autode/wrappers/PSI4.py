from autode.wrappers.base import ElectronicStructureMethod
from autode.log import logger


class PSI4(ElectronicStructureMethod):

    def generate_input(self, calc, molecule): # NOT FINISHED

        with open(calc.input.filename, 'w') as inp_file:
            print(f'invoked by autodE\n', file=inp_file)
            print(f'molecule  {molecule.name} ', '{')
            # example:
            #   O
            #   H 1 0.96
            #   H 1 0.96 2 104.5
            print('}\n')
            # example:
            # set basis cc-pVDZ
            # optimize('scf')


        raise NotImplementedError

    # input and output files can have arbitrary extensions for PSI4, we will use .inp for input and .out for output
    # PSI4 is invoked by: 'psi4 input_file.inp output_file.out'
    def get_input_filename(self,calc):
        return f'{calc.name}.inp'

    def get_output_filename(self,calc):
        return f'{calc.name}.out'

    def execute(self,calc):
        raise NotImplementedError

    def calculation_terminated_normally(self,calc):
        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if 'Psi4 exiting succesfully' in line or 'Buy a developer a beer!' in line:
                logger.info('PSI4 exited successfully.')
                return True
            elif 'Psi4 encountered an error.' in line or 'Buy a developer more coffee!' in line:
                logger.info('PSI4 encountered an error.')
                return False
            elif n_line > 10:
                # Message saying whether the job was successfull is usually in the last few lines in PSI4
                logger.info('Could not determine whether PSI4 exited successfully.')
                return False

    def get_energy(self,calc):
        'Output is in Eh.'
        for line in reversed(calc.output.file_lines):
            if '@' in line and 'Final Energy' in line:
                return float(line.split()[-1])
        logger.info('Could not find energy in psi4 output file.')
        return NoCalculationOutput

    def get_zero_point_energy(self,calc):
        'frequency() has to be used in the psi4 input file in order for this function to work. output is in Eh.'
        for line in reversed(calc.output.file_lines):
            if 'Total ZPE' in line and 'Electronic energy' in line:
                return float(line.split()[-2])
        logger.info('Could not find zero point energy in the psi4 output file.')
        return NoCalculationOutput

    def get_enthalpy(self,calc):
        'frequency() has to be used in the psi4 input file in order for this function to work. Output is in Eh.'
        for line in reversed(calc.output.file_lines):
            if 'Total H' in line and 'Enthalpy' in line:
                return float(line.split()[-2])
        logger.info('Could not find enthalpy in the psi4 output file.')
        raise NoCalculationOutput

    def get_free_energy(self,calc):
        'frequency() has to be used in the psi4 input file in order for this function to work. Output is in Eh.'
        for line in reversed(calc.output.file_lines):
            if 'Total G' in line and 'Free enthalpy' in line:
                return float(line.split()[-2])
        logger.info('Could not find Gibbs free energy in the psi4 output file.')
        return NoCalculationOutput

    def optimisation_converged(self,calc):
        for line in reversed(calc.output.file_lines):
            if 'Energy and wave function converged.' in line:
                return True
            # include a block to return False if a keyword is found that says energy did not converge - I do not know the keyword yet
        return False

    def optimisation_nearly_converged(self,calc):
        raise NotImplementedError

    def get_imaginary_freqs(self,calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self,calc,mode_number):
        raise NotImplementedError

    def get_final_atoms(self,calc):
        atoms = []
        optimized = False
        for line in calc.output.file_lines:
            if 'Optimization is complete!' in line:
                optimized = True

            elif optimized:
                if len(line.split() == 4):
                    try:
                        atom_label, x, y, z = line.split()
                        atoms.append(Atom(atom_label, x=float(x), y=float(y), z=float(z)))  # converting to float should raise an error if line does not contain coordinates
                        return atoms
                    except:
                        pass

        logger.info('Could not get the optimised geometry from the psi4 output file.')
        return AtomsNotFound

    def get_atomic_charges(self,calc):
        raise NotImplementedError

    def get_gradients(self,calc):
        raise NotImplementedError

    def __init__(self):
        super().__init__(name='psi4', path=Config.PSI4.path,
                         keywords_set=Config.PSI4.keywords,
                         implicit_solvation_type=Config.PSI4.implicit_solvation_type)

psi4 = PSI4()
