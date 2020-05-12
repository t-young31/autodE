from functools import wraps
import os
import shutil
from tempfile import mkdtemp
from autode.exceptions import NoAtomsInMolecule
from autode.exceptions import NoCalculationOutput
from autode.exceptions import NoConformers
from autode.exceptions import NoMolecularGraph
from autode.log import logger


def work_in(dir_ext):
    """Execute a function in a different directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                logger.info(f'Creating directory to store output files at {dir_path:}')
                os.mkdir(dir_path)

            os.chdir(dir_path)
            func(*args, **kwargs)
            os.chdir(here)

        return wrapped_function
    return func_decorator


def work_in_tmp_dir(filenames_to_copy, kept_file_exts):
    """Execute a function in a temporary directory.

    Arguments:
        filenames_to_copy (list(str)): Filenames to copy to the temp dir
        kept_file_exts (list(str): Filename extensions to copy back from the temp dir

    """

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            tmpdir_path = mkdtemp()
            logger.info(f'Creating tmpdir to work in: {tmpdir_path}')

            logger.info(f'Copying {filenames_to_copy}')
            for filename in filenames_to_copy:
                if filename.endswith('_mol.in'):
                    # MOPAC needs the file to be called this
                    shutil.move(filename, os.path.join(tmpdir_path, 'mol.in'))
                else:
                    shutil.copy(filename, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)

            logger.info('Function   ...running')
            func(*args, **kwargs)
            logger.info('           ...done')

            for filename in os.listdir(tmpdir_path):
                if any([filename.endswith(ext) for ext in kept_file_exts]):
                    logger.info(f'Coping back {filename}')
                    shutil.copy(filename, here)

            os.chdir(here)

            logger.info('Removing temporary directory')
            shutil.rmtree(tmpdir_path)

        return wrapped_function
    return func_decorator


def requires_atoms():
    """A function requiring a number of atoms to run"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'n_atoms')

            if args[0].n_atoms == 0:
                raise NoAtomsInMolecule

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_graph():
    """A function requiring a number of atoms to run"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'graph')

            if args[0].graph is None:
                raise NoMolecularGraph

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_conformers():
    """A function requiring a list of"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'conformers')

            if args[0].conformers is None:
                raise NoConformers

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_output():
    """A function requiring an output file and output file lines"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # Species must be the first argument
            assert hasattr(args[0], 'output_filename')
            if args[0].output_filename is None:
                raise NoCalculationOutput

            assert hasattr(args[0], 'output_file_exists')
            assert hasattr(args[0], 'output_file_lines')
            assert hasattr(args[0], 'rev_output_file_lines')

            if args[0].output_file_exists is False or args[0].output_file_lines is None:
                raise NoCalculationOutput

            return func(*args, **kwargs)

        return wrapped_function

    return func_decorator