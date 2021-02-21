import numpy as np
import autode as ade


def energy(array, species, method, coords):
    """
    Calculate the energy given some internal coordinates

    -------------------------------------------------------------------------
    Arguments:
        array (np.ndarray): Array of coordinates e.g. delocalised internals
                            shape = (M,)

        species (autode.species.Species):

        method (autode.wrappers.base.ElectronicStructureMethod):

        coords (autode.opt.internals.InternalCoordinates):

    Returns:
        (float): Energy in Ha
    """
    coords.s = array

    species.coordinates = coords.x   # Set the new cartesians

    # Run the calculation
    calc = ade.Calculation(name='tmp',
                           molecule=species,
                           method=method,
                           keywords=method.keywords.grad,
                           n_cores=ade.Config.n_cores)
    calc.run()

    # Convert the Cartesian gradient into internal coordinates
    coords.g = np.matmul(coords.B_T_inv.T, calc.get_gradients().flatten())
    species.energy = calc.get_energy()

    calc.clean_up(force=True, everything=True)

    return species.energy


def gradient(array, species, method, coords):
    """
    Calculate the energy given some internal coordinates

    -------------------------------------------------------------------------
    Arguments:
        array (np.ndarray): Array of coordinates e.g. delocalised internals
                            shape = (M,)

        species (autode.species.Species):

        method (autode.wrappers.base.ElectronicStructureMethod):

        coords (autode.opt.internals.InternalCoordinates):

    Returns:
        (np.ndarray): Gradient in Ha / Å shape = (M,)
    """
    if coords.g is None:
        raise RuntimeError('Gradient should be calculated and set in energy()')

    # TODO a better implementation of the below
    # if not np.isclose(array, coords.s):
    #     raise RuntimeError('Internal coordinates must be the same as with '
    #                        'used for the energy evaluation')

    return coords.g
