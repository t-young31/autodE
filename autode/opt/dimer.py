"""
Dimer method for finding transition states given two points on the PES.
Notation follows
https://aip.scitation.org/doi/10.1063/1.2815812
based on
1. https://aip.scitation.org/doi/10.1063/1.2104507


x : Cartesian coordinates
g : gradient in cartesian coordinates
"""
import autode as ade
import numpy as np


def get_ts_dimer(species_1, species_2, species_mid, method):
    """
    Generate a transition state given three points on the PES, one either side
    of a peak and a midpoint (species_mid)

    Arguments:
        method (autode.wrappers.base.ElectronicStructureMethod):
        species_1: (ade.species.Species)
        species_2: (ade.species.Species)
        species_mid: (ade.species.Species)

    Returns:
        (ade.transition_states.TransitionState):
    """
    x0 = species_mid.cooordinates.flatten()
    x1, x2 = species_1.coordinates.flatten(), species_2.coordinates.flatten()

    g1, g0 = get_gradient(species_1, method), get_gradient(species_mid, method)

    tau = (x1 - x2) / 2.0
    tau_hat = tau / np.linalg.norm(tau)

    f_r = -2.0*(g1 - g0) + 2.0*(np.dot((g1 - g0), tau_hat))*tau_hat

    theta = f_r / np.linalg.norm(f_r)
    delta = np.linalg.norm(x1 - x2) / 2.0

    # Curvature
    c_tau = np.dot((g1 - g0), tau_hat) / delta
    # and derivative with respect to the rotation (with ϕ=0)
    dc_tau_dphi = 2.0 * np.dot((g1 - g0), theta) / delta

    phi_1 = -0.5 * np.arctan(dc_tau_dphi / (2.0 * np.linalg.norm(c_tau)))

    # eq 11 is recursive?!!!

    phi_min = 0.5 * np.arctan(b1 / a1)

    # and rotate the dimer apprpriately
    x1 = x0 + delta * (tau_hat * np.cos(phi_min) + theta * np.sin(phi_min))
    x2 = x0 - delta * (tau_hat * np.cos(phi_min) + theta * np.sin(phi_min))

    return None


def get_gradient(species, method):
    """
    Get the gradient for a species

    Arguments:
        species: (ade.species.Species)
        method: (autode.wrappers.base.ElectronicStructureMethod):

    Returns:
        (np.ndarray): Gradient shape = (3*n_atoms,)  (Ha / Å)
    """
    calc = ade.Calculation(name='tmp_grad',
                           molecule=species,
                           method=method,
                           keywords=method.keywords.grad,
                           n_cores=ade.Config.n_cores)
    calc.run()
    grad = calc.get_gradients()
    calc.clean_up(force=True, everything=True)

    return grad.flatten()
