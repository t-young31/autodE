from copy import deepcopy
from itertools import combinations
from multiprocessing import Pool
import numpy as np
import os
from scipy.optimize import minimize
from time import time
from cconf_gen import v
from cconf_gen import dvdr
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.config import Config
from autode.input_output import xyz_file_to_atoms
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.mol_graphs import split_mol_across_bond


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds):
    # TODO divide and conquer?

    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)

    init_coords = coords.reshape(3 * n_atoms)
    res = minimize(v, x0=init_coords, args=(bonds, k, d0, c, fixed_bonds), method='CG', tol=tol, jac=dvdr)

    return res.x.reshape(n_atoms, 3)


def get_atoms_rotated_stereocentres(species, atoms, rand):
    """If two stereocentres are bonded, rotate them randomly with respect to each other

    Arguments:
        species (autode.species.Species):
        atoms (list(autode.atoms.Atom)):
        rand (np.RandomState): random state
    """

    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    # Check on every pair of stereocenters
    for (atom_i, atom_j) in combinations(stereocentres, 2):
        if (atom_i, atom_j) in species.graph.edges:

            # Don't rotate if the bond connecting the centers is a π-bond
            if species.graph.edges[atom_i, atom_j]['pi'] is True:
                logger.info('Stereocenters were π bonded – not rotating')
                continue

            left_idxs, _ = split_mol_across_bond(species.graph, bond=(atom_i, atom_j))

            # Rotate the left hand side randomly
            rot_axis = atoms[atom_i].coord - atoms[atom_j].coord
            theta = 2*np.pi*rand.rand()
            [atoms[i].rotate(axis=rot_axis, theta=theta, origin=atoms[atom_i].coord) for i in left_idxs]

    return atoms


def add_dist_consts_across_stereocentres(species, dist_consts):
    """
    Add distances constraints across two bonded stereocentres, for example for a Z alkene, (hopefully) ensuring
    that in the conformer generation the stereochemistry is retained

    Arguments:
        species (autode.species.Species):
        dist_consts (dict): keyed with tuple of atom indexes and valued with the distance (Å), or None
    """
    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    if dist_consts is None:
        dist_consts = {}

    # Check on every pair of stereocenters
    for (atom_i, atom_j) in combinations(stereocentres, 2):

        # If they are not bonded don't alter
        if (atom_i, atom_j) not in species.graph.edges:
            continue

        # Add a single distance constraint between the nearest neighbours of each stereocentre
        for atom_i_neighbour in species.graph.neighbors(atom_i):
            for atom_j_neighbour in species.graph.neighbors(atom_j):
                if atom_i_neighbour != atom_j and atom_j_neighbour != atom_i:

                    # Fix the distance to the current value
                    dist_consts[(atom_i_neighbour, atom_j_neighbour)] = species.get_distance(atom_i_neighbour,
                                                                                             atom_j_neighbour)

    logger.info(f'Have {len(dist_consts)} distance constraint(s)')
    return dist_consts


def get_non_random_atoms(species):
    """Get the atoms that won't be randomised in the conformer generation. Stereocentres and nearest neighbours"""
    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    non_rand_atoms = deepcopy(stereocentres)
    for stereocentre in stereocentres:
        non_rand_atoms += list(species.graph.neighbors(stereocentre))

    if len(non_rand_atoms) > 0:
        logger.info(f'Not randomising atom index(es) {set(non_rand_atoms)}')

    return set(non_rand_atoms)


def get_simanl_atoms(species, dist_consts=None, conf_n=0):
    """V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4

    Arguments:
        species (autode.species.Species): Species, Molecule, TSguess, TS
        dist_consts (dict): Key = tuple of atom indexes, Value = distance
        conf_n (int): Number of this conformer generated

    Returns:
        (np.ndarray): Coordinates of the generated conformer
    """
    xyz_filename = f'{species.name}_conf{conf_n}_siman.xyz'

    for filename in os.listdir(os.getcwd()):
        if filename == xyz_filename:
            logger.info('Conformer has already been generated')
            return xyz_file_to_atoms(filename=filename)

    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()
    atoms = get_atoms_rotated_stereocentres(species=species, atoms=deepcopy(species.atoms), rand=rand)

    # Add the distance constraints as fixed bonds
    d0 = get_ideal_bond_length_matrix(atoms=species.atoms, bonds=species.graph.edges())

    # Add distance constraints across stereocentres e.g. for a Z double bond then modify d0 appropriately
    dist_consts = add_dist_consts_across_stereocentres(species=species, dist_consts=dist_consts)

    constrained_bonds = []
    for bond, length in dist_consts.items():
        i, j = bond
        d0[i, j] = length
        d0[j, i] = length
        constrained_bonds.append(bond)

    # Randomise coordinates
    fixed_atom_indexes = get_non_random_atoms(species=species)
    for i, atom in enumerate(atoms):
        if i in fixed_atom_indexes:
            continue

        atom.coord = rand.uniform(-10.0, 10.0, 3)

    logger.info('Minimising species...')
    st = time()
    coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]), bonds=species.graph.edges,
                                    k=0.1, c=0.01, d0=d0, tol=species.n_atoms/5E4, fixed_bonds=constrained_bonds)
    logger.info(f'                    ... ({time()-st:.3f} s)')

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    # Print an xyz file so rerunning will read the file
    atoms_to_xyz_file(atoms=atoms, filename=xyz_filename)

    return atoms
