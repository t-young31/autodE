import numpy as np


def xyz2coord(xyzs):
    """
    For a set of xyzs in the form e.g [[C, 0.0, 0.0, 0.0], ...] convert to a np array of coordinates, containing just
    just the x, y, z coordinates
    :param xyzs: List of xyzs
    :return: numpy array of coords
    """
    if isinstance(xyzs[0], list):
        return np.array([np.array(line[1:4]) for line in xyzs])
    else:
        return np.array(xyzs[1:4])


def coords2xyzs(coords, old_xyzs):
    """
    Insert a set of coordinates into a set of xyzs to a new list of xyzs
    :param coords: list(nd.array)
    :param old_xyzs: (list(list))
    :return: (list(list))
    """
    assert len(old_xyzs) == len(coords)
    return [[old_xyzs[n][0]] + coords[n].tolist() for n in range(len(old_xyzs))]


def calc_distance_matrix(xyzs):
    """
    Calculate a distance matrix
    :param xyzs: List of xyzs
    :return:
    """

    n_atoms = len(xyzs)
    coords = xyz2coord(xyzs)
    distance_matrix = np.zeros([n_atoms, n_atoms])

    for atom_i in range(n_atoms):
        for atom_j in range(n_atoms):
            dist = np.linalg.norm(coords[atom_i] - coords[atom_j])
            distance_matrix[atom_i, atom_j] = dist

    return distance_matrix


def get_neighbour_list(atom_i, mol):
    """
    Calculate a neighbour list from atom i as a list of atom labels
    :param atom_i: (int) index of the atom
    :param mol: (object) Molecule object
    :return:
    """

    distance_vector = mol.distance_matrix[atom_i]
    dists_and_atom_labels = {}
    for atom_j, dist in enumerate(distance_vector):
        dists_and_atom_labels[dist] = mol.xyzs[atom_j][0]

    atom_label_neighbour_list = []
    for dist, atom_label in sorted(dists_and_atom_labels.items()):
        atom_label_neighbour_list.append(atom_label)

    return atom_label_neighbour_list


def get_identical_pairs(atoms_and_matches, n_atoms):
    """
    For a set of atoms and equivalent atoms get all the pairs of atom_i, atom_j which are equivalent

    :param atoms_and_matches: (dict) keyed with atom id and value a list (could be empty)
    :param n_atoms: (int) number of atoms
    :return: dict
    """
    identical_pairs = {}

    for atom_i in range(n_atoms):
        for atom_j in range(n_atoms):
            if atom_i != atom_j:
                pair = (atom_i, atom_j)
                identical_pairs[pair] = []

                for match_i in atoms_and_matches[atom_i] + [atom_i]:
                    for match_j in atoms_and_matches[atom_j] + [atom_j]:
                        if match_i != match_j:
                            equiv_pair = (match_i, match_j)

                            if equiv_pair != pair:
                                identical_pairs[pair].append(equiv_pair)

    return identical_pairs


def calc_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


i = np.array([1.0, 0.0, 0.0])
j = np.array([0.0, 1.0, 0.0])
k = np.array([0.0, 0.0, 1.0])
