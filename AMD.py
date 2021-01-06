"""
Collection of functions to caculate PPD(k), AMD(k) and WPD(k).
"""

from itertools import product
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

def dist(p):
    return sum(x**2 for x in p)

def generate_concentric_cloud(motif, cell):
    ymax = defaultdict(int)
    d = 0
    while True:
        positive_int_lattice = []
        while True:
            batch = []
            for x in product(range(d+1), repeat=2):
                pt = x + (ymax[x],)
                if dist(pt) <= d**2:
                    batch.append(pt)
                    ymax[x] += 1
            if not batch:
                break
            positive_int_lattice += batch
        positive_int_lattice.sort(key=dist)

        # expand positive_int_lattice to int_lattice with reflections
        int_lattice = []
        for p in positive_int_lattice:
            int_lattice.append(p)
            if p[0]: int_lattice.append((-p[0], p[1], p[2]))
            if p[1]: int_lattice.append((p[0], -p[1], p[2]))
            if p[2]: int_lattice.append((p[0], p[1], -p[2]))
            if p[0] and p[1]: int_lattice.append((-p[0], -p[1], p[2]))
            if p[0] and p[2]: int_lattice.append((-p[0], p[1], -p[2]))
            if p[1] and p[2]: int_lattice.append((p[0], -p[1], -p[2]))
            if p[0] and p[1] and p[2]: int_lattice.append((-p[0], -p[1], -p[2]))

        lattice = np.array(int_lattice) @ cell
        yield np.concatenate([motif + translation for translation in lattice])
        d += 1

def PDD(motif, cell, k):
    """
    Returns PDD(k).

    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,3)
        cell: unit cell (cartesian). ndarray shape (3,3)
        k: no of columns in output. int >= 1
    Returns:
        ndarray shape (m,k) where m = motif.shape[0].
    """

    # generates point cloud in concentric layers
    g = generate_concentric_cloud(motif, cell)

    # get at least k points in the cloud
    points = 0
    cloud = []
    while points <= k:
        l = next(g)
        points += l.shape[0]
        cloud.append(l)
    cloud.append(next(g))
    cloud = np.concatenate(cloud)

    # nearest neighbour distance query
    tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    d_, _ = tree.query(motif, k=k+1, n_jobs=-1)
    d = np.zeros_like(d_)

    # keep generating layers and getting distances until they don't change
    while not np.array_equal(d, d_):
        d = np.copy(d_)
        cloud = np.append(cloud, next(g), axis=0)
        tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
        d_, _ = tree.query(motif, k=k+1, n_jobs=-1)

    return d_[:, 1:]

def AMD(motif, cell, k):
    """
    Returns AMD(k).

    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,3)
        cell: unit cell (cartesian). ndarray shape (3,3)
        k: length of output. int >= 1
    Returns:
        ndarray shape (k,) where m = motif.shape[0].
    """
    return np.average(PDD(motif, cell, k), axis=0)

def WPD(motif, cell, k, tol=None):
    """
    Returns WPD(k). Weights are in the first column, so
        >>> x = WPD(motif, cell, k)
        >>> weights = x[:,0]
        >>> distances = x[:,1:]
    If tol is not None, Rows are grouped together such that any
    row in a group is less than tol away (Euclidean distance)
    from at least one other row in the same group.
    If tol=None, rows must be exactly equal to be grouped.

    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,3)
        cell: unit cell (cartesian). ndarray shape (3,3)
        k: no of columns in output. int >= 1
    Returns:
        ndarray shape (m,k+1) where m = motif.shape[0].
    """

    pdd = PDD(motif, cell, k)
    M = pdd.shape[0]
    wpd = []
    if tol is not None:
        import networkx as nx
        d = pdist(pdd)
        d[d == 0] = -1
        distances = squareform(d)
        distances[distances >= tol] = 0
        indexes = np.argwhere(np.triu(distances))
        g = nx.Graph()
        g.add_edges_from(indexes)
        for group in nx.connected_components(g):
            row = [*np.average(pdd[list(group), :], axis=0), len(group) / M]
            wpd.append(row)
        not_grouped = [i for i in range(M) if i not in g.nodes]
        for index in not_grouped:
            row = [*pdd[index], 1 / M]
            wpd.append(row)
    else:
        counter = Counter([tuple(x) for x in pdd])
        for c in counter:
            wpd.append([*c, counter[c] / M])
    wpd.sort()
    wpd = np.array(wpd)
    wpd = np.concatenate((wpd[:, -1:], wpd[:, :-1]), axis=1)

    return np.array(wpd)

def motif_cell_fromCrystal(crystal, fill_cell=False):
    """
    ccdc.crystal.Crystal --> np.array shape (m,3), np.array shape (3,3).
    Optional param fill_cell (default False) will expand an asymmetric unit to
    fill the unit cell. This is necessary if the .cif contains an asymmetric unit
    that needs to be expanded by symmetry operations.
    """
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell([*crystal.cell_lengths, *crystal.cell_angles])
    mol = crystal.molecule if not fill_cell else crystal.packing(inclusion='OnlyAtomsIncluded')
    motif = np.array([[a.coordinates.x, a.coordinates.y, a.coordinates.z] for a in mol.atoms])
    return motif, cell

def motif_cell_fromCIF(path, fill_cell=False):
    """
    str (path to cif) --> np.array(s) shape (m,3), np.array(s) shape (3,3).
    Returns cartesian motif(s) and cell(s) in cif for use in the functions in
    this file. Returns a list [(motif1, cell1), ...] for all structures in the
    cif file.
    Optional param fill_cell (default False) will expand an asymmetric unit to
    fill the unit cell. This is necessary if the .cif contains an asymmetrix unit
    that is expanded by symmetry operations.
    """
    from ccdc import io
    reader = io.CrystalReader(path)
    return [motif_cell_fromCrystal(crystal, fill_cell=fill_cell) for crystal in reader]

def AMD_estimate(motif, cell, k):
    """
    Returns an estimate of AMD(motif, cell, k), given by
    cbrt((3 * Vol(cell) * k) / (4 * pi * m))
    where m = motif.shape[0]
    """
    m = motif.shape[0]
    det = np.linalg.det(cell)
    c = np.cbrt((3 * det) / (4 * np.pi * m))
    return [np.cbrt(x) * c for x in range(k+1)]

def WPD_EMD(wpd, wpd_):
    """
    Takes two wpd matrices (as returned by WPD()) and returns the Earth mover's
    distance between them.
    """
    from Wasserstein import wasserstein
    dm = cdist(wpd[:, 1:], wpd_[:, 1:], metric='euclidean')
    return wasserstein(wpd[:, 0], wpd_[:, 0], dm)

def example():
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    motif = np.random.uniform(size=(100, 3))
    print(AMD(motif, cell, 10000))

if __name__ == "__main__":
    example()