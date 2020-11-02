"""
Collection of functions to caculate PPD(k), AMD(k) and WPD(k).
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from itertools import product

def dist(p):
    return sum(x**2 for x in p)

def generate_N3():
    """
    generates batches of positive integer lattice points in 3D
    ordered by distance to the origin.
    """
    ymax = defaultdict(int) 
    d = 0
    while True:
        yieldable = []
        while True:
            batch = []
            for x in product(range(d+1), repeat=2):
                pt = x + (ymax[x],)
                if dist(pt) <= d**2:
                    batch.append(pt)
                    ymax[x] += 1
            if not batch:
                break
            yieldable += batch
        yieldable.sort(key=dist)
        for p in yieldable:
            yield p
        d += 1
        yield None

def generate_Z3():
    """
    generates all integer lattice points in batches
    ordered by distance to the origin
    """
    for p in generate_N3():
        yield p
        if p is not None:
            if p[0]: yield -p[0], p[1], p[2]
            if p[1]: yield p[0], -p[1], p[2]
            if p[2]: yield p[0], p[1], -p[2] 
            if p[0] and p[1]: yield -p[0], -p[1], p[2]
            if p[0] and p[2]: yield -p[0], p[1], -p[2]
            if p[1] and p[2]: yield p[0], -p[1], -p[2] 
            if p[0] and p[1] and p[2]: yield -p[0], -p[1], -p[2]

def generate_concentric_lattice(motif, cell):
    """
    generates periodic point cloud in 'spherical' layers
    """
    t = generate_Z3()

    points = []
    while True:
        point = next(t)
        if point is None:
            lattice = np.array(points) @ cell
            layer = np.concatenate([motif + translation for translation in lattice])
            yield layer
            points = []
        else:
            points.append(point)

def PDD(motif, cell, k):
    """
    Returns PDD(t;k).

    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,3)
        lattice: unit cell (cartesian). ndarray shape (3,3)
        k: no of columns in output. int >= 1
    Returns:
        ndarray shape (m,k) where m = motif.shape[0].

    """

    g = generate_concentric_lattice(motif, cell)

    cloud = next(g)
    while cloud.shape[0] <= k:
        cloud = np.append(cloud, next(g), axis=0)

    tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    d_, _ = tree.query(motif, k=k+1, n_jobs=-1)
    d = np.zeros_like(d_)

    while not np.array_equal(d, d_):
        d = np.copy(d_)
        cloud = np.append(cloud, next(g), axis=0)
        tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
        d_, _ = tree.query(motif, k=k+1, n_jobs=-1)
    return d_[:,1:]

def AMD(motif, cell, k):
    """
    Returns PDD(t;k).

    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,3)
        lattice: unit cell (cartesian). ndarray shape (3,3)
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
        lattice: unit cell (cartesian). ndarray shape (3,3)
        k: no of columns in output. int >= 1
    Returns:
        ndarray shape (m,k+1) where m = motif.shape[0].
    """
    from collections import Counter
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform

    pdd = PDD(motif, cell, k)
    M = pdd.shape[0]
    wpd = []
    if tol is not None:
        d = pdist(pdd)
        d[d==0] = -1
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
    wpd = np.concatenate((wpd[:,-1:], wpd[:,:-1]), axis=1)

    return np.array(wpd)

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

def motif_cell_fromCrystal(crystal):
    """
    ccdc.crystal.Crystal --> np.array shape (m,3), np.array shape (3,3)
    """
    from ase.geometry import cellpar_to_cell
    motif = np.array([[a.coordinates.x, a.coordinates.y, a.coordinates.z] 
                           for a in crystal.molecule.atoms])
    cell = cellpar_to_cell([*crystal.cell_lengths, *crystal.cell_angles])
    return motif, cell

def motif_cell_fromCIF(path):
    """
    Returns cartesian motif(s) and cell(s) in cif
    for use in the functions in this file.
    If cif contains 1 crystal, returns (motif, cell)
    If cif contains > 1 crystal, returns a list [(motif1, cell1), ...]
    """
    from ccdc import io
    reader = io.CrystalReader(path)
    if len(reader) == 1:
        return motif_cell_fromCrystal(reader[0])
    else:
        return [motif_cell_fromCrystal(crystal) for crystal in reader]


if __name__ == "__main__":

    """
    PDD, AMD and WPD all accept (motif, cell, k)
    where motif and cell are in cartesian form. 

    If reading from a CIF, use the helper function motif_cell_fromCIF(path)
    to extract the relevant data. (ase required)

    Example uses:
        One CIF, one crystal:
            >>> path = 'path/to/my/cif/file.cif' 
            >>> motif, cell = motif_cell_fromCIF(path)
            >>> amd = AMD(motif, cell, 1000)
        One CIF, many crystals:
            >>> path = 'path/to/my/cif/file.cif' 
            >>> amds = [AMD(motif, cell, 1000) for motif, cell in motif_cell_fromCIF(path)]
        Many CIFs (one crystal each) in a directory:
            >>> import os
            >>> path = 'path/to/my/cifs'
            >>> amds = []
            >>> for file in os.listdir(path):
            >>>     if file.endswith('.cif'):
            >>>         motif, cell = motif_cell_fromCIF(os.path.join(path, file))
            >>>         amds.append(AMD(motif, cell, 1000))

    """
    motif, cell = motif_cell_fromCIF("Data\CIFs\T2_simulated\job_00001.cif")
    print(WPD(motif, cell, 200))
