"""
Collection of functions to caculate PPD(k), AMD(k) and WPD(k)
in n dimensions (motif points in R^n and a cell in R^(nxn)).
"""

from itertools import product, combinations
from collections import defaultdict, Counter
import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

def dist(p):
    return sum(x**2 for x in p)

def generate_concentric_cloud(motif, cell):
    n = motif.shape[1]
    ymax = defaultdict(int)
    d = 0
    while True:
        positive_int_lattice = []
        while True:
            batch = []
            for x in product(range(d+1), repeat=n-1):
                pt = [*x, ymax[x]]
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
            for n_reflections in range(1, n+1):
                for indexes in combinations(range(n), n_reflections):
                    if all((p[i] for i in indexes)):
                        p_ = list(p)
                        for i in indexes:
                            p_[i] *= -1
                        int_lattice.append(p_)

        lattice = np.array(int_lattice) @ cell
        yield np.concatenate([motif + translation for translation in lattice])
        d += 1

def PDD(motif, cell, k):
    """
    Parameters:
        motif: Cartesian coords of motif points. ndarray of shape (m,n)
        cell: unit cell (cartesian). ndarray shape (n,n)
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
        motif: Cartesian coords of motif points. ndarray of shape (m,n)
        cell: unit cell (cartesian). ndarray shape (n,n)
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
        motif: Cartesian coords of motif points. ndarray of shape (m,n)
        cell: unit cell (cartesian). ndarray shape (n,n)
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

def AMD_estimate(motif, cell, k):
    """
    Returns an estimate of AMD(motif, cell, k)
    """
    n = motif.shape[1]
    c = PPC(motif, cell)
    return [(x ** (1. / n)) * c for x in range(k+1)]

def PPC(motif, cell):
    """
    Returns the PPC of a given cartesian cell and motif.
    """
    m, n = motif.shape
    det = np.linalg.det(cell)
    t = (n - n % 2) / 2
    if n % 2 == 0:
        V = (np.pi ** t) / math.factorial(t)
    else:
        V = (2 * math.factorial(t) * (4 * np.pi) ** t) / math.factorial(n)
    return (det / (m * V)) ** (1./n)

def WPD_EMD(wpd, wpd_):
    """
    Takes two wpd matrices (as returned by WPD()) and returns the Earth mover's
    distance between them.
    """
    from Wasserstein import wasserstein
    dm = cdist(wpd[:, 1:], wpd_[:, 1:], metric='euclidean')
    return wasserstein(wpd[:, 0], wpd_[:, 0], dm)
