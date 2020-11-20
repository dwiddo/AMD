## AMD.py
 
AMD.py contains a collection of functions for computing the PDD, AMD and WPD of crystal structures. PDD, AMD and WPD all accept (motif, cell, k) where motif and cell are in cartesian form and k>0 is an integer. If reading from a CIF, use the helper function motif_cell_fromCIF(path) to extract the relevant data (requires ccdc and ase).

### Requirements: 
- All functions require numpy and scipy. 
- WPD requires networkx if tol is not None.
- motif_cell_fromCIF requires ccdc and ase (used to read in .cif files).
- WPD_EMD, or in general comparing WPDs with earth mover's distance, requires [Wasserstein.py](https://www.dropbox.com/s/hzd2phmmitx6q0a/Wasserstein.py?dl=0).

Use the function motif_cell_fromCIF() to read in relevant data from a .cif file. Note that it returns a list of motifs and cells, so even if the file contains one structure, the only motif and cell are extracted with
```py
motif, cell = motif_cell_fromCIF(path)[0]
```
but if the file has multiple structures, you can use Python's loop syntax:
```py
for motif, cell in motif_cell_fromCIF(path):
    amd = AMD(motif, cell, 1000)
```
or
```py
amds = [AMD(m, c, 1000) for m, c in motif_cell_fromCIF(path)]
```
**If a cif has an asymmetric unit and symmetry operations**, i.e, the list of atomic sites in the cif is not all sites in the unit cell, use the optional parameter ``` fill_cell=True``` in ```motif_cell_fromCIF``` to pack the unit cell (this just passes the crystal through ```ccdc.crystal.Crystal.packing()```). **Functions will return incorrect values if this is missed.**

### Example uses

If not running from AMD.py, import with
```py
from AMD import AMD, motif_cell_fromCIF
```

-  One CIF, one crystal:
    ```py
    path = 'path/to/my/cif/file.cif' 
    motif, cell = motif_cell_fromCIF(path)[0]
    amd = AMD(motif, cell, 1000)
    ```
- One CIF, many crystals:
    ```py
    path = 'path/to/my/cif/file.cif' 
    amds = [AMD(motif, cell, 1000) for motif, cell in motif_cell_fromCIF(path)]
    ```
- Many CIFs (one crystal each) in a directory:
    ```py
    import os
    path = 'path/to/my/cifs'
    amds = []
    for file in os.listdir(path):
        if file.endswith('.cif'):
            motif, cell = motif_cell_fromCIF(os.path.join(path, file))[0]
            amds.append(AMD(motif, cell, 1000))
    ```
- From ccdc Crystal object:
    ```py
    motif, cell = motif_cell_fromCrystal(crystal)
    amd = AMD(motif, cell, 1000)
    ```
- From ase Atoms object:
    ```py
    motif = atoms.get_positions()
    cell = atoms.get_cell()
    amd = AMD(motif, cell, 1000)
    ```
- From pymatgen Structure object:
    ```py
    motif = structure.cart_coords
    cell = structure.lattice.matrix
    amd = AMD(motif, cell, 1000)
    ```

### WPD Notes

The function WPD(motif, cell, k) returns a single np.ndarray of shape (m,k+1) where m=motif.shape[0]. The first column contains the weights for the WPD. To separate the weights from the rest of the matrix, do
```py
x = WPD(motif, cell, k)
weights = x[:, 0]
distances = x[:, 1:]
```

WPD also accepts an optional tol parameter (default tol=None). If tol=None, rows are grouped only if they are exactly equal. Otherwise, rows are grouped when they are closer to each other than tol (Euclidean distance). It is not true that for any two rows in the same group, their distance is less than tol. Rather, for any row in a group there exists another row in the same group less than tol away.

### Earth mover's distance between WPDs
Requires [Wasserstein.py](https://www.dropbox.com/s/hzd2phmmitx6q0a/Wasserstein.py?dl=0). The function ```Wasserstein.wasserstein(w1, w2, dm)```  accepts two sets of weights and a distance matrix, returning the Wasserstein distance (earth mover's distance) between two suitable sets. Implementing the EMD between WPDs is just a few lines, 
```py
from scipy.spatial.distance import cdist
from Wasserstein import wasserstein
dm = cdist(wpd[:, 1:], wpd_[:, 1:], metric='euclidean')
emd = wasserstein(wpd[:, 0], wpd_[:, 0], dm)
```
this is the code in the helper function ```WPD_EMD(wpd, wpd_)``` which accepts two WPDs as returned by ``` WPD()```.
- Comparing two crystals:

    If the two crystals are in separate cifs, or are ase Atoms, pymatgen Structure or ccdc Crystal objects, see above how to extract the Cartesian motifs and cells. Then (for an integer k),
    ```py
    wpd = WPD(motif, cell, k)
    wpd_ = WPD(motif_, cell_, k)
    emd = WPD_EMD(wpd, wpd_)
    ```
- Comparing many crystals:

    The examples above explain how to extract many crystals and get their WPDs, for example given one cif with many crystals
    ```py
    wpds = [WPD(m, c, 100) for m, c in motif_cell_fromCIF(path)]
    ```
    To compare them all pairwise,
    ```py
    from itertools import combinations
    flat_dm = [WPD_EMD(wpd, wpd_) for wpd, wpd_ in combinations(wpds, 2)]
    ```
    then ``` flat_dm``` is a condensed distance vector (as described [in the scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html#scipy.spatial.distance.squareform)) containing all the pairwise distances between wpds. To get a symmetric square distance matrix, use
    ```py
    from scipy.spatial.distance import squareform
    square_dm = squareform(np.array(flat_dm))
    ```
    where ``` square_dm[i][j]``` is the EMD between ``` wpds[i]``` and ``` wpds[j]```. The condensed distance vector is accepted by ```scipy.cluster.hierarchy.linkage()``` used for hierarchical clustering. 