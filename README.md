## AMD.py
 
AMD.py contains a collection of functions for computing the PDD, AMD and WPD of crystal structures. PDD, AMD and WPD all accept (motif, cell, k) where motif and cell are in cartesian form and k>0 is an integer. If reading from a CIF, use the helper function motif_cell_fromCIF(path) to extract the relevant data. (ase required)

### Requirements: 
- All functions require numpy and scipy. 
- WPD requires networkx. 
- motif_cell_fromCIF requires ccdc and ase (used to read in .cif files).

### Example uses

If not running from AMD.py, import the relevant functions. Then

-  One CIF, one crystal:
    ```sh
    path = 'path/to/my/cif/file.cif' 
    motif, cell = motif_cell_fromCIF(path)
    amd = AMD(motif, cell, 1000)
    ```
- One CIF, many crystals:
    ```sh
    path = 'path/to/my/cif/file.cif' 
    amds = [AMD(motif, cell, 1000) for motif, cell in motif_cell_fromCIF(path)]
    ```
- Many CIFs (one crystal each) in a directory:
    ```sh
    import os
    path = 'path/to/my/cifs'
    amds = []
    for file in os.listdir(path):
        if file.endswith('.cif'):
            motif, cell = motif_cell_fromCIF(os.path.join(path, file))
            amds.append(AMD(motif, cell, 1000))
    ```
- From ccdc Crystal object:
    ```sh
    motif, cell = motif_cell_fromCrystal(crystal)
    amd = AMD(motif, cell, 1000)
    ```
- From ase Atoms object:
    ```sh
    motif = atoms.get_positions()
    cell = atoms.get_cell()
    amd = AMD(motif, cell, 1000)
    ```
- From pymatgen Structure object:
    ```sh
    motif = structure.cart_coords
    cell = structure.lattice.matrix
    amd = AMD(motif, cell, 1000)
    ```

### Notes on WPD
The function WPD(motif, cell, k) returns a single np.ndarray of shape (m,k+1) where m=motif.shape[0]. The first column contains the weights for the WPD. To separate the weights from the rest of the matrix, do
```sh
x = WPD(motif, cell, k)
weights = x[:,0]
distances = x[:,1:]
```

WPD also accepts an optional tol parameter (default tol=None). If tol=None, rows are grouped only if they are exactly equal. Otherwise, rows are grouped when they are closer to each other than tol (Euclidean distance). It is not true that for any two rows in the same group, their distance is less than tol. Rather, for any row in a group there exists another row in the same group less than tol away.
