# Born Channel data convention

`kraus84.npz` contains the 84 right-multiplication matrices `R_z` for the
Cayley–Dickson convention

`(a,b)(c,d) = (ac - d conjugate(b), conjugate(a)d + cb)`.

The package's canonical signed-basis table is stored in the historical core
coordinate presentation. The exact 061.11 machine reaches that same table via
`Phi(a,b) = (conjugate(a), b)`, which negates coordinates `e1` through `e7`.
This coordinate conversion is distinct from the left/right handedness relation
below.

The paper is written using left multiplication `L_z`. The two families are
orthogonally similar:

`R_z = -C L_z C`,

where `C` is sedenion conjugation. Consequently they define orthogonally
similar channel matrices and have identical spectra, ranks, moments, and
invariant-sector dimensions. Matrix-by-matrix reproduction must nevertheless
use the handedness stated here. In particular, the released artifact realizes
the concrete clock as `R_{e8}` rather than `L_{e8}`.
