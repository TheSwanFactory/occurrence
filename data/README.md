# Born Channel data convention

`kraus84.npz` contains the 84 right-multiplication matrices `R_z` for the
Cayley–Dickson convention

`(a,b)(c,d) = (ac - d conjugate(b), conjugate(a)d + cb)`.

The paper is written using left multiplication `L_z`. The two families are
orthogonally similar:

`R_z = -C L_z C`,

where `C` is sedenion conjugation. Consequently they define orthogonally
similar channel matrices and have identical spectra, ranks, moments, and
invariant-sector dimensions. Matrix-by-matrix reproduction must nevertheless
use the handedness stated here. In particular, the released artifact realizes
the concrete clock as `R_{e8}` rather than `L_{e8}`.
