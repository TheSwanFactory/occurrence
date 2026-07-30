"""Search for the simplest exact stationary identity for the spine share.

If a quadratic h and constant c obey

    s(x) - c = h(x) - E_z[h(K_z x / ||K_z x||)]

pointwise, then every stationary measure has E[s] = c.  This script searches
the complete 136-dimensional space of quadratic forms h, first with
c = 1/8 + 1/147 fixed and then with c free.  A genuine identity would have
machine-zero residual on fresh states.

This is a falsification check for the simplest analytic proof, not a proof
that the proposed constant is wrong and not a search over higher-degree or
rational coboundaries.
"""

import numpy as np


def main() -> None:
    data = np.load("data/kraus84.npz")
    kraus = data["K"]
    event_count, dim, _ = kraus.shape
    eye = np.eye(dim)
    e0 = eye[0]

    channel = sum(np.kron(k.T, k.T) for k in kraus) / event_count
    eigenvalues, eigenvectors = np.linalg.eigh(channel)
    clock = eigenvectors[:, np.argmin(abs(eigenvalues + 1))].reshape(dim, dim)
    clock *= np.sqrt(dim) / np.linalg.norm(clock)
    e8 = clock @ e0
    spine = np.outer(e0, e0) + np.outer(e8, e8)
    upper = np.triu_indices(dim)

    def quadratic_coordinates(matrix: np.ndarray) -> np.ndarray:
        coordinates = matrix[upper].copy()
        coordinates[upper[0] != upper[1]] *= 2
        return coordinates

    def coboundary_row(x: np.ndarray) -> np.ndarray:
        images = np.einsum("aij,j->ai", kraus, x)
        norms = np.linalg.norm(images, axis=1)
        images = images[norms > 1e-12] / norms[norms > 1e-12, None]
        next_second_moment = np.einsum("ai,aj->ij", images, images) / len(images)
        return quadratic_coordinates(np.outer(x, x) - next_second_moment)

    rng = np.random.default_rng(20260730)
    candidate = 1 / 8 + 1 / 147

    def sample(count: int) -> tuple[np.ndarray, np.ndarray]:
        states = rng.normal(size=(count, dim))
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        rows = np.array([coboundary_row(x) for x in states])
        shares = np.einsum("ai,ij,aj->a", states, spine, states)
        return rows, shares

    train_rows, train_shares = sample(1200)
    test_rows, test_shares = sample(400)

    fixed_coefficients = np.linalg.lstsq(
        train_rows, train_shares - candidate, rcond=1e-12
    )[0]
    fixed_residual = (
        test_rows @ fixed_coefficients - (test_shares - candidate)
    )

    augmented = np.column_stack([np.ones(len(train_rows)), train_rows])
    free_solution = np.linalg.lstsq(augmented, train_shares, rcond=1e-12)[0]
    free_residual = (
        np.column_stack([np.ones(len(test_rows)), test_rows]) @ free_solution
        - test_shares
    )

    print(f"quadratic coboundary rank: {np.linalg.matrix_rank(train_rows)}/136")
    print(f"candidate c: {candidate:.15f}")
    print(
        "fixed-candidate test residual: "
        f"rms={np.sqrt(np.mean(fixed_residual**2)):.3e}, "
        f"max={np.max(abs(fixed_residual)):.3e}"
    )
    print(f"best free constant: {free_solution[0]:.15f}")
    print(
        "free-constant test residual: "
        f"rms={np.sqrt(np.mean(free_residual**2)):.3e}, "
        f"max={np.max(abs(free_residual)):.3e}"
    )

    # An exact quadratic identity would reproduce at floating-point noise.
    if np.sqrt(np.mean(fixed_residual**2)) < 1e-10:
        raise SystemExit("unexpected exact quadratic identity found")

    print(
        "RESULT: no quadratic stationary coboundary proves "
        "s* = 1/8 + 1/147; higher-degree/rational analysis remains open."
    )


if __name__ == "__main__":
    main()
