"""Numerical evidence for the solver claims made in the manuscript.

Section 6.4 of the paper states that the factor-pair Schwarz operator is
symmetric, positive semidefinite on the whole coefficient space, and positive
definite on the identified subspace only under a coverage condition on the
local kernels. Section 5 states that for two absorbed factors the pairwise
quantity rho_qr is the exact per-sweep contraction factor of MAP. Each test
below checks one of those statements numerically rather than resting on the
algebra as written.

The designs are small enough that the local solver takes the dense Cholesky
path, so the operator is deterministic and can be materialized column by
column.
"""

from __future__ import annotations

import unittest

import numpy as np

import within


def _solver(categories: np.ndarray) -> within.Solver:
    return within.Solver(np.asfortranarray(categories.astype(np.uint32)))


def _dense_preconditioner(categories: np.ndarray) -> np.ndarray:
    """Materialize M^-1 by applying it to each unit vector."""
    preconditioner = _solver(categories).preconditioner
    identity = np.eye(preconditioner.ncols)
    columns = [
        np.asarray(preconditioner.apply(np.ascontiguousarray(identity[:, j])))
        for j in range(preconditioner.ncols)
    ]
    return np.column_stack(columns)


def _design_matrix(categories: np.ndarray) -> np.ndarray:
    """Build the dense fixed-effect dummy matrix D for a small design."""
    n_obs, n_factors = categories.shape
    offsets = [0]
    for factor in range(n_factors):
        offsets.append(offsets[-1] + int(categories[:, factor].max()) + 1)

    design = np.zeros((n_obs, offsets[-1]))
    for row in range(n_obs):
        for factor in range(n_factors):
            design[row, offsets[factor] + int(categories[row, factor])] = 1.0
    return design


def _nullspace(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis of the nullspace as columns."""
    _, singular_values, right = np.linalg.svd(matrix)
    rank = int((singular_values > tol).sum())
    return right[rank:].T


# Two-factor designs. With one factor pair the Schwarz operator has a single
# family of subdomains, so its kernel should coincide with ker(D).
TWO_FACTOR_DESIGNS = {
    "connected": np.array([[0, 0], [0, 1], [1, 0], [1, 0], [2, 1], [2, 1]]),
    "two_components": np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]]
    ),
    "three_components": np.array(
        [[0, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 3], [4, 4], [5, 4]]
    ),
}

# Designs with three or more factors. Every level then sits in Q-1 pair
# subdomains, and the local kernel conditions from different pairs conflict.
MULTI_FACTOR_DESIGNS = {
    # The worker-firm-year panel of Section 4.
    "paper_toy": np.array(
        [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [2, 1, 0], [2, 1, 1]]
    ),
    # Two labour markets that share the year dimension.
    "split_worker_firm": np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [3, 1, 0],
            [3, 1, 1],
        ]
    ),
    # A third factor nested inside the second.
    "nested_year": np.array(
        [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1]]
    ),
}


def _random_four_factor_design(seed: int = 0, n_obs: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.integers(0, 6, n_obs),
            rng.integers(0, 4, n_obs),
            rng.integers(0, 3, n_obs),
            rng.integers(0, 2, n_obs),
        ]
    )


ALL_DESIGNS = {
    **{f"q2_{name}": design for name, design in TWO_FACTOR_DESIGNS.items()},
    **{f"qn_{name}": design for name, design in MULTI_FACTOR_DESIGNS.items()},
    "q4_random": _random_four_factor_design(),
}


class PreconditionerPropertyTests(unittest.TestCase):
    def test_operator_is_self_adjoint(self) -> None:
        """Symmetric partition weights make M^-1 its own adjoint.

        The weights act on both the restriction and the prolongation side of
        every local solve, so <x, M^-1 y> and <M^-1 x, y> must agree. This is
        checked through the operator itself, not through the dense matrix, so
        it also covers the application path used inside LSMR.
        """
        rng = np.random.default_rng(20260726)
        for name, design in ALL_DESIGNS.items():
            with self.subTest(design=name):
                preconditioner = _solver(design).preconditioner
                n_dofs = preconditioner.ncols
                for _ in range(8):
                    x = rng.standard_normal(n_dofs)
                    y = rng.standard_normal(n_dofs)
                    left = float(x @ np.asarray(preconditioner.apply(y.copy())))
                    right = float(y @ np.asarray(preconditioner.apply(x.copy())))
                    scale = np.linalg.norm(x) * np.linalg.norm(y)
                    self.assertLessEqual(abs(left - right), 1e-10 * scale)

    def test_spectrum_is_nonnegative(self) -> None:
        """M^-1 is positive semidefinite on the full coefficient space.

        It is a sum of terms R' W A^+ W R with A^+ positive semidefinite, so no
        eigenvalue may be negative. Symmetry alone would not give this.
        """
        for name, design in ALL_DESIGNS.items():
            with self.subTest(design=name):
                operator = _dense_preconditioner(design)
                eigenvalues = np.linalg.eigvalsh((operator + operator.T) / 2)
                self.assertGreaterEqual(eigenvalues.min(), -1e-10 * max(1.0, eigenvalues.max()))

    def test_two_factor_kernel_is_the_design_kernel(self) -> None:
        """With one factor pair, ker(M^-1) equals ker(D).

        Each subdomain kernel is the sign-flipped component constant, which is
        exactly the direction that leaves D alpha unchanged. So the operator is
        positive definite precisely on the identified subspace, and singular on
        the full space. The count of kernel directions tracks the number of
        connected components.
        """
        for name, design in TWO_FACTOR_DESIGNS.items():
            with self.subTest(design=name):
                operator = _dense_preconditioner(design)
                design_kernel = _nullspace(_design_matrix(design))
                eigenvalues = np.linalg.eigvalsh((operator + operator.T) / 2)
                scale = max(1.0, eigenvalues.max())
                n_zero = int((eigenvalues < 1e-10 * scale).sum())

                self.assertEqual(n_zero, design_kernel.shape[1])
                # Every design-kernel direction is annihilated by the operator.
                residual = operator @ design_kernel
                self.assertLessEqual(np.abs(residual).max(), 1e-10 * scale)

    def test_multi_factor_kernel_is_trivial(self) -> None:
        """With three or more factors the coverage condition holds strictly.

        Every level appears in Q-1 pair subdomains. A vector in the kernel would
        have to be a constant on one side and its negative on the other in each
        of those subdomains simultaneously, and those requirements conflict. The
        operator is then positive definite on the whole coefficient space, which
        is more than LSMR needs.

        Note that this also means M^-1 does not annihilate ker(D) when Q >= 3,
        which is why Appendix B argues that a kernel component in the iterate is
        harmless rather than absent.
        """
        for name, design in MULTI_FACTOR_DESIGNS.items():
            with self.subTest(design=name):
                operator = _dense_preconditioner(design)
                eigenvalues = np.linalg.eigvalsh((operator + operator.T) / 2)
                self.assertGreater(eigenvalues.min(), 1e-6 * eigenvalues.max())

                design_kernel = _nullspace(_design_matrix(design))
                self.assertGreater(design_kernel.shape[1], 0)
                # The design kernel is strictly larger than the operator kernel.
                self.assertGreater(
                    np.abs(operator @ design_kernel).max(), 1e-6 * eigenvalues.max()
                )

    def test_corrections_are_centred_within_each_component(self) -> None:
        """The returned local correction is centred, not merely the input.

        Algorithm 1 shows the projection applied to the right-hand side of each
        local solve. The output side matters too: a correction carrying a
        component constant would push the iterate along a direction the solver
        cannot resolve. For a two-factor design the check is exact, because
        ker(M^-1) is then ker(D).
        """
        rng = np.random.default_rng(4)
        for name, design in TWO_FACTOR_DESIGNS.items():
            with self.subTest(design=name):
                preconditioner = _solver(design).preconditioner
                design_kernel = _nullspace(_design_matrix(design))
                for _ in range(8):
                    x = rng.standard_normal(preconditioner.ncols)
                    correction = np.asarray(preconditioner.apply(x.copy()))
                    leakage = design_kernel.T @ correction
                    self.assertLessEqual(
                        np.abs(leakage).max(),
                        1e-10 * max(1.0, float(np.linalg.norm(correction))),
                    )

    def test_absorbed_residuals_are_gauge_invariant(self) -> None:
        """Relabelling levels does not move the absorbed residual.

        The solver reports residuals, not fixed-effect coefficients, so its
        output must not depend on which level within a component happens to be
        numbered first. The designs here include several with more than one
        connected component, where a coefficient-level comparison would be
        meaningless but the residual comparison is exact.
        """
        rng = np.random.default_rng(11)
        for name, design in ALL_DESIGNS.items():
            with self.subTest(design=name):
                rhs = np.asfortranarray(
                    rng.standard_normal((design.shape[0], 2))
                )
                options = within.LsmrOptions()
                baseline = within.solve_batch(
                    np.asfortranarray(design.astype(np.uint32)),
                    rhs,
                    options=options,
                )

                relabelled = design.copy()
                for factor in range(design.shape[1]):
                    n_levels = int(design[:, factor].max()) + 1
                    permutation = rng.permutation(n_levels)
                    relabelled[:, factor] = permutation[design[:, factor]]

                permuted = within.solve_batch(
                    np.asfortranarray(relabelled.astype(np.uint32)),
                    rhs,
                    options=options,
                )

                difference = np.abs(
                    np.asarray(baseline.demeaned) - np.asarray(permuted.demeaned)
                ).max()
                self.assertLessEqual(difference, 1e-8 * max(1.0, np.abs(rhs).max()))

    def test_components_are_absorbed_independently(self) -> None:
        """Stacking two disconnected designs absorbs each one on its own.

        A component that shares no levels with another contributes its own
        kernel direction and its own local solve. Absorbing the stacked design
        must give the same residuals as absorbing the two parts separately.
        """
        rng = np.random.default_rng(7)
        left = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        right = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

        stacked = np.vstack([left, right + np.array([left[:, 0].max() + 1, left[:, 1].max() + 1])])
        rhs_left = rng.standard_normal((left.shape[0], 1))
        rhs_right = rng.standard_normal((right.shape[0], 1))
        rhs_stacked = np.vstack([rhs_left, rhs_right])

        options = within.LsmrOptions()
        joint = within.solve_batch(
            np.asfortranarray(stacked.astype(np.uint32)),
            np.asfortranarray(rhs_stacked),
            options=options,
        )
        part_left = within.solve_batch(
            np.asfortranarray(left.astype(np.uint32)),
            np.asfortranarray(rhs_left),
            options=options,
        )
        part_right = within.solve_batch(
            np.asfortranarray(right.astype(np.uint32)),
            np.asfortranarray(rhs_right),
            options=options,
        )

        expected = np.vstack(
            [np.asarray(part_left.demeaned), np.asarray(part_right.demeaned)]
        )
        self.assertLessEqual(
            np.abs(np.asarray(joint.demeaned) - expected).max(), 1e-8
        )


class TwoFactorContractionTests(unittest.TestCase):
    """Section 5's claim that rho_qr is exact for two absorbed factors."""

    @staticmethod
    def _rho(categories: np.ndarray) -> float:
        design = _design_matrix(categories)
        n_left = int(categories[:, 0].max()) + 1
        left, right = design[:, :n_left], design[:, n_left:]
        normalized = (
            np.diag(1.0 / np.sqrt(np.diag(left.T @ left)))
            @ (left.T @ right)
            @ np.diag(1.0 / np.sqrt(np.diag(right.T @ right)))
        )
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        return float(singular_values[1] ** 2)

    @staticmethod
    def _observed_contraction(categories: np.ndarray, n_sweeps: int = 40) -> float:
        """Per-sweep contraction of residual MAP, measured after transients decay."""
        design = _design_matrix(categories)
        n_left = int(categories[:, 0].max()) + 1
        left, right = design[:, :n_left], design[:, n_left:]

        def residual_projector(block: np.ndarray) -> np.ndarray:
            return np.eye(block.shape[0]) - block @ np.linalg.pinv(block.T @ block) @ block.T

        sweep = residual_projector(right) @ residual_projector(left)
        limit = residual_projector(design)

        rng = np.random.default_rng(3)
        state = rng.standard_normal(design.shape[0])
        previous = state - limit @ state
        ratios = []
        for _ in range(n_sweeps):
            state = sweep @ state
            error = state - limit @ state
            if np.linalg.norm(previous) > 1e-12:
                ratios.append(np.linalg.norm(error) / np.linalg.norm(previous))
            previous = error
        return float(np.median(ratios[-5:]))

    def test_rho_is_the_exact_per_sweep_contraction(self) -> None:
        """rho_qr is the squared cosine of the Friedrichs angle, so MAP contracts by it.

        The paper uses this to say that a small two-factor gap means slow MAP in
        a precise sense, and that no such statement is available for three or
        more factors. Only the two-factor claim is testable here, which is the
        point.
        """
        for name, design in TWO_FACTOR_DESIGNS.items():
            if name != "connected":
                # Disconnected designs mix component rates; the paper reports the
                # smallest gap over components rather than one contraction factor.
                continue
            with self.subTest(design=name):
                self.assertAlmostEqual(
                    self._observed_contraction(design), self._rho(design), places=6
                )

    def test_paper_example_gap_is_one_third(self) -> None:
        """The appendix's worked spectral-gap example reports 1/3."""
        design = TWO_FACTOR_DESIGNS["connected"]
        self.assertAlmostEqual(1.0 - self._rho(design), 1.0 / 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
