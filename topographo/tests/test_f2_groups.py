"""Abstract F_2 group machinery (Issue 009 Gate 0, promoted in 0.8.3).

Pins the group orders, the Fano incidence counts, the Arm C rejection predicate
and the pp fence that the Issue-009 arms are built on. Torch-free and numpy-free,
so it runs in default CI; ``experiments/sfp_representation/groups.py`` remains
the artifact driver.
"""

from __future__ import annotations

import ast
import itertools
import pathlib

from topographo.core import f2_groups


def test_certificate_agrees_and_matches_the_pinned_digest():
    assert f2_groups.assert_f2_group_laws()
    payload = f2_groups.certificate()
    assert payload["verdict"]["agrees"]
    assert payload["verdict"]["count_disagreements"] == []
    assert payload["verdict"]["broken_relational_laws"] == []
    assert all(payload["relational_laws"].values())
    assert (
        f2_groups.certificate_sha256(payload)
        == f2_groups.EXPECTED_CERTIFICATE_SHA256
    )


def test_fano_plane_incidence_counts():
    lines = f2_groups.fano_lines()
    assert len(lines) == 7
    assert all(len(line) == 3 for line in lines)
    # Every pair of distinct nonzero points lies on exactly one line, and the
    # third point of that line is the XOR of the pair.
    for p, q in itertools.combinations(f2_groups.POINTS3, 2):
        hosts = [line for line in lines if p in line and q in line]
        assert len(hosts) == 1
        assert f2_groups.third_point(p, q) == p ^ q
        assert p ^ q in hosts[0]
    assert all(len(f2_groups.lines_through(p)) == 3 for p in f2_groups.POINTS3)


def test_group_orders():
    assert len(f2_groups.gl_3_2()) == 168
    assert len(f2_groups.gl_2_2()) == 6
    assert len(f2_groups.agl_2_2()) == 24
    assert len(f2_groups.EDGES) == 6
    assert len(f2_groups.vertex_permutations()) == 24


def test_arm_c_rejection_predicate_is_the_24_element_induced_class():
    # Matching-partition preservation is necessary but NOT sufficient: 48 of the
    # 720 edge permutations preserve the partition while only 24 are induced.
    induced = f2_groups.induced_edge_permutations()
    assert len(set(induced)) == 24
    all_perms = list(itertools.permutations(range(len(f2_groups.EDGES))))
    assert len(all_perms) == 720
    structure_preserving = [p for p in all_perms if f2_groups.is_structure_preserving(p)]
    matching_preserving = [
        p for p in all_perms if f2_groups.preserves_matching_partition(p)
    ]
    assert len(structure_preserving) == 24
    assert len(matching_preserving) == 48
    assert set(structure_preserving) == set(induced)
    assert len(all_perms) - len(structure_preserving) == 696


def test_pp_fence_all_six_permutations_preserve_the_xor_third_law():
    # The fence that makes a bare pp relabeling a symmetry, not a scramble.
    fence = f2_groups.pp_permutations_preserve_xor_third()
    assert fence["all_6_preserve_the_local_xor_third_law"]
    assert fence["all_6_lie_in_gl_2_2"]


def test_affine_covariance_and_chamber_action():
    for element in f2_groups.agl_2_2():
        matrix, _ = element
        for q in f2_groups.V2:
            for d in f2_groups.NONZERO2:
                shifted = tuple(a ^ b for a, b in zip(q, d))
                image = f2_groups.apply_affine(element, q)
                displaced = f2_groups.apply_affine(element, shifted)
                observed = tuple(a ^ b for a, b in zip(displaced, image))
                assert observed == f2_groups.pp_displacement_image(matrix, d)
        perm = f2_groups.chamber_permutation(element)
        assert sorted(perm) == list(range(4))
        inverse = f2_groups.chamber_permutation(f2_groups.invert_affine(element))
        assert tuple(inverse[perm[k]] for k in range(4)) == (0, 1, 2, 3)


def test_module_imports_nothing_beyond_the_standard_library():
    # The module claims to be self-contained label machinery; check the claim
    # against its own import graph rather than trusting the docstring.
    tree = ast.parse(pathlib.Path(f2_groups.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "collections", "hashlib", "itertools", "json"}
    assert "numpy" not in imported
    assert "topographo" not in imported
    assert "no Events" in f2_groups.certificate()["scope"]
