"""A dependency stamp names content, not a checkout.

The release worktrees, the isolated runtime and the main clone are the same tree at different
absolute paths. Stamps written under one root recorded that root's absolute paths as dependency
keys, so every key missed when the stamp was read from a second root: `artifact_is_current`
returned False for artifacts that were byte-for-byte current, and the documented stage order
silently retrained them. `state/modeling/bg_mixture_experts_2025.parquet` was the expensive case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crimerisk.build_freshness import (
    artifact_is_current,
    content_digest,
    dependency_key,
    dependency_stamp_path,
    digest_multiset,
    repo_root,
    write_dependency_stamp,
)

OTHER_ROOT = "/Users/someone/Projects/data-projects/national/crimerisk-wt-level"


def _tree(tmp_path: Path) -> tuple[Path, list[Path]]:
    """An artifact and two dependencies, stamped here."""
    artifact = tmp_path / "bg_mixture_experts_2025.parquet"
    artifact.write_bytes(b"expert weights")
    dependencies = []
    for name, payload in (("features.parquet", b"features"), ("weights.csv", b"w,1\n")):
        path = tmp_path / name
        path.write_bytes(payload)
        dependencies.append(path)
    write_dependency_stamp(artifact, dependencies)
    return artifact, dependencies


def _rebase_stamp_paths(artifact: Path, root: str) -> dict:
    """Rewrite the stamp as a run under a different repo root would have written it.

    Only the recorded paths change; every digest, and the artifact itself, is untouched.
    """
    stamp_path = dependency_stamp_path(artifact)
    payload = json.loads(stamp_path.read_text())
    payload["dependencies"] = {
        f"{root}/{Path(key).name}": digest for key, digest in payload["dependencies"].items()
    }
    stamp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def test_a_stamp_written_under_another_repo_root_still_reads_as_current(tmp_path: Path) -> None:
    artifact, dependencies = _tree(tmp_path)
    assert artifact_is_current(artifact, dependencies)

    before = content_digest(artifact)
    foreign = _rebase_stamp_paths(artifact, OTHER_ROOT)
    assert all(key.startswith(OTHER_ROOT) for key in foreign["dependencies"])

    assert artifact_is_current(artifact, dependencies)
    # Reading a current artifact rebuilds nothing and rewrites nothing.
    assert content_digest(artifact) == before


def test_changed_content_is_still_stale_under_a_foreign_root(tmp_path: Path) -> None:
    artifact, dependencies = _tree(tmp_path)
    _rebase_stamp_paths(artifact, OTHER_ROOT)
    dependencies[0].write_bytes(b"features, revised")
    assert not artifact_is_current(artifact, dependencies)


def test_a_dropped_dependency_is_still_stale_under_a_foreign_root(tmp_path: Path) -> None:
    artifact, dependencies = _tree(tmp_path)
    _rebase_stamp_paths(artifact, OTHER_ROOT)
    assert not artifact_is_current(artifact, dependencies[:1])


def test_a_rewritten_artifact_invalidates_its_own_stamp(tmp_path: Path) -> None:
    artifact, dependencies = _tree(tmp_path)
    _rebase_stamp_paths(artifact, OTHER_ROOT)
    artifact.write_bytes(b"expert weights, retrained")
    # The stamp no longer describes what is on disk, so the mtime fallback decides.
    assert artifact_is_current(artifact, dependencies) is True
    assert json.loads(dependency_stamp_path(artifact).read_text())["artifact"] == content_digest(
        artifact
    )


def test_new_stamps_record_repo_relative_keys_for_paths_inside_the_checkout() -> None:
    inside = repo_root() / "state" / "modeling" / "bg_mixture_experts_2025.parquet"
    assert dependency_key(inside) == "state/modeling/bg_mixture_experts_2025.parquet"
    outside = Path("/opt/shared/reference.parquet")
    assert dependency_key(outside) == "/opt/shared/reference.parquet"


def test_the_digest_multiset_ignores_keys_and_not_duplicates() -> None:
    assert digest_multiset({"a": "1", "b": "2"}) == digest_multiset({"/x/b": "2", "/x/a": "1"})
    assert digest_multiset({"a": "1", "b": "1"}) != digest_multiset({"a": "1"})


MIXTURE_ARTIFACT = repo_root() / "state" / "modeling" / "bg_mixture_experts_2025.parquet"


@pytest.mark.skipif(
    not dependency_stamp_path(MIXTURE_ARTIFACT).exists(),
    reason="mixture expert stamp not present in this checkout",
)
def test_the_real_mixture_expert_stamp_decides_the_same_under_any_repo_root(tmp_path: Path) -> None:
    """The stage-10 stamp, on the artifact the trap cost the most on.

    A stamp may be written under one checkout and read under another, and the verdict must
    not depend on that: the same artifact and the same dependency bytes have to decide the
    same way whichever root the stamp names. The real stamp supplies the shape of the case
    -- how many dependencies it records and what their digests are -- and the two stamps
    below differ only in the root their keys name.

    Whether the stamp in THIS checkout happens to hold foreign keys is not the property
    under test: it holds local keys after a local build and foreign ones after a build
    somewhere else, and the invariant has to hold either way.
    """
    stamp_path = dependency_stamp_path(MIXTURE_ARTIFACT)
    recorded = json.loads(stamp_path.read_text())["dependencies"]
    assert recorded, "the mixture expert stamp records no dependencies"

    artifact = tmp_path / MIXTURE_ARTIFACT.name
    artifact.write_bytes(b"stand-in for the reviewed expert table")
    dependencies = []
    for index, digest in enumerate(sorted(recorded.values())):
        # Bytes chosen per recorded dependency; only the KEY changes between the two stamps.
        path = tmp_path / f"dep_{index}.bin"
        path.write_text(digest)
        dependencies.append(path)

    write_dependency_stamp(artifact, dependencies)
    local_verdict = artifact_is_current(artifact, dependencies)
    assert local_verdict is True

    payload = json.loads(dependency_stamp_path(artifact).read_text())
    payload["dependencies"] = {
        key.replace(str(tmp_path), "/some/other/checkout"): digest
        for key, digest in payload["dependencies"].items()
    }
    dependency_stamp_path(artifact).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert artifact_is_current(artifact, dependencies) is local_verdict
