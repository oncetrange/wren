from pathlib import Path

import pytest

from wren.checkpoint import Checkpoints


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "keep.py").write_text("v1\n")
    (root / "gone.py").write_text("delete me\n")
    return root


def test_restore_reverts_modifies_deletes_and_recreates(project: Path, tmp_path: Path):
    cps = Checkpoints(project, root=tmp_path / "shadow")
    assert cps.enabled
    snap = cps.snapshot("first")

    (project / "keep.py").write_text("v2\n")
    (project / "gone.py").unlink()
    (project / "new").mkdir()
    (project / "new/file.py").write_text("created\n")

    assert sorted(cps.changed_files(snap)) == ["A new/file.py", "D gone.py", "M keep.py"]

    cps.restore(snap)
    assert (project / "keep.py").read_text() == "v1\n"
    assert (project / "gone.py").read_text() == "delete me\n"
    assert not (project / "new").exists()  # emptied directories are removed too


def test_respects_gitignore_and_leaves_project_git_alone(project: Path, tmp_path: Path):
    (project / ".gitignore").write_text("secret.txt\n")
    (project / "node_modules").mkdir()
    (project / "node_modules/lib.js").write_text("x")
    cps = Checkpoints(project, root=tmp_path / "shadow")
    snap = cps.snapshot("p")
    (project / "secret.txt").write_text("not tracked")
    (project / "node_modules/lib.js").write_text("y")
    assert cps.changed_files(snap) == []
    cps.restore(snap)
    assert (project / "secret.txt").read_text() == "not tracked"
    assert not (project / ".git").exists()


def test_restore_to_an_earlier_snapshot(project: Path, tmp_path: Path):
    cps = Checkpoints(project, root=tmp_path / "shadow")
    first = cps.snapshot("a")
    (project / "keep.py").write_text("v2\n")
    cps.snapshot("b")
    (project / "keep.py").write_text("v3\n")
    cps.restore(first)
    assert (project / "keep.py").read_text() == "v1\n"


def test_disabled_for_home_directory(tmp_path: Path):
    assert not Checkpoints(Path.home(), root=tmp_path / "shadow").enabled


def test_shadow_repo_inside_work_tree_is_not_snapshotted(project: Path):
    cps = Checkpoints(project, root=project / ".shadow")
    snap = cps.snapshot("p")
    (project / "keep.py").write_text("v2\n")
    assert cps.changed_files(snap) == ["M keep.py"]
    cps.restore(snap)
    assert (project / "keep.py").read_text() == "v1\n"
    assert (cps.git_dir / "HEAD").exists()


def test_restore_in_empty_project(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cps = Checkpoints(empty, root=tmp_path / "shadow")
    snap = cps.snapshot("p")
    (empty / "new.txt").write_text("x")
    cps.restore(snap)
    assert list(empty.iterdir()) == []
    cps.restore(cps.snapshot("again"))  # nothing tracked on either side
