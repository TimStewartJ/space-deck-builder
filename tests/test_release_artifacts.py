"""Tests for release-asset artifact bundling and verification."""
import hashlib
import subprocess
import zipfile

import pytest

from scripts import release_artifacts as ra


def _write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _artifact_root(tmp_path):
    root = tmp_path / "result"
    _write(root / "checkpoints" / "seed0.pth", b"model-0")
    _write(root / "snapshots" / "seed_0" / "PPO_1.pth", b"snapshot-1")
    _write(root / "summary.json", b"{}")
    return root


def test_bundle_is_deterministic_and_lists_every_file(tmp_path):
    root = _artifact_root(tmp_path)
    files = ra.collect_files(root, ["**/*.pth"])

    first = ra.build_bundle(root, files, tmp_path / "a" / "bundle.zip")
    second = ra.build_bundle(root, files, tmp_path / "b" / "bundle.zip")

    assert first == second
    listed = ra.verify_bundle(tmp_path / "a" / "bundle.zip")
    assert listed == {
        "checkpoints/seed0.pth": hashlib.sha256(b"model-0").hexdigest(),
        "snapshots/seed_0/PPO_1.pth": hashlib.sha256(b"snapshot-1").hexdigest(),
    }


def test_extract_reproduces_files(tmp_path):
    root = _artifact_root(tmp_path)
    bundle = tmp_path / "out" / "bundle.zip"
    ra.build_bundle(root, ra.collect_files(root, ["**/*.pth"]), bundle)

    written = ra.extract_bundle(bundle, tmp_path / "restored")

    assert len(written) == 2
    assert (tmp_path / "restored" / "checkpoints" / "seed0.pth").read_bytes() == b"model-0"
    assert (tmp_path / "restored" / "snapshots" / "seed_0" / "PPO_1.pth").read_bytes() == b"snapshot-1"


def test_verify_rejects_wrong_sidecar_and_tampered_entries(tmp_path):
    root = _artifact_root(tmp_path)
    bundle = tmp_path / "out" / "bundle.zip"
    ra.build_bundle(root, ra.collect_files(root, ["**/*.pth"]), bundle)

    sidecar = tmp_path / "out" / "bundle.zip.sha256"
    sidecar.write_text("0" * 64 + "  bundle.zip\n", encoding="utf-8")
    with pytest.raises(ra.ArtifactError, match="checksum mismatch"):
        ra.verify_bundle(bundle)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "checkpoints/seed0.pth":
                payload = b"evil"
            target.writestr(name, payload)
    (tmp_path / "tampered.zip.sha256").write_text(
        ra.sha256_file(tampered) + "  tampered.zip\n", encoding="utf-8"
    )
    with pytest.raises(ra.ArtifactError, match="checksum mismatch inside bundle"):
        ra.verify_bundle(tampered)


def test_verify_rejects_unsafe_paths(tmp_path):
    unsafe = tmp_path / "unsafe.zip"
    digest = hashlib.sha256(b"x").hexdigest()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr(ra.SUMS_ENTRY, f"{digest}  ../escape.pth\n")
        archive.writestr("../escape.pth", b"x")
    (tmp_path / "unsafe.zip.sha256").write_text(
        ra.sha256_file(unsafe) + "  unsafe.zip\n", encoding="utf-8"
    )

    with pytest.raises(ra.ArtifactError, match="unsafe artifact path"):
        ra.extract_bundle(unsafe, tmp_path / "dest")
    assert not (tmp_path / "escape.pth").exists()


def test_publish_refuses_files_that_differ_from_published_hashes(tmp_path):
    root = _artifact_root(tmp_path)
    (root / "ARTIFACTS.sha256").write_text(
        f"{'0' * 64}  checkpoints/seed0.pth\n", encoding="utf-8"
    )

    with pytest.raises(ra.ArtifactError, match="does not match ARTIFACTS.sha256"):
        ra.publish(
            root=root, tag="v1", bundle="bundle", patterns=["**/*.pth"],
            output_dir=tmp_path / "out", dry_run=True,
            runner=lambda command: pytest.fail("dry run must not call gh"),
        )


def test_publish_creates_missing_release_and_uploads_assets(tmp_path):
    root = _artifact_root(tmp_path)
    calls = []

    def runner(command):
        calls.append(list(command))
        code = 1 if command[1:3] == ["release", "view"] else 0
        return subprocess.CompletedProcess(command, code, "", "")

    ra.publish(
        root=root, tag="v1", bundle="bundle", patterns=["**/*.pth"],
        output_dir=tmp_path / "out", title="Result v1", repo="owner/repo",
        runner=runner,
    )

    assert calls[0] == ["gh", "release", "view", "v1", "--repo", "owner/repo"]
    assert calls[1][:5] == ["gh", "release", "create", "v1", "--verify-tag"]
    upload = calls[2]
    assert upload[:4] == ["gh", "release", "upload", "v1"]
    assert upload[4].endswith("bundle.zip")
    assert upload[5].endswith("bundle.zip.sha256")
    assert "--clobber" in upload


def test_fetch_downloads_verifies_and_extracts(tmp_path):
    import shutil

    root = _artifact_root(tmp_path)
    published = tmp_path / "published"
    ra.build_bundle(root, ra.collect_files(root, ["**/*.pth"]), published / "bundle.zip")

    def runner(command):
        target = command[command.index("--dir") + 1]
        for name in ("bundle.zip", "bundle.zip.sha256"):
            shutil.copy(published / name, target)
        return subprocess.CompletedProcess(command, 0, "", "")

    written = ra.fetch(tag="v1", bundle="bundle", dest=tmp_path / "dest", runner=runner)

    assert sorted(p.name for p in written) == ["PPO_1.pth", "seed0.pth"]
