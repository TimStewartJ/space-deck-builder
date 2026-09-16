"""Publish and fetch large result artifacts as GitHub release assets.

Checkpoints and snapshot pools are published as release assets attached to
the git tag that identifies a result, instead of being committed. A bundle is
a deterministic zip whose first entry, ``SHA256SUMS``, lists every contained
file. ``publish`` uploads ``<bundle>.zip`` and ``<bundle>.zip.sha256``;
``fetch`` downloads both, verifies the archive and every entry, and extracts
beneath a destination directory.

Examples::

    python scripts/release_artifacts.py publish \
        --root results/optimized_selfplay_baseline \
        --tag v0.3-selfplay-baseline \
        --bundle optimized_selfplay_baseline-checkpoints

    python scripts/release_artifacts.py fetch \
        --tag v0.3-selfplay-baseline \
        --bundle optimized_selfplay_baseline-checkpoints \
        --dest results/optimized_selfplay_baseline
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

SUMS_ENTRY = "SHA256SUMS"
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


class ArtifactError(RuntimeError):
    """Raised when an artifact bundle is missing, unsafe, or corrupted."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if (
        not name
        or rel.is_absolute()
        or ".." in rel.parts
        or "\\" in name
        or ":" in name
    ):
        raise ArtifactError(f"unsafe artifact path: {name!r}")
    return rel


def collect_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    """Return files under *root* matching any glob pattern, sorted by path."""
    root = root.resolve()
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found[path.relative_to(root).as_posix()] = path
    if not found:
        raise ArtifactError(
            f"no files under {root} match {', '.join(patterns)}"
        )
    return [found[key] for key in sorted(found)]


def _read_published_sums(root: Path) -> dict[str, str]:
    sums_path = root / "ARTIFACTS.sha256"
    if not sums_path.is_file():
        return {}
    published: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            published[name.strip().lstrip("*")] = digest.lower()
    return published


def build_bundle(root: Path, files: Sequence[Path], output_zip: Path) -> str:
    """Write a deterministic bundle and return its SHA-256.

    When *root* contains ``ARTIFACTS.sha256``, every bundled file it lists
    must still match the published hash.
    """
    root = root.resolve()
    published = _read_published_sums(root)
    entries: list[tuple[str, str, Path]] = []
    for path in files:
        rel = path.resolve().relative_to(root).as_posix()
        _safe_relative(rel)
        digest = sha256_file(path)
        expected = published.get(rel)
        if expected is not None and expected != digest:
            raise ArtifactError(
                f"{rel} does not match ARTIFACTS.sha256 ({digest} != {expected})"
            )
        entries.append((rel, digest, path))
    entries.sort(key=lambda item: item[0])
    sums = "".join(f"{digest}  {rel}\n" for rel, digest, _ in entries)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name, payload in [(SUMS_ENTRY, sums.encode("utf-8"))] + [
            (rel, path.read_bytes()) for rel, _, path in entries
        ]:
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compresslevel=6)
    temporary.replace(output_zip)
    digest = sha256_file(output_zip)
    Path(str(output_zip) + ".sha256").write_text(
        f"{digest}  {output_zip.name}\n", encoding="utf-8"
    )
    return digest


def verify_bundle(output_zip: Path) -> dict[str, str]:
    """Verify a bundle and its sidecar hash; return ``{relpath: sha256}``."""
    sidecar = Path(str(output_zip) + ".sha256")
    if not output_zip.is_file() or not sidecar.is_file():
        raise ArtifactError(f"missing bundle or checksum for {output_zip}")
    expected_zip = sidecar.read_text(encoding="utf-8").split()[0].lower()
    actual_zip = sha256_file(output_zip)
    if actual_zip != expected_zip:
        raise ArtifactError(
            f"{output_zip.name} checksum mismatch ({actual_zip} != {expected_zip})"
        )
    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
        if not names or names[0] != SUMS_ENTRY:
            raise ArtifactError(f"{output_zip.name} does not start with {SUMS_ENTRY}")
        listed: dict[str, str] = {}
        for line in archive.read(SUMS_ENTRY).decode("utf-8").splitlines():
            if line.strip():
                digest, rel = line.split(None, 1)
                rel = rel.strip()
                _safe_relative(rel)
                listed[rel] = digest.lower()
        contained = names[1:]
        if len(contained) != len(set(contained)):
            raise ArtifactError(f"{output_zip.name} contains duplicate entries")
        if set(contained) != set(listed):
            raise ArtifactError(
                f"{output_zip.name} entries do not match {SUMS_ENTRY}"
            )
        for rel in contained:
            digest = sha256_bytes(archive.read(rel))
            if digest != listed[rel]:
                raise ArtifactError(f"{rel} checksum mismatch inside bundle")
    return listed


def extract_bundle(output_zip: Path, dest: Path) -> list[Path]:
    """Verify a bundle, then extract its listed files beneath *dest*."""
    listed = verify_bundle(output_zip)
    dest = dest.resolve()
    written: list[Path] = []
    with zipfile.ZipFile(output_zip) as archive:
        for rel in sorted(listed):
            target = (dest / Path(*_safe_relative(rel).parts)).resolve()
            if dest not in target.parents:
                raise ArtifactError(f"unsafe extraction target: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(rel))
            written.append(target)
    return written


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), capture_output=True, text=True)


def _gh(runner: Runner, args: Sequence[str], repo: str | None) -> subprocess.CompletedProcess:
    command = ["gh", *args]
    if repo:
        command += ["--repo", repo]
    return runner(command)


def publish(
    *,
    root: Path,
    tag: str,
    bundle: str,
    patterns: Sequence[str],
    output_dir: Path,
    title: str | None = None,
    notes_file: Path | None = None,
    repo: str | None = None,
    dry_run: bool = False,
    runner: Runner = _default_runner,
) -> Path:
    files = collect_files(root, patterns)
    output_zip = output_dir / f"{bundle}.zip"
    digest = build_bundle(root, files, output_zip)
    print(f"bundle: {output_zip} ({len(files)} files, sha256 {digest})")
    verify_bundle(output_zip)
    if dry_run:
        return output_zip

    view = _gh(runner, ["release", "view", tag], repo)
    if view.returncode != 0:
        create = ["release", "create", tag, "--verify-tag", "--title", title or tag]
        if notes_file is not None:
            create += ["--notes-file", str(notes_file)]
        else:
            create += ["--notes", f"Result artifacts for {tag}."]
        result = _gh(runner, create, repo)
        if result.returncode != 0:
            raise ArtifactError(f"gh release create failed: {result.stderr}")
    upload = _gh(
        runner,
        ["release", "upload", tag, str(output_zip), f"{output_zip}.sha256", "--clobber"],
        repo,
    )
    if upload.returncode != 0:
        raise ArtifactError(f"gh release upload failed: {upload.stderr}")
    print(f"published {output_zip.name} to release {tag}")
    return output_zip


def fetch(
    *,
    tag: str,
    bundle: str,
    dest: Path,
    repo: str | None = None,
    runner: Runner = _default_runner,
) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        result = _gh(
            runner,
            [
                "release", "download", tag,
                "--pattern", f"{bundle}.zip",
                "--pattern", f"{bundle}.zip.sha256",
                "--dir", tmp, "--clobber",
            ],
            repo,
        )
        if result.returncode != 0:
            raise ArtifactError(f"gh release download failed: {result.stderr}")
        written = extract_bundle(Path(tmp) / f"{bundle}.zip", dest)
    print(f"extracted {len(written)} verified files into {dest}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pub = sub.add_parser("publish", help="Bundle artifacts and upload them to a release")
    pub.add_argument("--root", type=Path, required=True)
    pub.add_argument("--tag", required=True)
    pub.add_argument("--bundle", required=True)
    pub.add_argument("--pattern", action="append", dest="patterns")
    pub.add_argument("--output-dir", type=Path, default=Path("dist") / "release")
    pub.add_argument("--title")
    pub.add_argument("--notes-file", type=Path)
    pub.add_argument("--repo")
    pub.add_argument("--dry-run", action="store_true")

    get = sub.add_parser("fetch", help="Download, verify, and extract a bundle")
    get.add_argument("--tag", required=True)
    get.add_argument("--bundle", required=True)
    get.add_argument("--dest", type=Path, required=True)
    get.add_argument("--repo")

    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            publish(
                root=args.root,
                tag=args.tag,
                bundle=args.bundle,
                patterns=args.patterns or ["**/*.pth"],
                output_dir=args.output_dir,
                title=args.title,
                notes_file=args.notes_file,
                repo=args.repo,
                dry_run=args.dry_run,
            )
        else:
            fetch(tag=args.tag, bundle=args.bundle, dest=args.dest, repo=args.repo)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
