"""Prepare official archives from a local folder and run the full paired benchmark."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from zipfile import ZipFile

from author_source import SOURCE_FILES, ZIG_RELEASE, ZIG_SHA256, choose_compiler
from experiment_io import ROOT, sha256, write_json
from run_author import load_full


def stage_file(original, destination):
    if destination.is_file():
        if sha256(original) != sha256(destination):
            raise ValueError(f"Existing staged file differs from {original.name}: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    shutil.copyfile(original, partial)
    partial.replace(destination)


def stage_source(archive, destination):
    archive_hash = sha256(archive)
    source_dir = destination / archive_hash
    with ZipFile(archive) as package:
        members = {}
        for entry in package.infolist():
            parts = PurePosixPath(entry.filename).parts
            if not entry.is_dir() and len(parts) == 2 and parts[1] in SOURCE_FILES:
                if parts[1] in members:
                    raise ValueError(f"Duplicate source file in {archive.name}: {parts[1]}")
                members[parts[1]] = entry
        if set(members) != set(SOURCE_FILES):
            raise ValueError(f"Expected the five libsvm_cd source files in {archive.name}.")
        roots = {PurePosixPath(entry.filename).parts[0] for entry in members.values()}
        if len(roots) != 1:
            raise ValueError(f"Expected a single source folder in {archive.name}.")
        source_dir.mkdir(parents=True, exist_ok=True)
        for name, entry in members.items():
            data = package.read(entry)
            target = source_dir / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Staged source file changed: {target}")
            target.write_bytes(data)
        comment = package.comment.decode("ascii", errors="ignore").strip()
    write_json(source_dir / "archive_provenance.json", {
        "archive": archive.name,
        "archive_sha256": archive_hash,
        "archive_comment": comment,
    })
    return source_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("--source", type=Path, help="Existing author checkout or source ZIP.")
    parser.add_argument("--cxx", help="Existing g++, clang++ or Zig executable.")
    arguments = parser.parse_args()
    inputs_dir = arguments.inputs_dir.resolve()
    if not inputs_dir.is_dir():
        parser.error(f"Input folder does not exist: {inputs_dir}")

    work = ROOT / "work"
    data_dir = work / "data"
    toolchains = work / "toolchain"
    missing = []
    inputs = {}
    for name in ("ijcnn1", "ijcnn1.t"):
        path = next((p for p in (inputs_dir / name, inputs_dir / (name + ".bz2"),
                                data_dir / name, data_dir / (name + ".bz2")) if p.is_file()), None)
        if path is None:
            missing.append(f"{name}.bz2")
        else:
            inputs[name] = path

    source = arguments.source
    if source is None:
        source = next((p for p in (inputs_dir / "libsvm_cd", inputs_dir / "libsvm_cd-master",
                                  inputs_dir / "libsvm_cd-master.zip", inputs_dir / "libsvm_cd.zip",
                                  work / "libsvm_cd") if p.exists()), None)
    if source is None or not source.exists():
        missing.append("libsvm_cd-master.zip")

    cxx = arguments.cxx
    if cxx is None:
        cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("zig")
    portable = toolchains / ZIG_RELEASE / "zig.exe"
    zig_archive = inputs_dir / (ZIG_RELEASE + ".zip")
    if cxx is None and portable.is_file():
        cxx = str(portable)
    if cxx is None and not (zig_archive.is_file() or (toolchains / zig_archive.name).is_file()):
        missing.append(zig_archive.name)
    if missing:
        parser.exit(2, "Missing local inputs: " + ", ".join(missing) + "\n")

    for path in inputs.values():
        if path.parent != data_dir:
            stage_file(path, data_dir / path.name)
    train, labels, test, n_test = load_full(data_dir)
    print(f"Verified official files: {train.name} ({len(labels)}) and {test.name} ({n_test})", flush=True)

    if source.is_file():
        source = stage_source(source, work / "author_archives")
    if cxx is None:
        cached_archive = toolchains / zig_archive.name
        if zig_archive.is_file():
            stage_file(zig_archive, cached_archive)
        if sha256(cached_archive) != ZIG_SHA256:
            raise ValueError("Portable Zig archive differs from the published SHA-256.")
        cxx = choose_compiler(None, toolchains)

    command = [sys.executable, str(ROOT / "run_author.py"), "--source", str(source.resolve()),
               "--data-dir", str(data_dir), "--cxx", cxx]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
