"""Read the fixed LIBSVM files and write the Group 19 output formats."""

import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ("heart_scale", "ijcnn1_seed19")
HISTORY_FIELDS = ("iteration", "dual_objective", "kkt_violation", "equality_residual", "box_violation")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, *, expected_sha256=None):
    if destination.is_file():
        if expected_sha256 and sha256(destination) != expected_sha256:
            raise ValueError(f"Unexpected SHA-256 for {destination}.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    downloader = shutil.which("curl.exe") if os.name == "nt" else None
    if downloader is not None:
        command = [downloader, "--fail", "--location", "--silent", "--show-error",
                   "--connect-timeout", "18", "--max-time", "240", "--output", str(partial), url]
        if subprocess.run(command, check=False).returncode:
            raise RuntimeError(f"Could not download {destination.name} from {url}.")
    else:
        with urlopen(url, timeout=30) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if expected_sha256 and sha256(partial) != expected_sha256:
        raise ValueError(f"Downloaded file does not match the expected SHA-256 for {destination.name}.")
    partial.replace(destination)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_libsvm(path, n_features):
    rows = path.read_bytes().splitlines()
    X = np.zeros((len(rows), n_features), dtype=np.float64)
    y = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        label, *entries = row.split()
        y[i] = float(label)
        previous = 0
        for entry in entries:
            index, value = entry.split(b":")
            index = int(index)
            if not previous < index <= n_features:
                raise ValueError(f"Invalid feature index on input line {i + 1}.")
            X[i, index - 1] = float(value)
            previous = index
    return X, y


def load_fixed(dataset):
    config = read_json(ROOT / "reference" / dataset / "config.json")
    folder = ROOT / "inputs" / dataset
    file = folder / "subset.libsvm"
    expected = config.get("subset_hash", config["dataset_hash"])
    if sha256(file) != expected:
        raise ValueError(f"Fixed input SHA-256 mismatch for {dataset}.")
    if config.get("indices_hash") and sha256(folder / "indices.npy") != config["indices_hash"]:
        raise ValueError(f"Fixed indices SHA-256 mismatch for {dataset}.")
    X, y = load_libsvm(file, config["n_features"])
    if len(y) != config["n_train"]:
        raise ValueError(f"Fixed input length mismatch for {dataset}.")
    return config, X, y


def machine():
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
