"""Read binary LIBSVM data without reordering samples or scaling features."""

from pathlib import Path

import numpy as np


def load_libsvm(path):
    rows = []
    labels = []
    n_features = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            label = float(fields[0])
            if label not in (-1, 1):
                raise ValueError(f"Line {line_number}: expected a binary label (-1 or +1)")
            row = []
            last_index = 0
            for field in fields[1:]:
                index_text, value_text = field.split(":", 1)
                index, value = int(index_text), float(value_text)
                if index <= last_index or not np.isfinite(value):
                    raise ValueError(f"Line {line_number}: invalid feature {field!r}")
                row.append((index - 1, value))
                last_index = index
            n_features = max(n_features, last_index)
            rows.append(row)
            labels.append(label)
    if not rows:
        raise ValueError("The dataset contains no samples")
    X = np.zeros((len(rows), n_features), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, value in row:
            X[row_index, feature_index] = value
    return X, np.asarray(labels, dtype=np.int8)
