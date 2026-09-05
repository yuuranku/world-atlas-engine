"""Location-independent numeric and entity fingerprints."""
import hashlib
import json
from pathlib import Path
import numpy as np


def array_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def semantic_checks(output: Path) -> dict:
    society = json.loads((output / "review/society.json").read_text(encoding="utf-8"))
    society.pop("gridDigest")
    document = json.dumps(society, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"gridArrayDigest": array_digest(output / "grid/world-grid.npz"),
            "societyArrayDigest": array_digest(output / "review/society.npz"),
            "societyDocumentDigest": hashlib.sha256(document.encode()).hexdigest()}
