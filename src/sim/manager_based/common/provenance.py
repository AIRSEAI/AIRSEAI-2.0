"""Source hashes for runs made from an uncommitted worktree."""

import hashlib
import json
import platform
from pathlib import Path


def write_manifest(path, **metadata):
    root = Path(__file__).resolve().parents[4]
    files = sorted(
        [
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("benchmark/run").glob("*.py"),
            *root.joinpath("benchmark/dev/vector_env").glob("*.py"),
        ]
    )
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    data = dict(python=platform.python_version(), sources=sources, **metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
