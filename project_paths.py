"""Resolve project resources independently of the process working directory."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(
    path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve absolute paths as given and relative paths from the project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate.resolve()


def persist_uploaded_dataset(contents: bytes) -> Path:
    """Cache an uploaded dataset ZIP outside the repository by its content hash."""
    if not contents:
        raise ValueError("Uploaded dataset ZIP is empty.")

    digest = hashlib.sha256(contents).hexdigest()
    upload_directory = Path(tempfile.gettempdir()) / "hybrid-cnn-inverter" / "uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / f"{digest}.zip"
    if destination.is_file():
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=upload_directory,
            prefix=f"{digest}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination
