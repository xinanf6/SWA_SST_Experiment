"""Use ordinary workspace directories instead of Windows mode-0700 temp ACLs."""

from pathlib import Path
import shutil
import uuid

import pytest


@pytest.fixture
def tmp_path():
    """Yield a bounded test directory and clean only that verified directory."""
    root = (Path(__file__).resolve().parents[1] / "output" / "offline_tests").resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / uuid.uuid4().hex
    target.mkdir()
    try:
        yield target
    finally:
        resolved = target.resolve()
        if resolved.parent != root:
            raise RuntimeError("Refusing cleanup outside offline test workspace")
        shutil.rmtree(str(resolved))
