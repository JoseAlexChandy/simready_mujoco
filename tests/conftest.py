import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

from build import build_cabinet  # noqa: E402


@pytest.fixture
def cabinet_usd(tmp_path: Path) -> Path:
    """Fresh copy of the synthetic cabinet asset, written to a temp folder."""
    path = tmp_path / "cabinet.usda"
    build_cabinet(path)
    return path
