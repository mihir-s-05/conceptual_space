import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

# CSPACE_HOME relocates generated state (data, archive) without moving the code.
HOME = Path(os.environ.get("CSPACE_HOME", ROOT))
DATA_DIR = HOME / "data"
ARCHIVE_DIR = HOME / "archive"

SEEDS_DIR = PACKAGE_DIR / "seeds"
SCHEMA_DIR = PACKAGE_DIR / "schemas"
