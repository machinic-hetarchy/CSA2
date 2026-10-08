"""Load parameter values from config/params.yaml."""

from pathlib import Path

import yaml

# Repository root: src/lake/config.py -> src/lake -> src -> root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "params.yaml"


def load_params(path=DEFAULT_CONFIG):
    """Return the contents of the YAML config file as a nested dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)
