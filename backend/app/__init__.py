"""ETAIQ backend application package."""
import sys
from pathlib import Path

# Add project root directory to sys.path so that we can import "ml.training" etc
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "0.1.0"
