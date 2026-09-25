import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
ML = SERVICE.parent
for p in (SERVICE, ML / 'pipeline'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))