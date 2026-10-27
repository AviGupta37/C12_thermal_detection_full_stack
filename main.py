"""
Root entry point forwarding directly to c12_thermal_detection.main
"""

import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "c12_thermal_detection"
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from c12_thermal_detection.main import main

if __name__ == "__main__":
    sys.exit(main())

