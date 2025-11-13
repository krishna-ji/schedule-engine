"""Production mode entry point - runs with all CPU cores."""

import sys

sys.argv.extend(["--config", "configs/cpsat.prod.yaml"])
from main import main

if __name__ == "__main__":
    sys.exit(main())
