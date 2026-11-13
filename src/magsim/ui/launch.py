"""`magsim-ui` console script: `streamlit run src/magsim/ui/app.py` with the project root as cwd
(so .streamlit/config.toml applies). Extra arguments go to `streamlit run`, e.g. `--server.port 8600`."""

import subprocess
import sys
from pathlib import Path

from magsim.ui import ROOT


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    app = Path(__file__).resolve().with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *args]
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
