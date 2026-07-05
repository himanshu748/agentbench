import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from backend.main import app  # noqa: E402,F401  (Vercel entrypoint)
