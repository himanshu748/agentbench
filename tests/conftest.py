import os
import pathlib
import sys
import tempfile

# Force mock mode and an isolated database before any backend module is imported.
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["AGENTBENCH_DB"] = str(pathlib.Path(tempfile.mkdtemp()) / "test.db")
os.environ.pop("AGENTBENCH_MODELS", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
