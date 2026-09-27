"""Test suite for the churn pipeline.

Run from the repository root:

    .venv\\Scripts\\python.exe -m unittest discover -v

The stage-2 tests import `ModelTuning` from the repository root. Putting that on
`sys.path` here (rather than relying on how discovery was invoked) keeps the suite
working under `python -m unittest discover`, under `-s tests`, and when a single
test module is run directly.

Nothing in here may take minutes. The full tuning run is ~21 min by design; if a
test needs to exercise it end to end, it does so on a subsample with a tiny grid.
"""

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextlib.contextmanager
def quiet():
    """Swallow stdout from code under test.

    `ModelTuning` prints progress and per-candidate notes; that is right for a real
    run and noise in a test report. Use this around calls into it.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer
