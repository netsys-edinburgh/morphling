import subprocess
import sys


def test_atexit_fires_on_exception():
    script = (
        "import atexit, sys\n"
        "def _cleanup():\n"
        "    sys.stderr.write('ATEXIT_FIRED\\n')\n"
        "atexit.register(_cleanup)\n"
        "raise RuntimeError('deliberate crash')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "ATEXIT_FIRED" in result.stderr
