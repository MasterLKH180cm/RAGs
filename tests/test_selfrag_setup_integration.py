"""Real file-lock lifecycle with tiny fixtures; no model inference or downloads."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from scripts import selfrag_local as tool


pytestmark = pytest.mark.integration


def test_prepare_rejects_concurrent_owner_before_checking_or_writing_artifacts(monkeypatch, tmp_path):
    content = b"tiny verified model fixture"
    (tmp_path / tool.MODEL_FILENAME).write_bytes(content)
    monkeypatch.setattr(tool, "MODEL_BYTES", len(content))
    monkeypatch.setattr(tool, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    started, release = Event(), Event()
    checks = []

    def verify_runtime(_directory):
        checks.append("runtime")
        started.set()
        assert release.wait(timeout=3), "First preparer was not released"

    monkeypatch.setattr(tool, "verify_runtime", verify_runtime)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(tool.prepare, tmp_path, tmp_path / "runtime", "unused-python")
        try:
            assert started.wait(timeout=3), "First preparer did not acquire its lock"
            with pytest.raises(ValueError, match="already running"):
                tool.prepare(tmp_path, tmp_path / "runtime", "unused-python")
            assert checks == ["runtime"]
        finally:
            release.set()
        assert first.result(timeout=3)["model"]["verified"] is True
    assert (tmp_path / tool.MODEL_FILENAME).read_bytes() == content


def test_preparation_lock_is_released_after_failure_and_process_crash(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        with tool.preparation_lock(tmp_path):
            raise RuntimeError("interrupted")
    with tool.preparation_lock(tmp_path):
        pass
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from scripts.selfrag_local import preparation_lock\n"
        "with preparation_lock(Path(sys.argv[1])):\n"
        "    os._exit(17)\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=5)
    assert result.returncode == 17
    assert (tmp_path / ".selfrag-prepare.lock").is_file()
    with tool.preparation_lock(tmp_path):
        pass
