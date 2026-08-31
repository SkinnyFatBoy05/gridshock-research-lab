from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def test_wheel_builds_with_exactly_one_report_template(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert archive.namelist().count("gridshock/templates/report.html.j2") == 1
