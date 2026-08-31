from pathlib import Path


def test_mypy_runs_once_against_the_declared_python_311_target() -> None:
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "if: matrix.python-version == '3.11'" in workflow
    assert workflow.count("run: uv run mypy src") == 1
    assert "uses: actions/checkout@v7" in workflow
    assert "uses: astral-sh/setup-uv@v10.0.1" in workflow
