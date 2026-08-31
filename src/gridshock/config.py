"""Typed project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by deterministic project commands."""

    root: Path
    demo_data: Path
    demo_manifest: Path
    reports: Path
    figures: Path
    experiment_manifest: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> ProjectPaths:
        """Resolve paths from an explicit root or the installed source tree."""

        project_root = (root or Path(__file__).resolve().parents[2]).resolve()
        reports = project_root / "reports"
        return cls(
            root=project_root,
            demo_data=project_root / "data" / "demo" / "gridshock_demo.csv.gz",
            demo_manifest=project_root / "data" / "demo" / "manifest.json",
            reports=reports,
            figures=reports / "figures",
            experiment_manifest=reports / "experiment_manifest.json",
        )
