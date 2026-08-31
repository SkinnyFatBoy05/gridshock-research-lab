"""Command-line interface for GridShock Research Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gridshock import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free command parser."""

    parser = argparse.ArgumentParser(
        prog="gridshock",
        description="Point-in-time-safe DE-LU power-market research.",
    )
    parser.add_argument("command", choices=("version",))
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process status."""

    parsed = build_parser().parse_args(args)
    if parsed.command == "version":
        print(f"GridShock Research Lab {__version__}")
        return 0
    return 2


def entrypoint() -> None:
    """Console-script adapter."""

    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
