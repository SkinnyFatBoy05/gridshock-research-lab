from __future__ import annotations

import socket

from gridshock.cli import main


def test_version_command_is_offline(monkeypatch, capsys) -> None:
    """A version check must never open a network connection."""

    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("version command attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbid_network)

    assert main(["version"]) == 0
    assert capsys.readouterr().out == "GridShock Research Lab 0.1.0\n"
