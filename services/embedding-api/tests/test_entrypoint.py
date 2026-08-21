from __future__ import annotations

import pytest

from embedding_api.__main__ import _server_port


def test_server_port_defaults_to_container_port(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_PORT", raising=False)

    assert _server_port() == 8001


@pytest.mark.parametrize("value", ["zero", "0", "65536"])
def test_server_port_fails_fast(monkeypatch, value: str) -> None:
    monkeypatch.setenv("EMBEDDING_PORT", value)

    with pytest.raises(ValueError, match="EMBEDDING_PORT"):
        _server_port()
