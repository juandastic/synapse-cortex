import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """ASGI and mocked providers are allowed; real service calls fail immediately."""

    def blocked(*args, **kwargs):
        raise AssertionError(
            "Unexpected network connection. Stub the external service in this test."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
