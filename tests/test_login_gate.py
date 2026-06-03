"""The unified login gate lives in the shared chrome (web_common) and ships on
every WebUI page; the old per-page #token-modal / #login-wrap login UIs are gone.

In OPEN mode (the default `client` fixture, loopback) every page renders, so we
can assert on the served HTML directly."""

import pytest

# (path, current-page label) for every WebUI page that renders shared chrome.
_PAGES = [
    "/stats",
    "/logs",
    "/quick-config",
    "/captures",
    "/reports",
    "/settings",
    "/settings/api-keys",
]


@pytest.mark.parametrize("path", _PAGES)
def test_page_ships_shared_login_gate(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code)
    html = r.text
    # The shared gate markup + API ride the OPEN_MODE_BANNER_JS chrome.
    assert 'id="login-gate"' in html, f"{path} missing the shared login gate"
    assert "_showLoginGate" in html, f"{path} missing the gate API"
    assert "lg-mark" in html, f"{path} missing the waveform brand mark"


@pytest.mark.parametrize("path", _PAGES)
def test_page_has_no_legacy_login_ui(client, path):
    html = client.get(path).text
    # The per-page login UIs were removed in favour of the shared gate.
    assert 'id="token-modal"' not in html, f"{path} still has a #token-modal"
    assert 'id="login-wrap"' not in html, f"{path} still has the #login-wrap card"
    assert 'id="login-token"' not in html, f"{path} still has the login-card input"
