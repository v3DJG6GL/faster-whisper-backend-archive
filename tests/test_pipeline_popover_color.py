"""UI fixes for /settings/pipeline + /quick-config:

  * shared tag-picker is a top-layer-popover combobox (escapes overflow:hidden)
  * colour palette is a popover too
  * lock-vs-colour CSS precedence (colour wins; locked shown by a chip)
  * lock icon is crisp inline SVG, not the 🔒/🔓 emoji
  * per-rule colours are mirrored onto /quick-config

These are client-rendered, so the assertions check the served HTML/JS/CSS
strings (same approach as test_routes_pipeline_page.py). A final test runs the
inline scripts through `node --check` to catch syntax breakage inside the big
Python-string templates (py_compile can't see into them)."""

import copy
import re
import shutil
import subprocess

import pytest


def _expose_first_regex_list_rule(app_module, color=None):
    """Expose (and optionally colour) the first regex-list rule so it shows on
    /quick-config. Mirrors the helper in test_routes_quick_config.py."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex-list":
            r["exposed"] = True
            if color is not None:
                r["color"] = color
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug


# ---- /quick-config colour mirroring (bug 2.1) --------------------------------

def test_quick_config_state_includes_color(client, app_module):
    slug = _expose_first_regex_list_rule(app_module, color="blue")
    assert slug is not None
    r = client.get("/quick-config/state")
    assert r.status_code == 200
    rules = {rd["name"]: rd for rd in r.json()["rules"]}
    assert slug in rules
    assert rules[slug].get("color") == "blue"


def test_quick_config_page_applies_and_styles_color(client):
    r = client.get("/quick-config")
    assert r.status_code == 200
    # client reads rule.color onto the card …
    assert "card.dataset.color = rule.color" in r.text
    # … and the colour-token CSS exists for it (border + wash).
    assert '.card[data-color="blue"]' in r.text
    assert '.card[data-color="pink"]' in r.text


# ---- pipeline popovers + combobox (bugs 1.1/1.2/1.3) -------------------------

def test_pipeline_uses_top_layer_popover_helper(client):
    t = client.get("/settings/pipeline").text
    assert "_anchorPopover" in t
    assert "showPopover" in t            # native Popover API
    # colour palette popover
    assert "color-pop" in t
    assert "popover" in t


def test_pipeline_tag_picker_is_aria_combobox(client):
    # TAG_PICKER_JS is injected into the pipeline page.
    t = client.get("/settings/pipeline").text
    assert "aria-autocomplete" in t
    assert "aria-activedescendant" in t
    assert "ArrowDown" in t and "ArrowUp" in t


# ---- lock-vs-colour precedence + SVG icon (bug 1.4) -------------------------

def test_pipeline_lock_color_precedence_scoped(client):
    t = client.get("/settings/pipeline").text
    # Locked rows are no longer tinted at all: the yellow row tint reused the
    # per-card colour mechanism and made an un-coloured locked rule read as
    # "coloured yellow". The lock chip alone signals locked, so the colour
    # token always owns the rail/border — there must be no locked row tint.
    assert ".rule-row.locked:not([data-color])" not in t
    # Terminal keeps a neutral accent, still scoped so a colour token wins.
    assert ".rule-row.terminal:not([data-color])" in t


def test_pipeline_color_palette_hidden_until_opened(client):
    # The colour palette is a [popover]; an unconditional author `display`
    # would defeat the UA rule that hides a closed popover, leaving it expanded
    # inline — clipped by the card's overflow:hidden and covering the expand
    # toggle. The closed state is re-asserted at higher specificity.
    t = client.get("/settings/pipeline").text
    assert ".color-pop:not(:popover-open) { display: none; }" in t


def test_pipeline_lock_icon_is_svg_not_emoji(client):
    t = client.get("/settings/pipeline").text
    # verbatim Bootstrap 'lock' path fragment replaces the emoji …
    assert "M8 0a4 4 0 0 1 4 4v2.05" in t
    # … and no lock emoji survive anywhere on the page.
    assert "🔒" not in t and "🔓" not in t


# ---- syntax guard over the inline scripts -----------------------------------

_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node unavailable for JS syntax check")
@pytest.mark.parametrize("url", ["/settings/pipeline", "/quick-config", "/settings"])
def test_inline_scripts_parse(client, url, tmp_path):
    html = client.get(url).text
    blocks = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S)
    assert blocks, "expected inline scripts on " + url
    errors = []
    for i, (attrs, code) in enumerate(blocks):
        if "src=" in attrs:
            continue
        m = re.search(r'type\s*=\s*["\']([^"\']+)["\']', attrs)
        if m and "javascript" not in m.group(1).lower():
            continue  # JSON / template blocks aren't JS
        if not code.strip():
            continue
        f = tmp_path / f"s{i}.js"
        # Force UTF-8: the scripts contain non-Latin-1 chars (e.g. ä≈a in the
        # collator comment), and Path.write_text() defaults to the platform
        # locale — cp1252 on Windows CI — which raises UnicodeEncodeError.
        f.write_text(code, encoding="utf-8")
        p = subprocess.run([_NODE, "--check", str(f)], capture_output=True, text=True)
        if p.returncode != 0:
            tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "?"
            errors.append(f"{url} block {i}: {tail}")
    assert not errors, "JS syntax errors:\n" + "\n".join(errors)
