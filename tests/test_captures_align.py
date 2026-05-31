"""Regression tests for captures_routes._align_words_to_final.

The merge / proposal FINAL RESULT is rebuilt word-by-word from this alignment
(via _renderGroundSpans), so the alignment must assign EVERY post-processed
token to some raw word — otherwise tokens that a rule *inserts* (e.g. the raw
word "Nurtax" expanding to "nur tags") silently disappear from the merged final
while still being present in the exported text. See the bug write-up: a 1->many
word-expanding rule dropped the expansion from the group preview.

These tests pin the invariant `" ".join(non-removed word) == final` across the
word-count-changing shapes: 1->1, 1->many, many->1, 1->0.

The alignment computes each raw word's post-processed form via
main._postprocess_text(); we monkeypatch it to a deterministic per-word map so
the cases are hermetic (no real pipeline / config needed).
"""

import main
import captures_routes as cr


def _words(*pairs):
    """Build a words_json-style list with synthetic but monotonic timings."""
    out = []
    for i, w in enumerate(pairs):
        out.append({"word": w, "start": float(i), "end": float(i) + 0.5})
    return out


def _join(out):
    """Reconstruct the displayed final from non-removed entries."""
    return " ".join(w["word"] for w in out if not w.get("removed"))


def _install_word_map(monkeypatch, mapping):
    """Fake main._postprocess_text: per-word lookup, identity by default."""
    def fake(text, model_name=None):
        return mapping.get(text.strip(), text)
    monkeypatch.setattr(main, "_postprocess_text", fake)


# ---------------------------------------------------------------------------
# 1 -> many expansion (the reported bug: "Nurtax" -> "nur tags")
# ---------------------------------------------------------------------------

def test_one_to_many_expansion_keeps_all_tokens(monkeypatch):
    _install_word_map(monkeypatch, {"Nurtax": "nur tags"})
    words = _words("Nurtax", "137", "Schrägstrich", "94")
    final = "nur tags 137 Schrägstrich 94"
    out = cr._align_words_to_final(words, final)

    assert len(out) == len(words)              # one entry per raw word
    assert _join(out) == final                 # no token dropped
    # The expanded word carries the joined run and flags the raw form.
    assert out[0]["word"] == "nur tags"
    assert not out[0].get("removed")
    assert out[0].get("raw_word") == "Nurtax"
    # Untouched words stay put.
    assert out[1]["word"] == "137"
    assert out[3]["word"] == "94"


def test_one_to_three_expansion(monkeypatch):
    # Compound-splitter style: one raw word becomes three tokens.
    _install_word_map(monkeypatch, {"Eisenbindestrichinfusion":
                                    "Eisen Bindestrich infusion"})
    words = _words("Eisenbindestrichinfusion", "heute")
    final = "Eisen Bindestrich infusion heute"
    out = cr._align_words_to_final(words, final)

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Eisen Bindestrich infusion"
    assert not out[0].get("removed")
    assert out[1]["word"] == "heute"


def test_expansion_at_end(monkeypatch):
    # Trailing raw word expands — exercises the after-last-anchor segment.
    _install_word_map(monkeypatch, {"Nurtax": "nur tags"})
    words = _words("Hallo", "Nurtax")
    final = "Hallo nur tags"
    out = cr._align_words_to_final(words, final)
    assert len(out) == 2
    assert _join(out) == final
    assert out[1]["word"] == "nur tags"
    assert not out[1].get("removed")


# ---------------------------------------------------------------------------
# many -> 1 contraction
# ---------------------------------------------------------------------------

def test_many_to_one_contraction(monkeypatch):
    # Two raw words merge into one final token: the survivor attaches to the
    # first raw word, the second is removed — no token lost or duplicated.
    _install_word_map(monkeypatch, {})  # identity; neither matches "Bindestrich"
    words = _words("Bind", "strich")
    final = "Bindestrich"
    out = cr._align_words_to_final(words, final)

    assert len(out) == 2
    assert _join(out) == final
    assert out[0]["word"] == "Bindestrich"
    assert out[1].get("removed") is True


def test_symbol_substitution_keeps_token(monkeypatch):
    # Dictation-map style "Schrägstrich" -> "/": a 1->1 symbol swap whose key
    # normalises to empty must still attach the symbol, not drop it.
    _install_word_map(monkeypatch, {"Schrägstrich": "/"})
    words = _words("137", "Schrägstrich", "94")
    final = "137 / 94"
    out = cr._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "/"
    assert not out[1].get("removed")


# ---------------------------------------------------------------------------
# 1 -> 0 deletion
# ---------------------------------------------------------------------------

def test_deletion_marks_removed(monkeypatch):
    _install_word_map(monkeypatch, {"äh": ""})
    words = _words("äh", "hallo", "welt")
    final = "hallo welt"
    out = cr._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[0].get("removed") is True
    assert out[1]["word"] == "hallo"
    assert out[2]["word"] == "welt"


# ---------------------------------------------------------------------------
# plain 1 -> 1 (no count change)
# ---------------------------------------------------------------------------

def test_plain_substitution(monkeypatch):
    _install_word_map(monkeypatch, {"weiss": "weiß"})
    words = _words("das", "weiss", "ich")
    final = "das weiß ich"
    out = cr._align_words_to_final(words, final)

    assert len(out) == 3
    assert _join(out) == final
    assert out[1]["word"] == "weiß"
    assert out[1].get("raw_word") == "weiss"
    assert not out[1].get("removed")


def test_identity_no_flags(monkeypatch):
    _install_word_map(monkeypatch, {})
    words = _words("eins", "zwei", "drei")
    final = "eins zwei drei"
    out = cr._align_words_to_final(words, final)

    assert _join(out) == final
    for w in out:
        assert not w.get("removed")
        assert "raw_word" not in w  # display == raw → no diff marker


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def test_output_length_always_matches_raw_count(monkeypatch):
    _install_word_map(monkeypatch, {"Nurtax": "nur tags", "äh": ""})
    words = _words("äh", "Nurtax", "137")
    final = "nur tags 137"
    out = cr._align_words_to_final(words, final)
    assert len(out) == len(words)   # chip word-indices depend on this
    assert _join(out) == final


def test_empty_words_returns_empty(monkeypatch):
    _install_word_map(monkeypatch, {})
    assert cr._align_words_to_final([], "anything") == []
