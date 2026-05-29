"""Tests for quick_config_state tokenization + SSE broadcast.

_tokenize / _extract_bigrams are pure. The subscribe/_broadcast tests use
asyncio.Queue objects (instantiable without a running loop) and reset the
module-level subscriber list themselves.
"""

import asyncio

import pytest

import quick_config_state as q


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_empty_and_none():
    assert q._tokenize("") == []
    assert q._tokenize(None) == []


def test_tokenize_drops_short_tokens():
    # single-char tokens are below _TOKEN_MIN_LEN (2)
    assert q._tokenize("a I x") == []


def test_tokenize_drops_overlong_tokens():
    long = "x" * (q._TOKEN_MAX_LEN + 1)
    assert q._tokenize(long) == []
    ok = "y" * q._TOKEN_MAX_LEN
    assert q._tokenize(ok) == [ok]


def test_tokenize_german_umlauts_preserved():
    assert q._tokenize("Müller Ärztin Größe") == ["Müller", "Ärztin", "Größe"]


def test_tokenize_drops_stopwords():
    assert q._tokenize("der die das Patient") == ["Patient"]


def test_tokenize_first_casing_wins_and_dedup():
    out = q._tokenize("Patient patient PATIENT")
    assert out == ["Patient"]


def test_tokenize_insertion_order():
    assert q._tokenize("Alpha Beta Gamma") == ["Alpha", "Beta", "Gamma"]


def test_tokenize_cap():
    words = " ".join(f"w{i:04d}word" for i in range(q._TOKEN_CAP + 30))
    assert len(q._tokenize(words)) == q._TOKEN_CAP


# ---------------------------------------------------------------------------
# _extract_bigrams
# ---------------------------------------------------------------------------

def test_bigrams_empty():
    assert q._extract_bigrams("") == []
    assert q._extract_bigrams(None) == []


def test_bigrams_basic_pair():
    assert q._extract_bigrams("Hans Peter") == ["Hans Peter"]


def test_bigrams_comma_blocks_pair():
    # The comma between Peter and und means no "Peter und" bigram; "und" is
    # a stopword which also blocks "und Anna".
    assert q._extract_bigrams("Hans Peter, und Anna Müller") == [
        "Hans Peter", "Anna Müller"
    ]


def test_bigrams_stopword_blocks():
    assert q._extract_bigrams("Patient mit Befund") == []


def test_bigrams_dedup_first_casing():
    out = q._extract_bigrams("Hans Peter hans peter")
    # "Peter hans" pair is also formed in between; verify dedup keeps first
    # casing of each distinct lowercased phrase.
    assert "Hans Peter" in out
    assert out.count("Hans Peter") == 1


def test_bigrams_whitespace_only_between_ok():
    assert q._extract_bigrams("Hans   Peter") == ["Hans Peter"]


def test_bigrams_cap():
    words = " ".join(f"w{i:04d}x" for i in range(q._BIGRAM_CAP + 30))
    assert len(q._extract_bigrams(words)) == q._BIGRAM_CAP


# ---------------------------------------------------------------------------
# subscribe / unsubscribe / _broadcast
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_subscribers():
    saved = list(q._subscribers)
    q._subscribers.clear()
    yield
    q._subscribers[:] = saved


def test_subscribe_registers_and_unsubscribe_removes():
    sub = q.subscribe()
    assert sub in q._subscribers
    assert sub.maxsize == q._SUBSCRIBER_QUEUE_MAX
    q.unsubscribe(sub)
    assert sub not in q._subscribers


def test_unsubscribe_unknown_is_noop():
    q.unsubscribe(asyncio.Queue())  # not registered; must not raise


def test_broadcast_delivers_to_all():
    a, b = q.subscribe(), q.subscribe()
    q._broadcast({"event": "trace", "data": 1})
    assert a.get_nowait()["data"] == 1
    assert b.get_nowait()["data"] == 1


def test_broadcast_drops_on_full_queue():
    sub = q.subscribe()
    # Fill the queue to its bound.
    for _ in range(q._SUBSCRIBER_QUEUE_MAX):
        sub.put_nowait({"x": 1})
    assert sub.full()
    # Overflow event is dropped silently; subscriber stays registered.
    q._broadcast({"event": "trace"})
    assert sub.qsize() == q._SUBSCRIBER_QUEUE_MAX
    assert sub in q._subscribers
