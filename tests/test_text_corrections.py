"""Exhaustive tests for text_corrections (pure, no I/O).

Covers clean_corrections (field caps, idx/idx_end range rules, list cap,
malformed tolerance) and three_way_merge_corrections (anchored delta rules,
anchorless union/dedup, ordering).
"""

import text_corrections as tc


# ---------------------------------------------------------------------------
# clean_corrections
# ---------------------------------------------------------------------------

def test_none_and_empty():
    assert tc.clean_corrections(None) == []
    assert tc.clean_corrections([]) == []


def test_non_dict_entries_dropped():
    assert tc.clean_corrections(["x", 5, None, ("a", "b")]) == []


def test_requires_nonempty_correct():
    assert tc.clean_corrections([{"wrong": "a", "correct": ""}]) == []
    assert tc.clean_corrections([{"wrong": "a", "correct": "   "}]) == []
    assert tc.clean_corrections([{"wrong": "a"}]) == []


def test_basic_entry_keeps_wrong_and_correct():
    out = tc.clean_corrections([{"wrong": " a ", "correct": " b "}])
    assert out == [{"wrong": "a", "correct": "b"}]


def test_wrong_may_be_empty_when_correct_present():
    out = tc.clean_corrections([{"correct": "b"}])
    assert out == [{"wrong": "", "correct": "b"}]


def test_field_cap_200():
    long = "x" * 250
    out = tc.clean_corrections([{"wrong": long, "correct": long}])
    assert len(out[0]["wrong"]) == tc.CAP_CORRECTION_FIELD == 200
    assert len(out[0]["correct"]) == 200


def test_idx_valid_range():
    assert tc.clean_corrections([{"correct": "b", "idx": 0}])[0]["idx"] == 0
    assert tc.clean_corrections([{"correct": "b", "idx": 9999}])[0]["idx"] == 9999


def test_idx_out_of_range_dropped():
    assert "idx" not in tc.clean_corrections([{"correct": "b", "idx": -1}])[0]
    assert "idx" not in tc.clean_corrections([{"correct": "b", "idx": 10_000}])[0]


def test_idx_non_int_dropped():
    assert "idx" not in tc.clean_corrections([{"correct": "b", "idx": 1.5}])[0]
    assert "idx" not in tc.clean_corrections([{"correct": "b", "idx": "3"}])[0]
    # bool is an int subclass but 0/1 are still in-range ints; True == 1.
    assert tc.clean_corrections([{"correct": "b", "idx": True}])[0]["idx"] is True


def test_idx_end_requires_greater_than_idx():
    out = tc.clean_corrections([{"correct": "b", "idx": 2, "idx_end": 5}])[0]
    assert out["idx_end"] == 5


def test_idx_end_equal_to_idx_dropped():
    out = tc.clean_corrections([{"correct": "b", "idx": 2, "idx_end": 2}])[0]
    assert "idx_end" not in out


def test_idx_end_less_than_idx_dropped():
    out = tc.clean_corrections([{"correct": "b", "idx": 5, "idx_end": 3}])[0]
    assert "idx_end" not in out


def test_idx_end_out_of_range_dropped():
    out = tc.clean_corrections([{"correct": "b", "idx": 2, "idx_end": 10_000}])[0]
    assert "idx_end" not in out


def test_idx_end_without_idx_dropped():
    out = tc.clean_corrections([{"correct": "b", "idx_end": 5}])[0]
    assert "idx" not in out and "idx_end" not in out


def test_list_cap_50():
    items = [{"correct": f"c{i}"} for i in range(80)]
    out = tc.clean_corrections(items)
    assert len(out) == tc.CAP_CORRECTIONS == 50


def test_unicode_preserved():
    out = tc.clean_corrections([{"wrong": "Müller", "correct": "Mueller"}])
    assert out[0]["wrong"] == "Müller"


# ---------------------------------------------------------------------------
# three_way_merge_corrections
# ---------------------------------------------------------------------------

def _chip(wrong, correct, idx=None, idx_end=None):
    c = {"wrong": wrong, "correct": correct}
    if idx is not None:
        c["idx"] = idx
    if idx_end is not None:
        c["idx_end"] = idx_end
    return c


def test_merge_all_none():
    assert tc.three_way_merge_corrections(None, None, None) == []


def test_merge_user_added_anchored():
    # edited has a chip not in baseline -> inserted.
    edited = [_chip("a", "b", idx=0)]
    out = tc.three_way_merge_corrections([], edited, [])
    assert out == edited


def test_merge_user_removed_anchored():
    # chip in baseline but not edited -> removed from current.
    base = [_chip("a", "b", idx=0)]
    cur = [_chip("a", "b", idx=0)]
    out = tc.three_way_merge_corrections(base, [], cur)
    assert out == []


def test_merge_user_edited_wins_over_concurrent():
    base = [_chip("a", "b", idx=0)]
    edited = [_chip("a", "BB", idx=0)]
    cur = [_chip("a", "CONCURRENT", idx=0)]
    out = tc.three_way_merge_corrections(base, edited, cur)
    assert out == edited


def test_merge_untouched_keeps_concurrent():
    # key in both base & edited with equal payload -> user untouched ->
    # keep concurrent value from current.
    base = [_chip("a", "b", idx=0)]
    edited = [_chip("a", "b", idx=0)]
    cur = [_chip("a", "CONCURRENT", idx=0)]
    out = tc.three_way_merge_corrections(base, edited, cur)
    assert out == cur


def test_merge_idx_end_part_of_key():
    # Same idx but different idx_end are distinct keys.
    base = []
    edited = [_chip("a", "b", idx=0, idx_end=2), _chip("a", "c", idx=0)]
    out = tc.three_way_merge_corrections(base, edited, [])
    assert len(out) == 2


def test_merge_anchorless_union_dedup():
    cur = [_chip("x", "y"), _chip("p", "q")]
    edited = [_chip("p", "q"), _chip("m", "n")]
    out = tc.three_way_merge_corrections([], edited, cur)
    # union deduped by (wrong, correct): x/y, p/q, m/n -> 3
    pairs = {(c["wrong"], c["correct"]) for c in out}
    assert pairs == {("x", "y"), ("p", "q"), ("m", "n")}


def test_merge_anchorless_edited_wins_collision():
    # Same (wrong, correct) key -> edited entry overwrites current's.
    cur = [{"wrong": "x", "correct": "y", "note": "old"}]
    edited = [{"wrong": "x", "correct": "y", "note": "new"}]
    out = tc.three_way_merge_corrections([], edited, cur)
    assert len(out) == 1
    assert out[0]["note"] == "new"


def test_merge_anchorless_not_collapsed():
    # Two distinct anchorless chips must not collapse into one.
    cur = [_chip("a", "b"), _chip("c", "d")]
    out = tc.three_way_merge_corrections([], [], cur)
    assert len(out) == 2


def test_merge_ordering_anchored_before_anchorless():
    cur = [_chip("z", "free"), _chip("a", "anchored", idx=3)]
    out = tc.three_way_merge_corrections([], [], cur)
    # anchored (idx sort key (0, idx)) come before anchorless ((1, 0)).
    assert out[0]["correct"] == "anchored"
    assert out[1]["correct"] == "free"


def test_merge_anchored_sorted_by_idx():
    cur = [_chip("a", "1", idx=5), _chip("b", "2", idx=1), _chip("c", "3", idx=3)]
    out = tc.three_way_merge_corrections([], [], cur)
    assert [c["idx"] for c in out] == [1, 3, 5]
