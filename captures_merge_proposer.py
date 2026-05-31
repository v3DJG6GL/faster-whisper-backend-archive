"""
Auto-merge proposer for /captures fine-tuning data curation.

Given the user's ungrouped captures, ranks plausible merges into ~26 s
"groups" that respect the 28 s hard cap already enforced by
captures_routes.create_sample_api (mirrors its raw-`duration_seconds`
arithmetic; see captures_routes.py:896-901).

Heuristic rationale (see captures-finetune-findings.md + research notes):
  - Whisper's encoder window is 30 s; ~25-30 s merged samples preserve
    timestamp-prediction (HF, SwissText 2024, ivrit.ai 2025). Short padded
    clips erode it.
  - Same recording session → similar acoustic environment → bias toward
    same-session grouping. SESSION_GAP_S of 10 min segments sessions but
    is also penalized softly via the density score.
  - Avoid grouping near-duplicate transcripts (echo / redictation pairs);
    duplication inside one 30 s window inflates effective batch duplicates.
  - Language must match (BCP-47 primary subtag).

Outputs are RANKED, never auto-executed. The /captures UI hands the user
the existing merge-modal preview before any write.

In-memory cache: results are cached per (user_id) with a TTL and are
invalidated explicitly by capture writes (see invalidate() callers in
captures_store + capture_samples_store).
"""
from __future__ import annotations

import difflib
import logging
import time
from typing import Any

import config as cfg
import captures_store

logger = logging.getLogger(__name__)


# The finished-sample duration cap is the global cfg.CAPTURES_SAMPLE_MAX_DURATION_S
# (read in _generate); the inter-member gap mirrors the global VAD-internal
# silence. Kept out of imports to avoid a circular dep with captures_routes.
_DEFAULT_GAP_MS = 300

# Sentinel cache key for admin "all users" view.
_ALL_USERS = "__all__"

# {cache_key: (generated_ts, [ProposedGroup, ...])}
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

# Per-capture trimmed-duration cache. Capture audio is immutable once written,
# so a capture id maps to a stable trimmed duration; the file mtime + the two
# trim-config values guard against the rare edit / config change. Kept separate
# from _CACHE (proposals) so it survives the proposal TTL and is reused across
# runs — the first run after a miss pays one VAD pass per eligible capture.
# {capture_id: (mtime, edge_pad_ms, max_gap_ms, trimmed_seconds)}
_TRIM_DUR_CACHE: dict[str, tuple[float, int, int, float]] = {}


def trimmed_duration_s(row: dict[str, Any]) -> float:
    """Trimmed audio duration (seconds) a capture contributes to a merged
    group. Mirrors what audio_merge.merge_wavs would cut per member. Falls back
    to raw `duration_seconds` when group trimming is disabled, the audio can't
    be read, or VAD is unavailable (matches audio_vad_trim's own fallback).

    Public so captures_routes (merge validation + the manual merge-estimate
    endpoint) can reuse the same cached per-capture trim as the proposer."""
    raw = float(row.get("duration_seconds") or 0.0)
    if not getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False):
        return raw
    cid = row.get("id") or ""
    relpath = row.get("audio_relpath") or ""
    if not cid or not relpath:
        return raw
    edge = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300))
    max_gap = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300))
    try:
        import os
        import audio_merge
        import audio_vad_trim
        abs_p = captures_store.abs_audio_path(relpath)
        mtime = os.path.getmtime(abs_p)
        hit = _TRIM_DUR_CACHE.get(cid)
        if hit is not None and hit[0] == mtime and hit[1] == edge and hit[2] == max_gap:
            return hit[3]
        pcm, n = audio_merge.read_pcm(abs_p)
        res = audio_vad_trim.trim_pcm_for_merge(
            pcm, n, edge_pad_ms=edge, max_internal_gap_ms=max_gap,
        )
        dur_s = float(res.get("new_duration_ms") or 0) / 1000.0 or raw
        _TRIM_DUR_CACHE[cid] = (mtime, edge, max_gap, dur_s)
        return dur_s
    except Exception as e:  # read/VAD failure → raw fallback
        logger.debug("[proposer] trim-duration fallback for %s: %s", cid, e)
        return raw


def _bcp47_primary(lang: str | None) -> str:
    if not lang:
        return ""
    return lang.strip().lower().replace("_", "-").split("-")[0]


def _normalize_text(s: str | None) -> str:
    return (s or "").strip().lower()


def _pick_text(row: dict[str, Any]) -> str:
    # Prefer post-pipeline training text → final → raw. Used for both the
    # duplicate-similarity check and the proposal-card preview.
    for k in ("text_for_training", "final", "raw"):
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _dur(m: dict[str, Any]) -> float:
    """Group-contribution duration of a member: the cached trimmed value when
    the eligible row was annotated, else raw `duration_seconds`."""
    v = m.get("_trim_dur_s")
    return float(v) if v is not None else float(m.get("duration_seconds") or 0.0)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _build_sample_score(members: list[dict[str, Any]], gap_s: float,
                        target_s: float, edge_s: float = 0.0) -> dict[str, float]:
    """Return per-component scores + composite for ranking."""
    n = len(members)
    # Packed audio uses TRIMMED bodies (what the merged WAV actually is) +
    # (n-1) join gaps + the two uniform outer margins; wall span uses RAW
    # duration since the recording occupied real wall-clock time before trim.
    total_dur = sum(_dur(m) for m in members) + gap_s * (n - 1) + 2.0 * edge_s
    first_ts = float(members[0]["created_ts"])
    last = members[-1]
    last_end = float(last["created_ts"]) + float(last["duration_seconds"])
    wall_dur = max(last_end - first_ts, total_dur)

    # Peak at target_s, decays linearly to 0 at 0 or 2*target_s.
    fill_score = max(0.0, 1.0 - abs(total_dur - target_s) / target_s)
    # Density = packed audio / wall-clock span. n=2 trivially ~1.0 so neutralize.
    density_score = 0.5 if n < 3 else max(0.0, min(1.0, total_dur / wall_dur))
    # Peak at 4 members, decays by 1/8 per step in either direction.
    member_score = max(0.0, 1.0 - abs(n - 4) / 8.0)
    reviewed_count = sum(1 for m in members if (m.get("status") or "") == "reviewed")
    reviewed_boost = 0.1 * (reviewed_count / n)

    composite = (
        0.45 * fill_score
        + 0.25 * density_score
        + 0.20 * member_score
        + reviewed_boost
    )
    return {
        "total_dur": total_dur,
        "wall_dur": wall_dur,
        "fill_score": fill_score,
        "density_score": density_score,
        "member_score": member_score,
        "reviewed_count": reviewed_count,
        "composite": composite,
    }


def _format_reason(scored: dict[str, float], n: int) -> str:
    total = scored["total_dur"]
    wall = scored["wall_dur"]
    # Wall is total when n=1 (degenerate) — only show "from a X session" when
    # the wall span is materially larger than packed audio.
    if wall > total * 1.5 and wall >= 60:
        mins = int(wall // 60)
        secs = int(wall % 60)
        span = f"{mins} min {secs} s session" if mins else f"{secs} s session"
    else:
        span = f"{wall:.0f} s session"
    return f"{n} clips from a {span}, packs to {total:.1f} s"


def _generate_candidates_for_bucket(
    bucket: list[dict[str, Any]],
    gap_s: float,
    dup_threshold: float,
    target_s: float,
    hard_cap_s: float,
    edge_s: float = 0.0,
    min_sample_s: float = 0.0,
) -> list[tuple[float, list[dict[str, Any]]]]:
    """For one (session, language) bucket of chronologically-sorted captures,
    enumerate one candidate group per starting index by walking forward and
    skipping any successor that overflows the budget or duplicates an
    already-included member. Returns [(composite_score, [members]), ...].

    Greedy with skip-allowed lookahead. O(N^2). Not globally optimal, but
    produces sensible session-bundles for typical N ≤ 30. The non-overlap
    pass in propose_merges() handles cross-candidate member contention.
    """
    n = len(bucket)
    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    texts = [_normalize_text(_pick_text(m)) for m in bucket]
    for i in range(n):
        members: list[dict[str, Any]] = [bucket[i]]
        member_texts: list[str] = [texts[i]]
        # Outer margins (both ends) count toward the cap from the start.
        used_dur = 2.0 * edge_s + _dur(bucket[i])
        for j in range(i + 1, n):
            cand = bucket[j]
            cand_dur = _dur(cand)
            # Adding j costs cand_dur + one gap (between prior member and j).
            tentative = used_dur + cand_dur + gap_s
            if tentative > hard_cap_s:
                # Doesn't fit; try smaller successors (skip-allowed pack).
                continue
            ct = texts[j]
            is_dup = any(_ratio(ct, mt) > dup_threshold for mt in member_texts)
            if is_dup:
                continue
            members.append(cand)
            member_texts.append(ct)
            used_dur = tentative
        # Single-capture samples (group-of-one) are allowed; drop a candidate
        # whose finished length is below the sample-min floor or above the cap
        # (the latter guards a lone over-long member that has no pair).
        scored = _build_sample_score(members, gap_s, target_s, edge_s)
        if scored["total_dur"] < min_sample_s or scored["total_dur"] > hard_cap_s:
            continue
        candidates.append((scored["composite"], members))
    return candidates


def _build_proposal(
    members: list[dict[str, Any]],
    gap_s: float,
    target_s: float,
    language: str,
    user_id: str,
    edge_s: float = 0.0,
) -> dict[str, Any]:
    scored = _build_sample_score(members, gap_s, target_s, edge_s)
    return {
        "member_ids": [m["id"] for m in members],
        "member_previews": [
            {
                "id": m["id"],
                "user_id": m.get("user_id") or "",
                # username is filled by the endpoint via api_keys_store.get_usernames
                "username": None,
                "created_ts": float(m["created_ts"]),
                "duration_s": _dur(m),
                "status": m.get("status") or "",
                "preview": (_pick_text(m) or "")[:80],
            }
            for m in members
        ],
        "total_duration_s": round(scored["total_dur"], 3),
        "wall_duration_s": round(scored["wall_dur"], 3),
        "language": language,
        "user_id": user_id,
        # username is filled by the endpoint (same lookup as member_previews)
        "username": None,
        "member_count": len(members),
        "fill_score": round(scored["fill_score"], 4),
        "density_score": round(scored["density_score"], 4),
        "member_score": round(scored["member_score"], 4),
        "reviewed_count": int(scored["reviewed_count"]),
        "composite_score": round(scored["composite"], 4),
        "reason": _format_reason(scored, len(members)),
    }


def _eligible(row: dict[str, Any], min_clip_s: float, hard_cap_s: float) -> bool:
    if (row.get("status") or "") in {"dismissed", "audio_missing"}:
        return False
    if row.get("sample_id"):
        return False
    # Min on RAW duration (the ingestion floor every stored capture clears);
    # cap on the TRIMMED body, so a raw-long / trims-short clip (e.g. 40 s raw,
    # 27 s of speech) is now eligible instead of wrongly rejected.
    raw = float(row.get("duration_seconds") or 0.0)
    if raw < min_clip_s:
        return False
    if trimmed_duration_s(row) > hard_cap_s:
        return False
    if not (row.get("language") or "").strip():
        return False
    if not (row.get("text_for_training") or row.get("final") or row.get("raw")):
        return False
    return True


def propose_merges(
    *,
    user_id_filter: str | None,
    is_admin: bool,
    caller_user_id: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (proposals, was_cached). `proposals` is a list of dicts ready
    for JSON serialization, sorted by composite score desc and capped at
    cfg.CAPTURES_PROPOSER_MAX_PROPOSALS.

    Non-admin callers always see their own captures only (user_id_filter
    is ignored). Admin callers may pass user_id_filter=None to see proposals
    across all users (sentinel cache key '__all__').
    """
    # Resolve effective filter + cache key.
    if not is_admin:
        effective_user_id = caller_user_id
        cache_key = caller_user_id or _ALL_USERS
    elif user_id_filter:
        effective_user_id = user_id_filter
        cache_key = user_id_filter
    else:
        effective_user_id = None
        cache_key = _ALL_USERS

    ttl = max(1, int(cfg.CAPTURES_PROPOSER_CACHE_TTL_S))
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < ttl:
        return list(cached[1]), True

    session_gap_s = max(1, int(cfg.CAPTURES_PROPOSER_SESSION_GAP_S))
    # Per-capture floor is the (raw) ingestion minimum; every stored capture
    # already clears it, so this is mostly a belt-and-braces guard.
    min_clip_s = float(cfg.CAPTURE_RECORDINGS_MIN_DURATION_SEC)
    dup_threshold = float(cfg.CAPTURES_PROPOSER_DUP_THRESHOLD)
    max_proposals = max(1, int(cfg.CAPTURES_PROPOSER_MAX_PROPOSALS))
    target_s = float(cfg.CAPTURES_PROPOSER_TARGET_S)
    hard_cap_s = float(cfg.CAPTURES_SAMPLE_MAX_DURATION_S)
    # Inter-member gap estimate mirrors the global silence knob (the merge
    # inserts this between members), so "packs to X s" matches the real WAV.
    gap_s = float(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300)) / 1000.0
    # Uniform outer margin on both ends of the merged WAV (counts toward cap).
    edge_s = float(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300)) / 1000.0
    # Finished-sample floor — drop proposals (incl. solos) shorter than this.
    min_sample_s = float(getattr(cfg, "CAPTURES_SAMPLE_MIN_DURATION_S", 1.0))

    # Pull a bounded window of recent captures. 500 keeps work bounded for
    # the rare admin "all users" view; per-user views typically have far
    # fewer ungrouped rows.
    rows = captures_store.list_captures(
        status=None,
        limit=500,
        user_id=effective_user_id,
    )
    eligible = [r for r in rows if _eligible(r, min_clip_s, hard_cap_s)]

    # Annotate each survivor with its trimmed group-contribution duration once
    # (cached across runs). All downstream packing + scoring + display reads go
    # through _dur(), so a proposal's "packs to X s" reflects the real merged
    # length and fill_score targets trimmed speech, not raw audio.
    for r in eligible:
        r["_trim_dur_s"] = trimmed_duration_s(r)

    # Sort chronological ascending so session segmentation walks forward in time.
    eligible.sort(key=lambda r: float(r["created_ts"]))

    # Session segmentation.
    sessions: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    prev_ts: float | None = None
    for r in eligible:
        ts = float(r["created_ts"])
        if prev_ts is not None and (ts - prev_ts) > session_gap_s:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(r)
        prev_ts = ts
    if cur:
        sessions.append(cur)

    # Per session × user × language → candidates. user_id partition matters
    # because create_sample_api enforces same-user (captures_routes.py:878-882);
    # without it, the admin "all users" view could emit proposals that the
    # merge endpoint rejects.
    all_candidates: list[tuple[float, list[dict[str, Any]], str, str]] = []
    for sess in sessions:
        by_keys: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in sess:
            lang = _bcp47_primary(r.get("language"))
            uid = r.get("user_id") or ""
            if not lang or not uid:
                continue
            by_keys.setdefault((uid, lang), []).append(r)
        for (uid, lang), bucket in by_keys.items():
            for score, members in _generate_candidates_for_bucket(
                bucket, gap_s, dup_threshold, target_s, hard_cap_s, edge_s,
                min_sample_s,
            ):
                all_candidates.append((score, members, lang, uid))

    # Rank, then take non-overlapping greedy.
    all_candidates.sort(key=lambda c: c[0], reverse=True)
    claimed: set[str] = set()
    proposals: list[dict[str, Any]] = []
    for _, members, lang, uid in all_candidates:
        if any(m["id"] in claimed for m in members):
            continue
        for m in members:
            claimed.add(m["id"])
        proposals.append(_build_proposal(members, gap_s, target_s, lang, uid, edge_s))
        if len(proposals) >= max_proposals:
            break

    _CACHE[cache_key] = (now, proposals)
    logger.info(
        "[proposer] user=%s n_eligible=%d sessions=%d candidates=%d proposals=%d",
        cache_key, len(eligible), len(sessions), len(all_candidates), len(proposals),
    )
    return list(proposals), False


def invalidate(user_id: str | None) -> None:
    """Drop cached proposals for a user (and the all-users entry, which any
    write may affect). Called from captures_store + capture_samples_store
    write paths. Safe to call with no current cache entry."""
    if user_id:
        _CACHE.pop(user_id, None)
    _CACHE.pop(_ALL_USERS, None)
