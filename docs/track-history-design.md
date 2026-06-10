# Track History & Metadata Correction — Design

Status: **Phase 1 + 2 complete** · Updated 2026-06-10

Phase 2 built: `spotify_splitter/metadata_edit.py` (`edit_track_metadata` — mutagen
`date`/`genre` rewrite, path-validated to the output dir, year 1900–2100) +
`TrackHistoryWriter.update_metadata`; `/edit-metadata` POST endpoint; per-row ✎
inline edit in the Recorded Tracks table (saved rows only). Tests:
`tests/test_metadata_edit.py`.

Built: `spotify_splitter/track_history.py` (`TrackResult` + capped JSONL
`TrackHistoryWriter`); `SegmentManager.on_track_result` emits at all four outcome
points with resolved year/genre; `--history-file` CLI wiring; service `/history`
endpoint (+ `--history-file` spawn arg, persistent `~/.cache/.../history.jsonl`);
web UI "Recorded Tracks" table. **Step 3 done:** the log-grep activity panel
(`filter_recorder_logs`, `/logs`, `/toggle-verbose`) is removed; the table is the
sole activity view. Next: Phase 2 (inline year/genre edit via mutagen tag rewrite,
path-validated to the output dir).

## Goal

Give the web UI a structured **Recorded Tracks** history showing each track's
outcome (saved / skipped-incomplete / skipped-already-exists / failed) plus the
**year and genre** that were actually tagged — so LastFM mistakes are easy to spot.
Later, let the user **correct** the year/genre of a saved track in place.

This also retires the brittle **log-grep** activity panel (`filter_recorder_logs`):
a structured per-track record is the proper replacement the original roadmap
flagged ("replace log-scraping with a structured channel").

## Why it's a natural fit

- The **export worker** in `segmenter.py` is the single chokepoint where every
  outcome happens and already emits them (`Saved`, `IncompleteTrackSkip`,
  "already exists", export failure).
- **Year/genre are resolved at save time** inside `_export` (LastFM + the
  title/album regex fallback) right before tagging — so a saved record can carry
  the *final* values written to the file. That's exactly what makes a wrong year
  (e.g. a 1972 song tagged 2025 from a re-release) jump out in a table.
- We already use `mutagen` to write ID3, so correcting a tag is a tag rewrite —
  no re-encode.

---

## Phase 1 — Structured history (read-only)

### Record shape

One record per finished track:

```json
{
  "ts": "2026-06-09T14:30:05Z",
  "outcome": "saved",            // saved | skipped_incomplete | skipped_exists | failed
  "artist": "Lobo",
  "title": "I'd Love You to Want Me",
  "album": "Of A Simple Man",
  "track_number": 7,
  "year": 1972,                  // resolved value actually tagged (null if unknown)
  "genre": "70s; pop; oldies",   // resolved value actually tagged (null if unknown)
  "path": "/home/.../07 - I'd Love You to Want Me.mp3",  // null for non-saved
  "duration_ms": 247600,
  "reason": "captured 54s of 220s"   // for skipped/failed; optional
}
```

### Capture point

In `segmenter._export` (export worker thread), after tags are computed/written,
emit the record with the **resolved** year/genre. For skip/fail outcomes, emit
from the segment thread / export worker where they're already detected. A single
helper `record_track_result(record)` keeps it one code path.

### Persistence — decision: **persistent, capped**

Write to a capped **JSONL** history file (e.g. `~/.cache/spotify_splitter/service/history.jsonl`,
or alongside the output dir). Persistent (survives restart) so a finished session
can be reviewed; cap to the last N (e.g. 500) by trimming on write. This is more
useful than an in-memory session list for "spot mistakes."

Open question: history alongside the **status file** (service-owned cache) vs in
the **output dir** (travels with the library). Lean: service cache for the UI;
revisit if users want a portable log.

### Web UI

A **Recorded Tracks** table (new card or replacing the activity-log panel):
outcome icon, artist – title, year, genre, (relative) path. Read from a new
`/history` endpoint (reads the JSONL, newest first). This lets us delete the
log-grep `filter_recorder_logs` path once parity is confirmed.

### Acceptance

- Saved tracks appear with the year/genre actually written to the file.
- Skipped (incomplete / already-exists) and failed tracks appear with the reason.
- History survives a service restart; capped.
- Pure `record_track_result` + `/history` reader are unit-tested.

---

## Phase 2 — Inline year/genre correction

LastFM is often wrong (re-release dates, mis-tagged genres). Let the user fix a
saved track's year/genre from the history row.

- **UI**: an edit affordance per saved row (year + genre fields) → POST to a new
  `/edit-metadata` endpoint with the track `path` + new values.
- **Backend**: re-open the file with `mutagen` (`EasyID3`), set `date`/`genre`,
  save. No re-encode. Update the history record in place.
- **Safety**: validate the `path` is **under the configured output dir** before
  writing (no arbitrary file writes); reject if missing. Year validation
  (1900–2100); genre free-text.
- Optional extras: "re-fetch from LastFM" button; bulk edit; write to the M3U if
  affected.

### Acceptance

- Editing a row rewrites the file's ID3 `date`/`genre` (verified via `mutagen`),
  path-validated to the output dir, and refreshes the history entry.

---

## Migration / sequencing

1. `record_track_result` + JSONL history writer (capped) + capture in the export
   worker with resolved year/genre. Unit tests.
2. `/history` endpoint + Recorded Tracks table in the web UI.
3. Retire `filter_recorder_logs` once the table covers the activity panel's role.
4. (Phase 2) `/edit-metadata` endpoint + `mutagen` tag rewrite + per-row edit UI.

## Notes / open questions

- Schema includes `path` and explicit `year`/`genre` from the start so Phase 2
  editing has what it needs without a schema change.
- The history channel is structured status' natural extension — keep it separate
  from the live `status.json` (which is a single current snapshot) as an
  append-only log.
- Decide table-vs-augment for the activity panel (replace the grep panel, or show
  both initially).
