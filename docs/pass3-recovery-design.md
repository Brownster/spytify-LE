# Pass 3 Recovery Simplification

## Goal

Collapse the SegmentManager recovery maze into one small, explicit policy:

- Build and queue a segment once.
- Advance capture state once the export job owns a copied audio slice.
- Retry only transient export I/O on the export worker.
- Log and notify on persistent failure, then keep recording.

The recorder must prefer uninterrupted capture over repeated in-line recovery work. Once Pass 2 moved export to a worker, segment-level retries stopped being useful: the segment thread no longer waits for ffmpeg, LastFM, artwork, or playlist writes.

## Current Problems

`segmenter.py` still carries recovery paths from the pre-worker design:

- `_process_segment_with_recovery`
- `_attempt_segment_recovery`
- `_attempt_graceful_degradation`
- `_attempt_export_recovery`
- `_attempt_export_degradation`
- `_export_with_minimal_processing`
- `_handle_processing_failure`

These paths add complexity, but several no longer match the data flow:

- Segment retries can re-run boundary detection after state may already be advanced.
- Degraded segment export still slices from the legacy `continuous_buffer`, not the frame ledger.
- Export recovery contains dead logic such as computing an alternate path without using it.
- Worker export failures cannot safely ask the segment thread to re-cut audio because the ledger may have discarded the source frames.

## Target Policy

### Segment Thread

The segment thread owns only capture-adjacent work:

1. Materialize the ledger window.
2. Run boundary detection.
3. Validate duration.
4. Apply final boundary correction.
5. Queue `ExportJob(audio_copy, track_info)`.
6. Advance markers and discard old ledger frames.

If boundary detection fails, keep the existing fallback slice between markers. If duration validation shows an incomplete track, keep the existing `IncompleteTrackSkip` behavior: advance past it and do not report an error.

If segment preparation raises an unexpected exception before an export job is queued, log `processing_error`, notify the UI, advance past the segment with `_advance_after_segment` when possible, and continue. Do not retry segment processing in-place.

### Export Worker

The export worker owns slow and failure-prone work:

- ffmpeg export
- ID3 tagging
- LastFM lookup
- artwork download
- playlist writes

The worker retries only transient I/O. A small classifier should return true for errors likely to succeed on retry:

- `OSError`
- `IOError`
- `TimeoutError`
- `requests.RequestException`

For these, retry `_export()` up to three attempts with a short bounded backoff. For other exceptions, fail immediately.

Persistent failure must:

1. Increment `export_errors`.
2. Log the path, track, attempt count, and exception.
3. Emit `ui_callback("export_error", ...)`.
4. Emit `ui_callback("processing_failure", ...)` after the final failed attempt.
5. Leave capture state untouched because the segment thread already advanced safely.

Tagging, LastFM, and artwork failures inside `_export()` should remain non-fatal where they already are. A failed metadata lookup must not fail the whole export.

## Removed Behavior

Pass 3 should delete the old degradation and speculative recovery paths:

- Segment-level retry loop.
- Segment recovery by trimming or mutating marker positions.
- Minimal degraded export.
- Export degradation.
- Alternate-path recovery that computes a path but never exports to it.
- Post-failure "file exists despite errors" reconciliation in the segment thread.

This deliberately removes the idea that a failed export can be recovered by re-processing the segment. After export is decoupled, that is no longer a safe invariant.

## Statistics And UI

Keep the public statistics keys for compatibility:

- `processing_errors`
- `export_errors`
- `recovery_attempts`
- `successful_recoveries`
- `degraded_exports`
- `current_processing_track`
- `processing_retry_count`
- `last_successful_export`

After simplification:

- `recovery_attempts` means export retry attempts.
- `successful_recoveries` means export succeeded after at least one retry.
- `degraded_exports` should remain present but stay at zero unless a future real degraded export path is added.
- `processing_retry_count` should remain present but stay at zero.

The existing `_stats_lock` remains the single guard for these counters.

## Shutdown Ordering

Do not change the Pass 2 shutdown contract:

1. `shutdown_cleanup()` ingests queued audio.
2. It processes pending marker pairs.
3. It waits for export jobs.
4. It stops the export worker.
5. Engine finalization runs the external tagger after exports finish.

This order prevents the tagger and playlist close from outrunning the final export.

## Migration Steps

1. Add `_is_transient_export_error(error)` and a small retry helper around `_export()`.
2. Replace `_export_with_error_handling()` with the new worker-owned policy.
3. Replace `_process_segment_with_recovery()` with a single `_process_segment_once()` path or inline it into `process_segments()`.
4. Remove the degradation and speculative recovery helpers.
5. Preserve `IncompleteTrackSkip` exactly.
6. Update tests that assert recovery/degradation behavior to assert the new policy instead.

## Test Plan

Add or update tests for:

- A transient export error retries and then succeeds.
- A non-transient export error fails once and emits `export_error` plus `processing_failure`.
- Segment preparation failure logs/notifies once, advances markers, and does not retry.
- Incomplete tracks remain skips, not failures.
- Shutdown still drains exports before post-run finalization.
- Public error-statistics keys remain present.

Run:

```bash
.venv/bin/pytest tests/test_segmenter.py tests/test_error_recovery_integration.py
.venv/bin/pytest -m "not slow"
```

## Non-Goals

- Do not rewrite `ErrorRecoveryManager` in this slice.
- Do not merge `AudioStream` and `EnhancedAudioStream` in this slice.
- Do not split `segmenter.py` yet.
- Do not add a new degraded export format unless a real user need appears.
