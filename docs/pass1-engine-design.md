# Pass 1 Engine Design

Status: draft for implementation - Date: 2026-06-07

## Goal

Extract a `RecorderEngine` seam without changing the recording pipeline internals.
This pass moves orchestration out of `spotify_splitter.main` so the CLI and web
service can share one lifecycle/control surface before Pass 2 changes the hot path.

The extraction must keep `audio.py`, `segmenter.py`, and tagging/export behaviour
functionally unchanged.

## Non-goals

- Do not replace `SegmentManager`'s `AudioSegment` buffer yet.
- Do not split export onto a worker queue yet.
- Do not rewrite `EnhancedAudioStream`'s callback yet.
- Do not make web pause/resume cooperative yet; keep signal-based pause until the
  engine owns a real paused state.
- Do not move Rich `Live` rendering into the engine.

## Public Surface

Create `spotify_splitter/engine.py` with a small control API:

```python
class RecorderEngine:
    def start(self) -> None: ...
    def wait(self) -> None: ...
    def run(self) -> None: ...
    def stop(self, flush: bool = True) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def status(self) -> RecorderStatus: ...
    def handle_command(self, command: dict) -> bool: ...
```

Initial semantics:

- `start()` creates queues, stream, `SegmentManager`, and worker threads, launches
  them, and returns. It is always non-blocking.
- `wait()` blocks until the engine stops.
- `run()` is a convenience wrapper for subprocess/headless use: `start()` then `wait()`.
- `stop(flush=True)` routes through the single guarded cleanup path. With `flush=True`,
  it runs `SegmentManager.shutdown_cleanup()`, sends `("shutdown", None)`, joins the
  processing thread, and flushes cache.
- `pause()`/`resume()` update engine state only at first. Web pause/resume remain
  signal-based until cooperative pause is implemented.
- `status()` returns a snapshot of `RecorderStatus`.
- `handle_command()` accepts the current stdin NDJSON contract. It returns `True` when
  the command asks the command reader to exit, currently only `{"cmd": "stop"}`.

## State Split

The engine owns recorder facts:

- current track
- tracks recorded
- timer enabled/elapsed/remaining
- queue depth
- dropped frames, initially still `0` with the existing Pass 2 TODO
- buffer warnings
- lifecycle state: `starting`, `waiting`, `recording`, `processing`, `paused`,
  `stopping`, `stopped`, `error`
- last error

The CLI owns display strings and Rich rendering:

- panel titles
- emoji labels
- formatted timer text
- status wording such as "First track - will be discarded"
- color choices

The bridge between them is an observer callback:

```python
EngineEvent = Callable[[str, object], None]
```

The engine emits semantic events such as `track_change`, `playback_status`,
`processing`, `saved`, `buffer_warning`, and `error`. The CLI maps those to its Rich UI.

## Loop And Render Model

The engine is non-blocking. It owns recorder lifecycle, timer expiry, status heartbeat,
and shutdown decisions, but it does not own display cadence. `start()` launches engine
threads and returns; `wait()` and `run()` are the only blocking calls.

The engine emits events when recorder facts change, including a periodic `tick` event
for status/timer refresh. Timer expiry is handled inside the engine and results in the
same guarded `stop(flush=True)` path as stdin stop and KeyboardInterrupt.

The CLI owns Rich `Live` cadence and rendering. The interactive CLI should call
`engine.start()`, run its `Live` loop while polling `engine.status()` or reacting to
engine events, then call `engine.wait()` when stopping. The subprocess entrypoint can
use `engine.run()` when no interactive render loop is needed.

This keeps display timing out of the engine while avoiding duplicated lifecycle,
heartbeat, and timer logic in each frontend.

## Errors And Exit Codes

The engine must not raise `typer.Exit`. It raises plain domain exceptions instead:

```python
class RecorderError(Exception): ...
class StreamNotFoundError(RecorderError): ...
class RecorderConfigError(RecorderError): ...
class RecorderDbusError(RecorderError): ...
```

`main.record()` maps those exceptions to user-facing log messages and exit codes. The
current service semantics must be preserved: early exit with code `1` still means
"Spotify client not ready or no playback detected" to the supervisor when runtime is
less than 8 seconds.

Bad CLI-only input such as an invalid `--max-duration` can remain validated in
`main.record()` before constructing the engine, but any engine-level config validation
should use `RecorderConfigError`.

## Thread Ownership

`RecorderEngine` owns and joins:

- segment processing thread
- MPRIS tracking thread
- optional buffer health monitor thread
- optional stdin control thread
- metrics/dashboard/optimizer lifecycle while they still exist

The PortAudio callback remains owned by `AudioStream`/`EnhancedAudioStream`; the engine
only owns the stream context and queue.

Shutdown is engine-internal:

- one `cleanup_lock`
- one `cleanup_done` guard
- all stop paths call `RecorderEngine.stop(flush=True)`
- timer expiry, stdin stop, normal loop exit, and final cleanup share the same method
- the CLI wrapper catches `KeyboardInterrupt` around `engine.start()`/`engine.wait()`
  or `engine.run()` and calls `engine.stop(flush=True)`

Engine shutdown also owns the non-thread cleanup that currently lives in
`main.record()`:

- stop metrics collection
- stop dashboard/optimizer components
- close the playlist via `SegmentManager.close_playlist()`
- call `tag_output(...)` after recording stops

The CLI remains responsible for presenting final messages and mapping exceptions to
exit codes.

## Status File

The engine owns `RecorderStatus` mutation and writing:

- write on state changes
- write on errors
- write on timer second changes
- write heartbeat every 5 seconds
- metric-only writes remain coalesced

The service stale threshold remains 15 seconds, a 3x heartbeat window.

## Migration Steps

1. Add `RecorderEngineConfig` dataclass containing the resolved options currently local
   to `record()`.
2. Move queue creation, adaptive component setup, `SegmentManager` construction, and
   thread startup into `RecorderEngine`.
3. Move `publish_status()`, stdin command handling, heartbeat, and guarded cleanup into
   `RecorderEngine`.
4. Move timer expiry handling into `RecorderEngine`; keep timer display formatting in
   the CLI.
5. Move metrics/dashboard/optimizer shutdown, playlist close, and post-run
   `tag_output(...)` into engine shutdown.
6. Keep `main.record()` responsible for Typer parsing, config/profile resolution, Rich
   UI construction, KeyboardInterrupt handling, exit-code mapping, and translating
   engine events to display updates.
7. Preserve existing service subprocess behaviour. It should keep using
   `--status-file` and `--control-stdin`.

## Acceptance

- `.venv/bin/pytest -m "not slow"` remains green.
- CLI `record` options behave the same.
- Web start/stop/status continue using the structured status/control channels.
- No changes to audio slicing/export semantics are introduced in this pass.
