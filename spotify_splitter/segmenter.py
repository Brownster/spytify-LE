from __future__ import annotations

from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass
import logging
import queue
from typing import List, Optional, Iterable

import numpy as np
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC, ID3
import requests

from .mpris import TrackInfo
from .track_boundary_detector import TrackBoundaryDetector, BoundaryResult, TrackMarker as EnhancedTrackMarker
from .error_recovery import ErrorRecoveryManager, RecoveryAction, ErrorSeverity
from .lastfm_api import get_lastfm_client


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "Music"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return "".join(c for c in name if c not in r'/\\:*?"<>|')


def is_song(track: TrackInfo) -> bool:
    """Return ``True`` if ``track`` looks like a real Spotify song."""
    if not track or not track.id:
        return False
    track_id_str = str(track.id)
    return track_id_str.startswith("spotify:track:") or track_id_str.startswith(
        "/com/spotify/track/"
    )


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

TrackMarker = namedtuple("TrackMarker", "timestamp track_info frame")
TrackMarker.__new__.__defaults__ = (None,)


@dataclass
class AudioChunk:
    """A retained block of int16 PCM samples with an absolute frame offset."""

    start_frame: int
    samples: np.ndarray


@dataclass
class SegmentWindow:
    """Materialized audio window with absolute-frame addressing metadata."""

    audio: AudioSegment
    start_frame: int
    end_frame: int
    from_ledger: bool


class ChunkLedger:
    """Frame-addressed raw audio ledger for the capture hot path."""

    def __init__(self, samplerate: int = 44100, channels: int = 2) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.base_frame = 0
        self.total_frames = 0
        self.chunks: List[AudioChunk] = []

    @property
    def retained_frames(self) -> int:
        """Return the number of frames still retained in the ledger."""
        return self.total_frames - self.base_frame

    def append_float32(self, frames: np.ndarray) -> None:
        """Clip float32 samples, convert to int16, and append them as one chunk."""
        if frames.size == 0:
            return
        if frames.ndim != 2:
            raise ValueError("frames must have shape (frame_count, channels)")
        if frames.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {frames.shape[1]}"
            )

        clipped = np.clip(frames, -1.0, 1.0)
        samples = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        self.chunks.append(AudioChunk(self.total_frames, samples))
        self.total_frames += samples.shape[0]

    def slice_frames(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Return a copy of samples in the absolute frame range."""
        self._validate_range(start_frame, end_frame)
        if start_frame == end_frame:
            return np.empty((0, self.channels), dtype=np.int16)

        parts = []
        for chunk in self.chunks:
            chunk_start = chunk.start_frame
            chunk_end = chunk_start + chunk.samples.shape[0]
            overlap_start = max(start_frame, chunk_start)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start >= overlap_end:
                continue
            rel_start = overlap_start - chunk_start
            rel_end = overlap_end - chunk_start
            parts.append(chunk.samples[rel_start:rel_end])

        if not parts:
            return np.empty((0, self.channels), dtype=np.int16)
        return np.concatenate(parts, axis=0).copy()

    def to_audio_segment(self, start_frame: int, end_frame: int) -> AudioSegment:
        """Materialize a frame range as an AudioSegment exactly once."""
        samples = self.slice_frames(start_frame, end_frame)
        return AudioSegment(
            samples.tobytes(),
            frame_rate=self.samplerate,
            sample_width=2,
            channels=self.channels,
        )

    def discard_before(self, frame: int) -> None:
        """Discard retained samples before an absolute frame offset."""
        if frame < self.base_frame:
            raise ValueError("cannot discard before base_frame")
        if frame > self.total_frames:
            raise ValueError("cannot discard beyond total_frames")

        retained = []
        for chunk in self.chunks:
            chunk_start = chunk.start_frame
            chunk_end = chunk_start + chunk.samples.shape[0]
            if chunk_end <= frame:
                continue
            if chunk_start < frame:
                trimmed = chunk.samples[frame - chunk_start :].copy()
                retained.append(AudioChunk(frame, trimmed))
            else:
                retained.append(chunk)

        self.chunks = retained
        self.base_frame = frame

    def _validate_range(self, start_frame: int, end_frame: int) -> None:
        if start_frame < self.base_frame:
            raise ValueError("start_frame is before base_frame")
        if end_frame > self.total_frames:
            raise ValueError("end_frame is beyond total_frames")
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")


class SegmentManager:
    """Processes a continuous audio stream to find and export complete tracks with enhanced boundary handling and error recovery."""

    def __init__(
        self,
        samplerate: int = 44100,
        output_dir: Path = OUTPUT_DIR,
        fmt: str = "mp3",
        audio_queue: Optional[queue.Queue] = None,
        event_queue: Optional[queue.Queue] = None,
        playlist_path: Optional[Path] = None,
        bundle_playlist: bool = False,
        bundle_album_art_uri: Optional[str] = None,
        playlist_base_path: Optional[str] = None,
        ui_callback: Optional[callable] = None,
        grace_period_ms: int = 500,
        max_correction_ms: int = 2000,
        error_recovery: Optional[ErrorRecoveryManager] = None,
        enable_error_recovery: bool = True,
        max_processing_retries: int = 3,
        enable_graceful_degradation: bool = True,
        lastfm_api_key: Optional[str] = None,
        allow_overwrite: bool = False,
    ) -> None:
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.format = fmt
        self.audio_queue = audio_queue
        self.event_queue = event_queue
        self.ui_callback = ui_callback

        # Handle playlist path - if just a filename, place in output directory
        if playlist_path:
            playlist_path = Path(playlist_path)
            if not playlist_path.is_absolute() and playlist_path.parent == Path('.'):
                # Just a filename, place in output directory root
                self.playlist_path = output_dir / playlist_path
            else:
                self.playlist_path = playlist_path
        else:
            self.playlist_path = None

        self.bundle_playlist = bundle_playlist
        self.playlist_base_path = playlist_base_path or str(output_dir)
        self.lastfm_api_key = lastfm_api_key
        self.allow_overwrite = allow_overwrite
        if self.bundle_playlist and not self.playlist_path:
            raise ValueError("bundle_playlist requires playlist_path")
        self.bundle_album_name = (
            self.playlist_path.stem if self.bundle_playlist else None
        )
        self.bundle_track_number = 1
        self.bundle_album_art_uri = bundle_album_art_uri  # Custom artwork URI for bundle playlists
        self.bundle_album_art = None  # Store unified album art for bundle playlists
        self.artwork_session = requests.Session()
        self.playlist_file = None
        self.playlist_tracks = set()  # Track paths already in playlist to prevent duplicates
        if self.playlist_path:
            self.playlist_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.playlist_path.exists() else "w"
            self.playlist_file = self.playlist_path.open(mode, encoding="utf-8")
            if mode == "w":
                self.playlist_file.write("#EXTM3U\n")
            else:
                # Read existing playlist entries to avoid duplicates
                try:
                    with self.playlist_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                self.playlist_tracks.add(line)
                    logger.info("Loaded %d existing tracks from playlist", len(self.playlist_tracks))
                except Exception as e:
                    logger.warning("Failed to read existing playlist entries: %s", e)

        self.continuous_buffer = AudioSegment.empty()
        self.chunk_ledger = ChunkLedger(samplerate=samplerate, channels=2)
        self.continuous_buffer_start_frame = 0
        self.track_markers: List[TrackMarker] = []
        self.first_track_seen = False
        
        # Initialize TrackBoundaryDetector for enhanced boundary handling
        self.boundary_detector = TrackBoundaryDetector(
            grace_period_ms=grace_period_ms,
            max_correction_ms=max_correction_ms
        )
        
        # Initialize error recovery system
        self.error_recovery = error_recovery or ErrorRecoveryManager()
        self.enable_error_recovery = enable_error_recovery
        self.max_processing_retries = max_processing_retries
        self.enable_graceful_degradation = enable_graceful_degradation
        
        # Error tracking and statistics
        self.processing_errors = 0
        self.export_errors = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.degraded_exports = 0
        
        # Processing state for error recovery
        self.current_processing_track = None
        self.processing_retry_count = 0
        self.last_successful_export = None
        
        logger.info(
            "SegmentManager initialized with error recovery: enabled=%s, max_retries=%d, graceful_degradation=%s",
            enable_error_recovery, max_processing_retries, enable_graceful_degradation
        )

        # Log LastFM configuration status
        if self.lastfm_api_key:
            logger.info("LastFM API configured - will fetch year and genre metadata")
        else:
            logger.warning("LastFM API key not configured - year and genre tags will not be fetched")

    def _clear_continuous_buffer(self) -> None:
        """Clear the legacy pydub buffer and record its absolute frame origin."""
        self.continuous_buffer = AudioSegment.empty()
        self.continuous_buffer_start_frame = self.chunk_ledger.total_frames
        self._discard_ledger_before_buffer_start()

    def _drop_continuous_buffer_before(self, milliseconds: int) -> None:
        """Drop legacy buffer audio before a millisecond offset and advance its origin."""
        self.continuous_buffer = self.continuous_buffer[milliseconds:]
        self.continuous_buffer_start_frame += self._ms_to_frames(milliseconds)
        self._discard_ledger_before_buffer_start()

    def _drop_audio_before_frame(self, frame: int) -> int:
        """Drop retained audio before an absolute frame and return dropped ms."""
        discard_frame = min(
            max(frame, self.chunk_ledger.base_frame),
            self.chunk_ledger.total_frames,
        )
        dropped_frames = max(0, discard_frame - self._buffer_start_frame())
        dropped_ms = self._frames_to_ms(dropped_frames)

        if len(self.continuous_buffer):
            self.continuous_buffer = self.continuous_buffer[dropped_ms:]
        self.continuous_buffer_start_frame = discard_frame
        self.chunk_ledger.discard_before(discard_frame)
        return dropped_ms

    def _discard_ledger_before_buffer_start(self) -> None:
        """Keep the transition ledger bounded to the legacy buffer window."""
        discard_frame = min(
            self.continuous_buffer_start_frame,
            self.chunk_ledger.total_frames,
        )
        self.chunk_ledger.discard_before(discard_frame)
        
    def flush_cache(self) -> None:
        """Clear all cached data for clean startup."""
        logger.debug("Flushing SegmentManager cache...")
        self.chunk_ledger = ChunkLedger(
            samplerate=self.samplerate,
            channels=self.chunk_ledger.channels,
        )
        self._clear_continuous_buffer()
        self.track_markers.clear()
        self.first_track_seen = False
        
        # Clear audio queue if present
        if self.audio_queue:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Clear event queue if present  
        if self.event_queue:
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except queue.Empty:
                    break
        logger.debug("Cache flush complete")

    def close_playlist(self) -> None:
        """Close post-run resources if open."""
        if self.playlist_file:
            try:
                self.playlist_file.close()
            except Exception:
                pass
            self.playlist_file = None
        try:
            self.artwork_session.close()
        except Exception:
            pass
        
    def shutdown_cleanup(self) -> None:
        """Clean shutdown - process any remaining tracks."""
        logger.info("Starting shutdown cleanup...")
        
        # Process any remaining audio in queue
        self._ingest_audio()
        
        # Process any pending segments
        while len(self.track_markers) >= 2:
            self.process_segments()
            
        logger.info("Shutdown cleanup complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main processing loop; consumes queues until shutdown."""
        if self.audio_queue is None or self.event_queue is None:
            raise RuntimeError("SegmentManager requires audio and event queues")

        while True:
            try:
                event_type, data = self.event_queue.get(timeout=1.0)
                if event_type == "shutdown":
                    break
                if event_type == "track_change":
                    self._ingest_audio()
                    
                    if not self.first_track_seen:
                        logger.debug("First track seen: %s. Starting recording from next track.", data.title)
                        self.first_track_seen = True
                        # Clear buffer and start fresh
                        self._clear_continuous_buffer()
                        marker = TrackMarker(0, data, self.chunk_ledger.total_frames)
                        self.track_markers = [marker]
                    else:
                        marker_timestamp = self._current_buffer_ms()
                        marker = TrackMarker(
                            marker_timestamp,
                            data,
                            self._buffer_ms_to_frame(marker_timestamp),
                        )
                        self.track_markers.append(marker)
                        logger.debug("Marker added for track: %s", data.title)
                        self.process_segments()
            except queue.Empty:
                self._ingest_audio()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest_audio(self) -> None:
        """Pull all frames from ``audio_queue`` into the frame ledger."""
        if not self.audio_queue or self.audio_queue.empty():
            return

        while not self.audio_queue.empty():
            frames = self.audio_queue.get_nowait()
            self._append_frames_to_ledger(frames)

    def _append_frames_to_ledger(self, frames: np.ndarray) -> None:
        """Append audio frames to the transition ledger on the segment thread."""
        if frames.size == 0:
            return
        if frames.ndim == 2 and frames.shape[1] != self.chunk_ledger.channels:
            if self.chunk_ledger.retained_frames == 0:
                self.chunk_ledger = ChunkLedger(
                    samplerate=self.samplerate,
                    channels=frames.shape[1],
                )
            else:
                logger.warning(
                    "Skipping chunk ledger mirror for channel mismatch: expected %d, got %d",
                    self.chunk_ledger.channels,
                    frames.shape[1],
                )
                return
        self.chunk_ledger.append_float32(frames)

    def _frames_to_ms(self, frames: int) -> int:
        """Convert frame count to pydub millisecond units."""
        return round((frames / self.samplerate) * 1000)

    def _ms_to_frames(self, milliseconds: int) -> int:
        """Convert pydub millisecond units to frame count."""
        return round((milliseconds / 1000) * self.samplerate)

    def _buffer_start_frame(self) -> int:
        """Return the absolute frame that corresponds to continuous_buffer[0]."""
        return self.continuous_buffer_start_frame

    def _buffer_ms_to_frame(self, milliseconds: int) -> int:
        """Convert a continuous_buffer-relative millisecond offset to an absolute frame."""
        return self._buffer_start_frame() + self._ms_to_frames(milliseconds)

    def _current_buffer_ms(self) -> int:
        """Return retained audio duration in ms for marker compatibility."""
        if self.chunk_ledger.retained_frames:
            return self._frames_to_ms(self.chunk_ledger.retained_frames)
        return len(self.continuous_buffer)

    def _boundary_context_frames(self) -> int:
        """Return pre/post-roll frames needed by the millisecond boundary detector."""
        context_ms = (
            self.boundary_detector.grace_period_ms
            + self.boundary_detector.max_correction_ms
            + self.boundary_detector.continuity_validator.window_size_ms
        )
        return self._ms_to_frames(context_ms)

    def _marker_frame(self, marker: TrackMarker) -> int:
        """Return a marker's absolute frame, deriving it from ms for legacy callers."""
        if marker.frame is not None:
            return marker.frame
        return self._buffer_ms_to_frame(marker.timestamp)

    def _marker_ms(self, marker: TrackMarker) -> int:
        """Return a marker offset relative to continuous_buffer in milliseconds."""
        return self._frames_to_ms(self._marker_frame(marker) - self._buffer_start_frame())

    def _rebase_marker_after_drop(self, marker: TrackMarker, dropped_ms: int) -> TrackMarker:
        """Rebase a legacy marker timestamp while preserving absolute frame position."""
        return marker._replace(timestamp=max(0, marker.timestamp - dropped_ms))

    def _segment_window(self, start_marker: TrackMarker, end_marker: TrackMarker) -> SegmentWindow:
        """Materialize the smallest detector window available for a complete segment."""
        start_frame = self._marker_frame(start_marker)
        end_frame = self._marker_frame(end_marker)

        if (
            self.chunk_ledger.retained_frames
            and start_frame >= self.chunk_ledger.base_frame
            and end_frame <= self.chunk_ledger.total_frames
        ):
            context_frames = self._boundary_context_frames()
            window_start = max(self.chunk_ledger.base_frame, start_frame - context_frames)
            window_end = min(self.chunk_ledger.total_frames, end_frame + context_frames)
            return SegmentWindow(
                audio=self.chunk_ledger.to_audio_segment(window_start, window_end),
                start_frame=window_start,
                end_frame=window_end,
                from_ledger=True,
            )

        return SegmentWindow(
            audio=self.continuous_buffer,
            start_frame=self._buffer_start_frame(),
            end_frame=self._buffer_ms_to_frame(len(self.continuous_buffer)),
            from_ledger=False,
        )

    def _window_marker_ms(self, marker: TrackMarker, window: SegmentWindow) -> int:
        """Return marker offset in ms relative to a materialized segment window."""
        return self._frames_to_ms(self._marker_frame(marker) - window.start_frame)

    def _audio_from_window_ms(
        self,
        window: SegmentWindow,
        start_ms: int,
        end_ms: int,
    ) -> AudioSegment:
        """Slice a window-relative ms range from the ledger or legacy buffer."""
        if not window.from_ledger:
            return window.audio[start_ms:end_ms]

        start_frame = window.start_frame + self._ms_to_frames(start_ms)
        end_frame = window.start_frame + self._ms_to_frames(end_ms)
        start_frame = max(window.start_frame, min(start_frame, window.end_frame))
        end_frame = max(start_frame, min(end_frame, window.end_frame))
        return self.chunk_ledger.to_audio_segment(start_frame, end_frame)

    def process_segments(self) -> None:
        """Process the first complete segment using enhanced boundary detection with grace period and continuity validation."""
        if len(self.track_markers) < 2:
            return

        start_marker = self.track_markers[0]
        end_marker = self.track_markers[1]
        end_ms = self._marker_ms(end_marker)

        if not is_song(start_marker.track_info):
            logger.debug("Previous item was not a song, skipping segment.")
            if end_marker.frame is not None:
                retain_frame = max(
                    self.chunk_ledger.base_frame,
                    self._marker_frame(end_marker) - self._boundary_context_frames(),
                )
                dropped_ms = self._drop_audio_before_frame(retain_frame)
                self.track_markers = [
                    self._rebase_marker_after_drop(marker, dropped_ms)
                    for marker in self.track_markers[1:]
                ]
            else:
                self._drop_continuous_buffer_before(end_ms)
                self.track_markers.pop(0)
            return

        # Set current processing track for error recovery
        self.current_processing_track = start_marker.track_info
        self.processing_retry_count = 0

        logger.info("Processing segment for: %s", start_marker.track_info.title)
        if self.ui_callback:
            self.ui_callback("processing", start_marker.track_info)

        # Process with error recovery
        if self.enable_error_recovery:
            success = self._process_segment_with_recovery(start_marker, end_marker)
            if not success:
                # Don't log error here - let _handle_processing_failure check if file exists first
                self._handle_processing_failure(start_marker, end_marker)
        else:
            # Process without error recovery (legacy mode)
            self._process_segment_internal(start_marker, end_marker)

    def _process_segment_with_recovery(self, start_marker: TrackMarker, end_marker: TrackMarker) -> bool:
        """
        Process a segment with error recovery capabilities.
        
        Args:
            start_marker: Start marker for the segment
            end_marker: End marker for the segment
            
        Returns:
            True if processing was successful, False otherwise
        """
        for attempt in range(self.max_processing_retries + 1):
            try:
                self.processing_retry_count = attempt
                success = self._process_segment_internal(start_marker, end_marker)
                
                if success:
                    if attempt > 0:
                        self.successful_recoveries += 1
                        logger.info("Segment processing recovered after %d attempts: %s", 
                                  attempt, start_marker.track_info.title)
                        
                        # Notify UI of successful recovery
                        if self.ui_callback:
                            self.ui_callback("recovery_success", {
                                "track": start_marker.track_info,
                                "attempts": attempt + 1
                            })
                    
                    self.last_successful_export = start_marker.track_info
                    return True
                else:
                    # Processing returned False but didn't raise exception
                    if attempt < self.max_processing_retries:
                        logger.warning("Segment processing failed (attempt %d/%d), retrying: %s", 
                                     attempt + 1, self.max_processing_retries + 1, start_marker.track_info.title)
                        continue
                    else:
                        logger.error("Segment processing failed after all attempts: %s", start_marker.track_info.title)
                        return False
                        
            except Exception as e:
                self.processing_errors += 1
                self.recovery_attempts += 1
                
                # Handle the error through error recovery manager
                context = f"segment_processing_{start_marker.track_info.title}"
                recovery_action = self.error_recovery.handle_error(e, context)
                
                logger.warning("Processing error (attempt %d/%d) for %s: %s -> %s", 
                             attempt + 1, self.max_processing_retries + 1, 
                             start_marker.track_info.title, type(e).__name__, recovery_action.value)
                
                # Notify UI of processing error
                if self.ui_callback:
                    self.ui_callback("processing_error", {
                        "track": start_marker.track_info,
                        "error": str(e),
                        "attempt": attempt + 1,
                        "recovery_action": recovery_action.value
                    })
                
                # Handle different recovery actions
                if recovery_action == RecoveryAction.RETRY and attempt < self.max_processing_retries:
                    # Attempt recovery
                    recovery_success = self._attempt_segment_recovery(e, start_marker, end_marker)
                    if recovery_success:
                        logger.info("Segment recovery successful, retrying processing")
                        continue
                    else:
                        logger.warning("Segment recovery failed, continuing with retry")
                        continue
                        
                elif recovery_action == RecoveryAction.GRACEFUL_DEGRADE:
                    # Attempt graceful degradation
                    if self.enable_graceful_degradation:
                        degraded_success = self._attempt_graceful_degradation(start_marker, end_marker, e)
                        if degraded_success:
                            self.degraded_exports += 1
                            logger.info("Graceful degradation successful for: %s", start_marker.track_info.title)
                            return True
                        else:
                            logger.warning("Graceful degradation failed for: %s", start_marker.track_info.title)
                    
                    # Continue to next attempt if degradation failed
                    if attempt < self.max_processing_retries:
                        continue
                    else:
                        return False
                        
                elif recovery_action == RecoveryAction.ESCALATE:
                    # Escalate immediately - don't retry
                    self.error_recovery.escalate_error(e, context)
                    logger.critical("Error escalated for segment processing: %s", start_marker.track_info.title)
                    return False
                    
                else:
                    # For other recovery actions or final attempt, continue or fail
                    if attempt < self.max_processing_retries:
                        continue
                    else:
                        logger.error("All recovery attempts exhausted for: %s", start_marker.track_info.title)
                        return False
        
        return False

    def _process_segment_internal(self, start_marker: TrackMarker, end_marker: TrackMarker) -> bool:
        """
        Internal segment processing logic with enhanced boundary detection.
        
        Args:
            start_marker: Start marker for the segment
            end_marker: End marker for the segment
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Convert to enhanced track markers for boundary detection
            window = self._segment_window(start_marker, end_marker)
            start_ms = self._window_marker_ms(start_marker, window)
            end_ms = self._window_marker_ms(end_marker, window)
            enhanced_markers = [
                EnhancedTrackMarker(
                    timestamp=start_ms,
                    track_info=start_marker.track_info,
                    confidence=1.0
                ),
                EnhancedTrackMarker(
                    timestamp=end_ms,
                    track_info=end_marker.track_info,
                    confidence=1.0
                )
            ]

            # Use TrackBoundaryDetector for enhanced boundary detection
            boundary_result = self.boundary_detector.detect_boundary(
                window.audio, enhanced_markers
            )

            if not boundary_result:
                logger.warning("Boundary detection failed for track: %s", start_marker.track_info.title)
                # Fallback to original timestamp-based cutting
                song_chunk = self._audio_from_window_ms(window, start_ms, end_ms)
                cleanup_ms = end_ms
            else:
                # Use enhanced boundary detection results
                logger.debug(
                    "Boundary detection results - Start: %d, End: %d, Grace period: %s, Continuity: %s, Correction: %s",
                    boundary_result.start_frame, boundary_result.end_frame,
                    boundary_result.grace_period_applied, boundary_result.continuity_valid,
                    boundary_result.correction_applied
                )
                
                # Extract audio segment using detected boundaries
                song_chunk = self._audio_from_window_ms(
                    window,
                    boundary_result.start_frame,
                    boundary_result.end_frame,
                )
                cleanup_ms = boundary_result.end_frame
                
                # Log boundary detection details
                if boundary_result.grace_period_applied:
                    logger.debug("Grace period applied for smoother track transition")
                if boundary_result.correction_applied:
                    logger.debug("Boundary correction applied to improve audio continuity")
                if not boundary_result.continuity_valid:
                    logger.warning("Audio continuity validation failed - potential audio quality issues")

            # Validate duration with enhanced tolerance for grace period adjustments
            expected_duration_ms = getattr(start_marker.track_info, "duration_ms", 0)
            actual_duration_ms = len(song_chunk)
            
            # Increase tolerance when grace period or correction is applied
            tolerance_ms = 4000  # Base tolerance
            if boundary_result and (boundary_result.grace_period_applied or boundary_result.correction_applied):
                tolerance_ms = 6000  # Increased tolerance for enhanced processing
            
            if expected_duration_ms > 0:
                duration_diff = abs(actual_duration_ms - expected_duration_ms)
                if duration_diff > tolerance_ms:
                    logger.warning(
                        "Track duration mismatch too large (expected %dms, got %dms, diff %dms). Skipping save.",
                        expected_duration_ms, actual_duration_ms, duration_diff
                    )
                    # Validate frame integrity before skipping
                    is_valid, diagnostic = self.boundary_detector.validate_frame_integrity()
                    if not is_valid:
                        logger.warning("Frame integrity issue detected: %s", diagnostic)
                    return False  # Processing failed due to duration mismatch
                else:
                    logger.debug(
                        "Duration validation passed (expected %dms, got %dms, diff %dms)",
                        expected_duration_ms, actual_duration_ms, duration_diff
                    )
                    # Apply final boundary correction before export if needed
                    corrected_chunk = self._apply_final_boundary_correction(
                        song_chunk, start_marker.track_info, boundary_result
                    )
                    
                    # Export with error handling
                    export_success = self._export_with_error_handling(corrected_chunk, start_marker.track_info)
                    if not export_success:
                        return False
            else:
                logger.debug("No expected duration available, saving track")
                # Apply final boundary correction before export
                corrected_chunk = self._apply_final_boundary_correction(
                    song_chunk, start_marker.track_info, boundary_result
                )
                
                # Export with error handling
                export_success = self._export_with_error_handling(corrected_chunk, start_marker.track_info)
                if not export_success:
                    return False

            # Clean up buffer and markers using the cleanup timestamp
            if window.from_ledger:
                retain_frame = max(
                    self.chunk_ledger.base_frame,
                    self._marker_frame(end_marker) - self._boundary_context_frames(),
                )
                dropped_ms = self._drop_audio_before_frame(retain_frame)
            else:
                self._drop_continuous_buffer_before(cleanup_ms)
                dropped_ms = cleanup_ms
            self.track_markers = [
                self._rebase_marker_after_drop(marker, dropped_ms)
                for marker in self.track_markers[1:]
            ]
            
            return True
            
        except Exception as e:
            logger.error("Error in internal segment processing: %s", e)
            raise  # Re-raise for error recovery handling

    def _apply_final_boundary_correction(self, audio_chunk: AudioSegment, 
                                        track_info: TrackInfo, 
                                        boundary_result: Optional[BoundaryResult]) -> AudioSegment:
        """
        Apply final boundary correction before track export to ensure optimal audio quality.
        
        Args:
            audio_chunk: The audio segment to be corrected
            track_info: Track information for validation
            boundary_result: Boundary detection result, if available
            
        Returns:
            Corrected audio segment ready for export
        """
        if not boundary_result or not audio_chunk:
            return audio_chunk
            
        corrected_chunk = audio_chunk
        
        # Apply fade-in/fade-out for smoother transitions if grace period was applied
        if boundary_result.grace_period_applied:
            fade_duration_ms = min(50, len(corrected_chunk) // 10)  # Max 50ms or 10% of track
            
            if fade_duration_ms > 0:
                # Apply gentle fade-in at the beginning
                corrected_chunk = corrected_chunk.fade_in(fade_duration_ms)
                # Apply gentle fade-out at the end
                corrected_chunk = corrected_chunk.fade_out(fade_duration_ms)
                logger.debug(f"Applied {fade_duration_ms}ms fade-in/out for smoother transitions")
        
        # Validate final audio segment integrity
        if len(corrected_chunk) == 0:
            logger.error("Final boundary correction resulted in empty audio segment")
            return audio_chunk  # Return original if correction failed
            
        # Log final correction details
        original_duration = len(audio_chunk)
        corrected_duration = len(corrected_chunk)
        
        if original_duration != corrected_duration:
            logger.debug(
                f"Final boundary correction: {original_duration}ms -> {corrected_duration}ms "
                f"(diff: {corrected_duration - original_duration}ms)"
            )
            
        return corrected_chunk

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _get_track_path(self, t: TrackInfo) -> Path:
        if self.bundle_album_name:
            safe_artist = sanitize("Various Artists")
            safe_album = sanitize(self.bundle_album_name)
        else:
            safe_artist = sanitize(t.artist)
            safe_album = sanitize(t.album)
        safe_title = sanitize(t.title)

        folder = self.output_dir / safe_artist / safe_album
        num_prefix = ""
        if t.track_number:
            try:
                num_prefix = f"{int(t.track_number):02d} - "
            except Exception:
                num_prefix = f"{t.track_number} - "
        return folder / f"{num_prefix}{safe_title}.{self.format}"

    def _download_artwork(self, uri: str) -> bytes:
        """Download cover artwork through the manager's pooled session."""
        return self.artwork_session.get(uri, timeout=10).content

    def _export(self, segment: Iterable | AudioSegment, track_info: TrackInfo) -> None:
        """Export ``segment`` to disk and tag it with metadata."""
        if self.bundle_playlist:
            track_info = track_info._replace(track_number=self.bundle_track_number)

        if isinstance(segment, np.ndarray):
            if segment.dtype.kind == "f":
                segment = np.clip(segment, -1.0, 1.0)
                segment = (segment * np.iinfo(np.int16).max).astype(np.int16)
            audio_segment = AudioSegment(
                segment.tobytes(),
                frame_rate=self.samplerate,
                sample_width=2,
                channels=segment.shape[1],
            )
        elif isinstance(segment, AudioSegment):
            audio_segment = segment
        else:
            raise TypeError("segment must be numpy array or AudioSegment")

        path = self._get_track_path(track_info)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_already_exists = path.exists()
        if file_already_exists and not self.allow_overwrite:
            logger.info("File %s already exists, skipping export but adding to playlist", path)
        else:
            if file_already_exists:
                logger.info("File %s already exists, overwriting...", path)
            audio_segment.export(path, format=self.format, bitrate="320k")

            try:
                tags = EasyID3(path)
            except Exception:
                tags = EasyID3()

            # Fetch year and genre from LastFM if not present
            year = track_info.year
            genre = track_info.genre

            if not year or not genre:
                if not self.lastfm_api_key:
                    logger.debug("LastFM API key not set, skipping metadata fetch")
                else:
                    try:
                        lastfm = get_lastfm_client(api_key=self.lastfm_api_key)
                        lastfm_metadata = lastfm.get_track_metadata(
                            track_info.artist,
                            track_info.title,
                            track_info.album
                        )
                        if not year and lastfm_metadata.year:
                            year = lastfm_metadata.year
                            logger.info("Fetched year from LastFM: %s", year)
                        if not genre and lastfm_metadata.genres:
                            # Join multiple genres with semicolon
                            genre = "; ".join(lastfm_metadata.genres)
                            logger.info("Fetched genres from LastFM: %s", genre)
                        if not lastfm_metadata.year and not lastfm_metadata.genres:
                            logger.debug("No LastFM metadata available for %s - %s",
                                       track_info.artist, track_info.title)
                    except Exception as e:
                        logger.warning("Failed to fetch LastFM metadata: %s", e)

            # If still no year, try to parse from track title or album name
            # Common patterns: "Song - 2008 Remaster", "Remastered 2006", "2004 Remaster"
            if not year:
                import re
                # Check track title first
                year_match = re.search(r'(?:remaster(?:ed)?|remix|edition|version)\s+(\d{4})|(\d{4})\s+remaster',
                                      track_info.title, re.IGNORECASE)
                if not year_match:
                    # Check album name
                    year_match = re.search(r'(?:remaster(?:ed)?|remix|edition|version)\s+(\d{4})|(\d{4})\s+remaster',
                                          track_info.album, re.IGNORECASE)

                if year_match:
                    parsed_year = int(year_match.group(1) or year_match.group(2))
                    if 1900 <= parsed_year <= 2100:
                        year = parsed_year
                        logger.info("Parsed year from track/album name: %s", year)

            tags["artist"] = track_info.artist
            tags["title"] = track_info.title
            if self.bundle_album_name:
                tags["album"] = self.bundle_album_name
                tags["albumartist"] = "Various Artists"
            else:
                tags["album"] = track_info.album
                tags["albumartist"] = track_info.artist
            if track_info.track_number:
                tags["tracknumber"] = str(track_info.track_number)
            if year:
                tags["date"] = str(year)
            if genre:
                tags["genre"] = genre
            tags.save(path)

            try:
                audio = ID3(path)
            except Exception as e:
                logger.warning("Could not load file for tagging: %s", e)
                audio = None

            if audio and track_info.art_uri:
                try:
                    # For bundle playlists, use unified album art
                    if self.bundle_playlist:
                        if self.bundle_album_art is None:
                            # Download album art once for the bundle
                            if self.bundle_album_art_uri:
                                # Use custom artwork URL if provided
                                logger.info("Downloading custom album art for bundle playlist from: %s", self.bundle_album_art_uri)
                                self.bundle_album_art = self._download_artwork(
                                    self.bundle_album_art_uri
                                )
                            else:
                                # Fall back to first track's artwork
                                logger.info("Downloading first track's album art for bundle playlist")
                                self.bundle_album_art = self._download_artwork(track_info.art_uri)
                        # Use the stored unified album art for all tracks
                        img = self.bundle_album_art
                        logger.debug("Using unified album art for bundle track")
                    else:
                        # For non-bundle playlists, download individual track artwork
                        img = self._download_artwork(track_info.art_uri)

                    audio.add(APIC(3, "image/jpeg", 3, "Front cover", img))
                except Exception as e:
                    logger.warning("Failed to download or embed cover art: %s", e)

            if audio:
                audio.save()
        
        # Add to playlist with base path remapping
        if self.playlist_file:
            try:
                # Convert local path to remote path using playlist_base_path
                # Get relative path from output_dir and join with base path
                try:
                    relative_path = path.relative_to(self.output_dir)
                    playlist_entry = Path(self.playlist_base_path) / relative_path
                except ValueError:
                    # If path is not relative to output_dir, use absolute path
                    playlist_entry = path

                path_str = str(playlist_entry)
                if path_str not in self.playlist_tracks:
                    self.playlist_file.write(f"{path_str}\n")
                    self.playlist_file.flush()
                    self.playlist_tracks.add(path_str)
                    logger.debug("Added track to playlist: %s (mapped from %s)", path_str, path)
                else:
                    logger.debug("Track already in playlist, skipping: %s", path)
            except Exception as e:
                logger.warning("Failed to add track to playlist: %s", e)

        if file_already_exists:
            logger.info("Added existing file to playlist: %s", path)
        else:
            logger.info("Saved %s", path)
        # Notify UI of successful save
        if hasattr(self, 'ui_callback') and self.ui_callback:
            self.ui_callback("saved", track_info)

        if self.bundle_playlist:
            self.bundle_track_number += 1

    def _export_with_error_handling(self, segment: AudioSegment, track_info: TrackInfo) -> bool:
        """
        Export audio segment with comprehensive error handling and recovery.
        
        Args:
            segment: Audio segment to export
            track_info: Track information for export
            
        Returns:
            True if export was successful, False otherwise
        """
        max_export_retries = 3
        
        for attempt in range(max_export_retries):
            try:
                # Attempt export
                self._export(segment, track_info)
                
                if attempt > 0:
                    logger.info("Export recovered after %d attempts: %s", attempt, track_info.title)
                    
                return True
                
            except Exception as e:
                self.export_errors += 1
                
                # Handle export error through error recovery manager
                context = f"export_{track_info.title}"
                recovery_action = self.error_recovery.handle_error(e, context)
                
                logger.warning("Export error (attempt %d/%d) for %s: %s -> %s", 
                             attempt + 1, max_export_retries, 
                             track_info.title, type(e).__name__, recovery_action.value)
                
                # Notify UI of export error
                if self.ui_callback:
                    self.ui_callback("export_error", {
                        "track": track_info,
                        "error": str(e),
                        "attempt": attempt + 1,
                        "recovery_action": recovery_action.value
                    })
                
                # Handle different recovery actions
                if recovery_action == RecoveryAction.RETRY and attempt < max_export_retries - 1:
                    # Attempt export recovery
                    recovery_success = self._attempt_export_recovery(e, segment, track_info)
                    if recovery_success:
                        logger.info("Export recovery successful, retrying")
                        continue
                    else:
                        logger.warning("Export recovery failed, continuing with retry")
                        continue
                        
                elif recovery_action == RecoveryAction.GRACEFUL_DEGRADE:
                    # Attempt graceful degradation for export
                    if self.enable_graceful_degradation:
                        degraded_success = self._attempt_export_degradation(segment, track_info, e)
                        if degraded_success:
                            self.degraded_exports += 1
                            logger.info("Export graceful degradation successful for: %s", track_info.title)
                            return True
                        else:
                            logger.warning("Export graceful degradation failed for: %s", track_info.title)
                    
                    # Continue to next attempt if degradation failed
                    if attempt < max_export_retries - 1:
                        continue
                    else:
                        return False
                        
                elif recovery_action == RecoveryAction.ESCALATE:
                    # Escalate immediately - don't retry
                    self.error_recovery.escalate_error(e, context)
                    logger.critical("Export error escalated: %s", track_info.title)
                    return False
                    
                else:
                    # For other recovery actions or final attempt, continue or fail
                    if attempt < max_export_retries - 1:
                        continue
                    else:
                        logger.error("All export recovery attempts exhausted for: %s", track_info.title)
                        return False
        
        return False

    def _attempt_segment_recovery(self, error: Exception, start_marker: TrackMarker, end_marker: TrackMarker) -> bool:
        """
        Attempt to recover from segment processing errors.
        
        Args:
            error: The error that occurred
            start_marker: Start marker for the segment
            end_marker: End marker for the segment
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            # Recovery strategies based on error type
            error_type = type(error).__name__
            
            if "Memory" in error_type:
                # Memory-related errors - try to free up memory
                logger.info("Attempting memory recovery for segment processing")
                
                # Clear any temporary buffers
                import gc
                gc.collect()
                
                # Reduce buffer size temporarily if possible
                if hasattr(self, 'continuous_buffer') and len(self.continuous_buffer) > 60000:  # > 1 minute
                    # Keep only the last 30 seconds plus current segment
                    segment_duration = self._marker_ms(end_marker) - self._marker_ms(start_marker)
                    keep_duration = max(30000, segment_duration + 10000)  # 30s or segment + 10s buffer
                    
                    trim_point = len(self.continuous_buffer) - keep_duration
                    if trim_point > 0:
                        self._drop_continuous_buffer_before(trim_point)
                        
                        # Adjust markers
                        self.track_markers = [
                            self._rebase_marker_after_drop(marker, trim_point)
                            for marker in self.track_markers
                        ]
                        
                        logger.info("Trimmed buffer for memory recovery: removed %dms", trim_point)
                
                return True
                
            elif "IO" in error_type or "OSError" in error_type:
                # I/O related errors - check disk space and permissions
                logger.info("Attempting I/O recovery for segment processing")
                
                # Check if output directory is accessible
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    test_file = self.output_dir / ".test_write"
                    test_file.write_text("test")
                    test_file.unlink()
                    logger.info("Output directory accessibility verified")
                    return True
                except Exception as io_error:
                    logger.error("Output directory not accessible: %s", io_error)
                    return False
                    
            elif "Audio" in error_type or "pydub" in error_type.lower():
                # Audio processing errors - try to validate and repair audio data
                logger.info("Attempting audio recovery for segment processing")
                
                # Check if continuous buffer is valid
                if len(self.continuous_buffer) == 0:
                    logger.warning("Empty continuous buffer detected")
                    return False
                
                # Validate segment boundaries
                start_ms = self._marker_ms(start_marker)
                end_ms = self._marker_ms(end_marker)

                if start_ms >= len(self.continuous_buffer):
                    logger.warning("Start marker beyond buffer length")
                    return False
                    
                if end_ms > len(self.continuous_buffer):
                    logger.warning("End marker beyond buffer length, adjusting")
                    # Adjust end marker to buffer length
                    adjusted_marker = end_marker._replace(
                        timestamp=len(self.continuous_buffer),
                        frame=self._buffer_ms_to_frame(len(self.continuous_buffer)),
                    )
                    # Update the marker in the list
                    for i, marker in enumerate(self.track_markers):
                        if marker == end_marker:
                            self.track_markers[i] = adjusted_marker
                            break
                
                return True
                
            else:
                # Generic recovery - just wait a moment and try again
                logger.info("Attempting generic recovery for segment processing")
                import time
                time.sleep(0.1)
                return True
                
        except Exception as recovery_error:
            logger.error("Error during segment recovery: %s", recovery_error)
            return False

    def _attempt_export_recovery(self, error: Exception, segment: AudioSegment, track_info: TrackInfo) -> bool:
        """
        Attempt to recover from export errors.
        
        Args:
            error: The error that occurred
            segment: Audio segment to export
            track_info: Track information
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            error_type = type(error).__name__
            
            if "Permission" in error_type or "OSError" in error_type:
                # Permission or file system errors
                logger.info("Attempting file system recovery for export")
                
                # Ensure output directory exists and is writable
                path = self._get_track_path(track_info)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if file already exists and remove if necessary
                if path.exists():
                    try:
                        path.unlink()
                        logger.info("Removed existing file for retry: %s", path)
                    except Exception:
                        # Try alternative filename
                        import time
                        timestamp = int(time.time())
                        alt_path = path.with_stem(f"{path.stem}_{timestamp}")
                        logger.info("Using alternative filename: %s", alt_path)
                        # Update the path for this export attempt
                        return True
                
                return True
                
            elif "Memory" in error_type:
                # Memory errors during export
                logger.info("Attempting memory recovery for export")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Try to reduce audio quality temporarily if segment is very long
                if len(segment) > 300000:  # > 5 minutes
                    logger.info("Large segment detected, may need quality reduction")
                
                return True
                
            elif "requests" in error_type.lower() or "connection" in error_type.lower():
                # Network errors (likely album art download)
                logger.info("Attempting network recovery for export")
                
                # Skip album art download for this retry
                # This would require modifying the export method to accept a flag
                # For now, just wait and retry
                import time
                time.sleep(1.0)
                return True
                
            else:
                # Generic recovery
                logger.info("Attempting generic recovery for export")
                import time
                time.sleep(0.5)
                return True
                
        except Exception as recovery_error:
            logger.error("Error during export recovery: %s", recovery_error)
            return False

    def _attempt_graceful_degradation(self, start_marker: TrackMarker, end_marker: TrackMarker, error: Exception) -> bool:
        """
        Attempt graceful degradation for segment processing.
        
        Args:
            start_marker: Start marker for the segment
            end_marker: End marker for the segment
            error: The error that occurred
            
        Returns:
            True if graceful degradation was successful, False otherwise
        """
        try:
            logger.info("Attempting graceful degradation for segment processing: %s", start_marker.track_info.title)
            
            # Simplified processing without advanced features
            try:
                # Use basic timestamp-based cutting without boundary detection
                start_ms = self._marker_ms(start_marker)
                end_ms = self._marker_ms(end_marker)
                song_chunk = self.continuous_buffer[start_ms : end_ms]
                
                if len(song_chunk) == 0:
                    logger.warning("Empty audio chunk in graceful degradation")
                    return False
                
                # Skip duration validation in degraded mode
                logger.info("Graceful degradation: skipping duration validation")
                
                # Export with minimal processing
                degraded_success = self._export_with_minimal_processing(song_chunk, start_marker.track_info)
                
                if degraded_success:
                    # Clean up buffer and markers
                    cleanup_timestamp = end_ms
                    self._drop_continuous_buffer_before(cleanup_timestamp)
                    self.track_markers = [
                        self._rebase_marker_after_drop(marker, cleanup_timestamp)
                        for marker in self.track_markers[1:]
                    ]
                    
                    logger.info("Graceful degradation successful: %s", start_marker.track_info.title)
                    
                    # Notify UI of degraded export
                    if self.ui_callback:
                        self.ui_callback("degraded_export", {
                            "track": start_marker.track_info,
                            "reason": str(error)
                        })
                    
                    return True
                else:
                    logger.warning("Graceful degradation export failed: %s", start_marker.track_info.title)
                    return False
                    
            except Exception as degradation_error:
                logger.error("Error in graceful degradation processing: %s", degradation_error)
                return False
                
        except Exception as e:
            logger.error("Error in graceful degradation attempt: %s", e)
            return False

    def _attempt_export_degradation(self, segment: AudioSegment, track_info: TrackInfo, error: Exception) -> bool:
        """
        Attempt graceful degradation for export.
        
        Args:
            segment: Audio segment to export
            track_info: Track information
            error: The error that occurred
            
        Returns:
            True if graceful degradation was successful, False otherwise
        """
        try:
            logger.info("Attempting export graceful degradation: %s", track_info.title)
            
            # Try minimal processing export
            success = self._export_with_minimal_processing(segment, track_info)
            
            if success:
                logger.info("Export graceful degradation successful: %s", track_info.title)
                
                # Notify UI of degraded export
                if self.ui_callback:
                    self.ui_callback("degraded_export", {
                        "track": track_info,
                        "reason": str(error),
                        "type": "export"
                    })
                
                return True
            else:
                logger.warning("Export graceful degradation failed: %s", track_info.title)
                return False
                
        except Exception as e:
            logger.error("Error in export graceful degradation: %s", e)
            return False

    def _export_with_minimal_processing(self, segment: AudioSegment, track_info: TrackInfo) -> bool:
        """
        Export audio with minimal processing for graceful degradation.
        
        Args:
            segment: Audio segment to export
            track_info: Track information
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            path = self._get_track_path(track_info)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already exists and overwrite is not allowed
            if path.exists() and not self.allow_overwrite:
                logger.info("File %s already exists, skipping minimal export", path)
                return True

            if path.exists():
                logger.info("File %s already exists, overwriting with minimal export...", path)
            
            # Export with basic settings, no album art
            segment.export(path, format=self.format, bitrate="192k")  # Lower bitrate for degraded mode
            
            # Add basic tags only
            try:
                tags = EasyID3(path)
            except Exception:
                tags = EasyID3()
            
            # Only add essential tags
            tags["artist"] = track_info.artist
            tags["title"] = track_info.title
            if self.bundle_album_name:
                tags["album"] = self.bundle_album_name
                tags["albumartist"] = "Various Artists"
            else:
                tags["album"] = track_info.album
                tags["albumartist"] = track_info.artist
            tags.save(path)
            
            # Skip album art in degraded mode
            logger.info("Minimal export saved (no album art): %s", path)
            
            # Notify UI of successful save
            if self.ui_callback:
                self.ui_callback("saved", track_info)
            
            return True
            
        except Exception as e:
            logger.error("Error in minimal processing export: %s", e)
            return False

    def _handle_processing_failure(self, start_marker: TrackMarker, end_marker: TrackMarker) -> None:
        """
        Handle complete processing failure after all recovery attempts.

        Args:
            start_marker: Start marker for the failed segment
            end_marker: End marker for the failed segment
        """
        # CRITICAL: Check if file actually exists before logging failure
        # File might have been created successfully even if processing returned False
        expected_path = self._get_track_path(start_marker.track_info)

        if expected_path.exists() and expected_path.stat().st_size > 0:
            # File exists and is not empty - processing actually succeeded!
            logger.info("Track file exists despite processing errors - recovery was successful: %s",
                       start_marker.track_info.title)

            # Notify UI of successful recovery instead of failure
            if self.ui_callback:
                self.ui_callback("recovery_success", {
                    "track": start_marker.track_info,
                    "message": "Track saved successfully after error recovery"
                })
        else:
            # File truly doesn't exist - this is a real failure
            logger.error("Complete processing failure for: %s (file not created)", start_marker.track_info.title)

            # Notify UI of actual processing failure
            if self.ui_callback:
                self.ui_callback("processing_failure", {
                    "track": start_marker.track_info,
                    "error": "All recovery attempts failed - file not created"
                })

        # Clean up buffer and markers to continue with next track
        cleanup_timestamp = self._marker_ms(end_marker)
        self._drop_continuous_buffer_before(cleanup_timestamp)
        self.track_markers = [
            self._rebase_marker_after_drop(marker, cleanup_timestamp)
            for marker in self.track_markers[1:]
        ]
        
        # Generate error diagnostics
        if self.enable_error_recovery:
            diagnostics = self.error_recovery.get_diagnostics()
            logger.warning("Error diagnostics - Total errors: %d, Recovery rate: %.1f%%", 
                         diagnostics.total_errors, diagnostics.recovery_success_rate * 100)
            
            if diagnostics.recommendations:
                logger.info("Error recovery recommendations:")
                for rec in diagnostics.recommendations[:3]:  # Show top 3
                    logger.info("  - %s", rec)

    def get_error_statistics(self) -> dict:
        """
        Get comprehensive error statistics for the segment manager.
        
        Returns:
            Dictionary containing error statistics and recovery information
        """
        stats = {
            "processing_errors": self.processing_errors,
            "export_errors": self.export_errors,
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "degraded_exports": self.degraded_exports,
            "current_processing_track": self.current_processing_track.title if self.current_processing_track else None,
            "processing_retry_count": self.processing_retry_count,
            "last_successful_export": self.last_successful_export.title if self.last_successful_export else None,
            "error_recovery_enabled": self.enable_error_recovery,
            "graceful_degradation_enabled": self.enable_graceful_degradation,
            "max_processing_retries": self.max_processing_retries
        }
        
        # Add error recovery manager statistics if available
        if self.error_recovery:
            recovery_stats = self.error_recovery.get_statistics()
            stats["error_recovery_manager"] = recovery_stats
        
        return stats

    def generate_error_report(self) -> str:
        """
        Generate a comprehensive error report for diagnostics.
        
        Returns:
            Formatted error report string
        """
        stats = self.get_error_statistics()
        
        report_lines = [
            "=== SegmentManager Error Report ===",
            f"Processing Errors: {stats['processing_errors']}",
            f"Export Errors: {stats['export_errors']}",
            f"Recovery Attempts: {stats['recovery_attempts']}",
            f"Successful Recoveries: {stats['successful_recoveries']}",
            f"Degraded Exports: {stats['degraded_exports']}",
            f"Error Recovery Enabled: {stats['error_recovery_enabled']}",
            f"Graceful Degradation Enabled: {stats['graceful_degradation_enabled']}",
            ""
        ]
        
        # Add current state information
        if stats['current_processing_track']:
            report_lines.append(f"Current Processing Track: {stats['current_processing_track']}")
            report_lines.append(f"Processing Retry Count: {stats['processing_retry_count']}")
        
        if stats['last_successful_export']:
            report_lines.append(f"Last Successful Export: {stats['last_successful_export']}")
        
        report_lines.append("")
        
        # Add error recovery manager diagnostics if available
        if self.error_recovery:
            try:
                diagnostics = self.error_recovery.get_diagnostics()
                report_lines.extend([
                    "=== Error Recovery Manager Diagnostics ===",
                    f"Total Errors: {diagnostics.total_errors}",
                    f"Error Rate (per hour): {diagnostics.error_rate_per_hour:.2f}",
                    f"Recovery Success Rate: {diagnostics.recovery_success_rate:.1%}",
                    ""
                ])
                
                if diagnostics.most_common_errors:
                    report_lines.append("Most Common Errors:")
                    for error_type, count in diagnostics.most_common_errors:
                        report_lines.append(f"  - {error_type}: {count}")
                    report_lines.append("")
                
                if diagnostics.recommendations:
                    report_lines.append("Recommendations:")
                    for rec in diagnostics.recommendations:
                        report_lines.append(f"  - {rec}")
                    report_lines.append("")
                        
            except Exception as e:
                report_lines.append(f"Error generating recovery diagnostics: {e}")
        
        return "\n".join(report_lines)
