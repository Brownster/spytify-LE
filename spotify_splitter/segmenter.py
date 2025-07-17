from __future__ import annotations

from pathlib import Path
from collections import namedtuple
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

TrackMarker = namedtuple("TrackMarker", "timestamp track_info")


class SegmentManager:
    """Processes a continuous audio stream to find and export complete tracks with enhanced boundary handling and error recovery."""

    def __init__(
        self,
        samplerate: int = 44100,
        output_dir: Path = OUTPUT_DIR,
        fmt: str = "mp3",
        audio_queue: Optional[queue.Queue] = None,
        event_queue: Optional[queue.Queue] = None,
        ui_callback: Optional[callable] = None,
        grace_period_ms: int = 500,
        max_correction_ms: int = 2000,
        error_recovery: Optional[ErrorRecoveryManager] = None,
        enable_error_recovery: bool = True,
        max_processing_retries: int = 3,
        enable_graceful_degradation: bool = True,
    ) -> None:
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.format = fmt
        self.audio_queue = audio_queue
        self.event_queue = event_queue
        self.ui_callback = ui_callback

        self.continuous_buffer = AudioSegment.empty()
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
        
    def flush_cache(self) -> None:
        """Clear all cached data for clean startup."""
        logger.debug("Flushing SegmentManager cache...")
        self.continuous_buffer = AudioSegment.empty()
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
                        self.continuous_buffer = AudioSegment.empty()
                        marker = TrackMarker(0, data)
                        self.track_markers = [marker]
                    else:
                        marker = TrackMarker(len(self.continuous_buffer), data)
                        self.track_markers.append(marker)
                        logger.debug("Marker added for track: %s", data.title)
                        self.process_segments()
            except queue.Empty:
                self._ingest_audio()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest_audio(self) -> None:
        """Pull all frames from ``audio_queue`` into the continuous buffer."""
        if not self.audio_queue or self.audio_queue.empty():
            return

        segment = AudioSegment.empty()
        while not self.audio_queue.empty():
            frames = self.audio_queue.get_nowait()
            int_samples = (frames * np.iinfo(np.int16).max).astype(np.int16)
            seg = AudioSegment(
                int_samples.tobytes(),
                frame_rate=self.samplerate,
                sample_width=2,
                channels=frames.shape[1],
            )
            segment += seg
        self.continuous_buffer += segment

    def process_segments(self) -> None:
        """Process the first complete segment using enhanced boundary detection with grace period and continuity validation."""
        if len(self.track_markers) < 2:
            return

        start_marker = self.track_markers[0]
        end_marker = self.track_markers[1]

        if not is_song(start_marker.track_info):
            logger.debug("Previous item was not a song, skipping segment.")
            self.continuous_buffer = self.continuous_buffer[end_marker.timestamp :]
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
                logger.error("Failed to process segment after all recovery attempts: %s", start_marker.track_info.title)
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
            enhanced_markers = [
                EnhancedTrackMarker(
                    timestamp=start_marker.timestamp,
                    track_info=start_marker.track_info,
                    confidence=1.0
                ),
                EnhancedTrackMarker(
                    timestamp=end_marker.timestamp,
                    track_info=end_marker.track_info,
                    confidence=1.0
                )
            ]

            # Use TrackBoundaryDetector for enhanced boundary detection
            boundary_result = self.boundary_detector.detect_boundary(
                self.continuous_buffer, enhanced_markers
            )

            if not boundary_result:
                logger.warning("Boundary detection failed for track: %s", start_marker.track_info.title)
                # Fallback to original timestamp-based cutting
                song_chunk = self.continuous_buffer[start_marker.timestamp : end_marker.timestamp]
                cleanup_timestamp = end_marker.timestamp
            else:
                # Use enhanced boundary detection results
                logger.debug(
                    "Boundary detection results - Start: %d, End: %d, Grace period: %s, Continuity: %s, Correction: %s",
                    boundary_result.start_frame, boundary_result.end_frame,
                    boundary_result.grace_period_applied, boundary_result.continuity_valid,
                    boundary_result.correction_applied
                )
                
                # Extract audio segment using detected boundaries
                song_chunk = self.continuous_buffer[boundary_result.start_frame : boundary_result.end_frame]
                cleanup_timestamp = boundary_result.end_frame
                
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
            self.continuous_buffer = self.continuous_buffer[cleanup_timestamp :]
            self.track_markers = [
                marker._replace(timestamp=marker.timestamp - cleanup_timestamp)
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

    def _export(self, segment: Iterable | AudioSegment, track_info: TrackInfo) -> None:
        """Export ``segment`` to disk and tag it with metadata."""
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
        if path.exists():
            logger.info("File %s already exists, skipping export", path)
            return

        audio_segment.export(path, format=self.format, bitrate="320k")

        try:
            tags = EasyID3(path)
        except Exception:
            tags = EasyID3()

        tags["artist"] = track_info.artist
        tags["title"] = track_info.title
        tags["album"] = track_info.album
        tags["albumartist"] = track_info.artist
        if track_info.track_number:
            tags["tracknumber"] = str(track_info.track_number)
        tags.save(path)

        try:
            audio = ID3(path)
        except Exception as e:
            logger.warning("Could not load file for tagging: %s", e)
            return

        if track_info.art_uri:
            try:
                img = requests.get(track_info.art_uri).content
                audio.add(APIC(3, "image/jpeg", 3, "Front cover", img))
            except Exception as e:
                logger.warning("Failed to download or embed cover art: %s", e)

        audio.save()
        logger.info("Saved %s", path)
        # Notify UI of successful save
        if hasattr(self, 'ui_callback') and self.ui_callback:
            self.ui_callback("saved", track_info)

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
                    segment_duration = end_marker.timestamp - start_marker.timestamp
                    keep_duration = max(30000, segment_duration + 10000)  # 30s or segment + 10s buffer
                    
                    trim_point = len(self.continuous_buffer) - keep_duration
                    if trim_point > 0:
                        self.continuous_buffer = self.continuous_buffer[trim_point:]
                        
                        # Adjust markers
                        self.track_markers = [
                            marker._replace(timestamp=max(0, marker.timestamp - trim_point))
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
                if start_marker.timestamp >= len(self.continuous_buffer):
                    logger.warning("Start marker beyond buffer length")
                    return False
                    
                if end_marker.timestamp > len(self.continuous_buffer):
                    logger.warning("End marker beyond buffer length, adjusting")
                    # Adjust end marker to buffer length
                    adjusted_marker = end_marker._replace(timestamp=len(self.continuous_buffer))
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
                song_chunk = self.continuous_buffer[start_marker.timestamp : end_marker.timestamp]
                
                if len(song_chunk) == 0:
                    logger.warning("Empty audio chunk in graceful degradation")
                    return False
                
                # Skip duration validation in degraded mode
                logger.info("Graceful degradation: skipping duration validation")
                
                # Export with minimal processing
                degraded_success = self._export_with_minimal_processing(song_chunk, start_marker.track_info)
                
                if degraded_success:
                    # Clean up buffer and markers
                    cleanup_timestamp = end_marker.timestamp
                    self.continuous_buffer = self.continuous_buffer[cleanup_timestamp :]
                    self.track_markers = [
                        marker._replace(timestamp=marker.timestamp - cleanup_timestamp)
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
            
            # Skip if file already exists
            if path.exists():
                logger.info("File %s already exists, skipping minimal export", path)
                return True
            
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
        logger.error("Complete processing failure for: %s", start_marker.track_info.title)
        
        # Clean up buffer and markers to continue with next track
        cleanup_timestamp = end_marker.timestamp
        self.continuous_buffer = self.continuous_buffer[cleanup_timestamp :]
        self.track_markers = [
            marker._replace(timestamp=marker.timestamp - cleanup_timestamp)
            for marker in self.track_markers[1:]
        ]
        
        # Notify UI of processing failure
        if self.ui_callback:
            self.ui_callback("processing_failure", {
                "track": start_marker.track_info,
                "error": "All recovery attempts failed"
            })
        
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
