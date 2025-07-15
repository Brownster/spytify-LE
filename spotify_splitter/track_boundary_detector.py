"""
Track boundary detection with grace period support and audio continuity validation.

This module provides enhanced track boundary detection capabilities to prevent
audio loss and corruption during track transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque
import numpy as np
from pydub import AudioSegment

# TrackInfo will be imported where needed to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class BoundaryResult:
    """Result of track boundary detection and validation."""
    start_frame: int
    end_frame: int
    confidence: float
    continuity_valid: bool
    grace_period_applied: bool
    correction_applied: bool
    original_start: Optional[int] = None
    original_end: Optional[int] = None


@dataclass
class TrackMarker:
    """Enhanced track marker with timing information."""
    timestamp: int
    track_info: any  # TrackInfo from mpris module
    confidence: float = 1.0
    grace_period_start: Optional[int] = None
    grace_period_end: Optional[int] = None


class AudioContinuityValidator:
    """Validates audio continuity across track boundaries."""
    
    def __init__(self, window_size_ms: int = 100, threshold: float = 0.1):
        """
        Initialize continuity validator.
        
        Args:
            window_size_ms: Size of analysis window in milliseconds
            threshold: Threshold for continuity validation (0.0-1.0)
        """
        self.window_size_ms = window_size_ms
        self.threshold = threshold
    
    def validate_continuity(self, before: AudioSegment, after: AudioSegment) -> bool:
        """
        Validate audio continuity between two segments.
        
        Args:
            before: Audio segment before the boundary
            after: Audio segment after the boundary
            
        Returns:
            True if continuity is valid, False otherwise
        """
        if not before or not after:
            return False
            
        # Get analysis windows from end of before and start of after
        window_ms = min(self.window_size_ms, len(before), len(after))
        
        if window_ms < 10:  # Need at least 10ms for analysis
            logger.debug("Segments too short for continuity analysis")
            return True  # Assume valid for short segments
            
        # Extract analysis windows using millisecond slicing
        before_window = before[-window_ms:]
        after_window = after[:window_ms]
        
        # Convert to numpy arrays for analysis
        before_samples = np.array(before_window.get_array_of_samples())
        after_samples = np.array(after_window.get_array_of_samples())
        
        # Handle stereo audio
        if before_window.channels == 2:
            before_samples = before_samples.reshape((-1, 2))
            after_samples = after_samples.reshape((-1, 2))
            # Use left channel for analysis
            before_samples = before_samples[:, 0]
            after_samples = after_samples[:, 0]
        
        # Calculate RMS levels
        before_rms = np.sqrt(np.mean(before_samples.astype(float) ** 2))
        after_rms = np.sqrt(np.mean(after_samples.astype(float) ** 2))
        
        # Check for sudden level changes
        if before_rms > 0 and after_rms > 0:
            level_ratio = abs(before_rms - after_rms) / max(before_rms, after_rms)
            if level_ratio > self.threshold:
                logger.debug(f"Continuity check failed: level change {level_ratio:.3f} > {self.threshold}")
                return False
        elif before_rms == 0 and after_rms > 0:
            # Transition from silence to sound
            return False
        elif before_rms > 0 and after_rms == 0:
            # Transition from sound to silence
            return False
        
        # Check for silence gaps
        silence_threshold = max(before_rms, after_rms) * 0.01  # 1% of signal level
        before_silent = before_rms < silence_threshold
        after_silent = after_rms < silence_threshold
        
        if before_silent != after_silent:
            logger.debug("Continuity check failed: silence mismatch")
            return False
            
        return True


class FrameAccountingSystem:
    """Tracks audio frames to prevent loss or duplication."""
    
    def __init__(self):
        self.processed_frames = 0
        self.expected_frames = 0
        self.frame_history = deque(maxlen=1000)  # Track recent frame operations
    
    def record_frames(self, start_frame: int, end_frame: int, operation: str):
        """Record frame operation for accounting."""
        frame_count = end_frame - start_frame
        self.frame_history.append({
            'start': start_frame,
            'end': end_frame,
            'count': frame_count,
            'operation': operation
        })
        
        if operation == 'processed':
            self.processed_frames += frame_count
        elif operation == 'expected':
            self.expected_frames += frame_count
    
    def validate_frame_integrity(self) -> Tuple[bool, str]:
        """
        Validate that no frames were lost or duplicated.
        
        Returns:
            Tuple of (is_valid, diagnostic_message)
        """
        if self.expected_frames == 0:
            return True, "No frames expected yet"
            
        if self.processed_frames == self.expected_frames:
            return True, f"Frame integrity valid: {self.processed_frames} frames"
        elif self.processed_frames < self.expected_frames:
            lost = self.expected_frames - self.processed_frames
            return False, f"Frame loss detected: {lost} frames missing"
        else:
            duplicated = self.processed_frames - self.expected_frames
            return False, f"Frame duplication detected: {duplicated} extra frames"
    
    def reset(self):
        """Reset frame accounting."""
        self.processed_frames = 0
        self.expected_frames = 0
        self.frame_history.clear()


class TrackBoundaryDetector:
    """
    Enhanced track boundary detection with grace period support and continuity validation.
    """
    
    def __init__(self, grace_period_ms: int = 500, max_correction_ms: int = 2000):
        """
        Initialize track boundary detector.
        
        Args:
            grace_period_ms: Grace period in milliseconds for track transitions
            max_correction_ms: Maximum correction allowed in milliseconds
        """
        self.grace_period_ms = grace_period_ms
        self.max_correction_ms = max_correction_ms
        self.boundary_cache = {}
        self.continuity_validator = AudioContinuityValidator()
        self.frame_accounting = FrameAccountingSystem()
        
    def detect_boundary(self, audio_buffer: AudioSegment, 
                       markers: List[TrackMarker]) -> Optional[BoundaryResult]:
        """
        Detect and validate track boundary with grace period and continuity checks.
        
        Args:
            audio_buffer: Continuous audio buffer
            markers: List of track markers
            
        Returns:
            BoundaryResult with boundary information, or None if invalid
        """
        if len(markers) < 2:
            return None
            
        start_marker = markers[0]
        end_marker = markers[1]
        
        # Apply grace period
        grace_start, grace_end = self.apply_grace_period(
            start_marker.timestamp, end_marker.timestamp
        )
        
        # Initial boundary detection
        boundary_result = BoundaryResult(
            start_frame=grace_start,
            end_frame=grace_end,
            confidence=min(start_marker.confidence, end_marker.confidence),
            continuity_valid=False,
            grace_period_applied=True,
            correction_applied=False,
            original_start=start_marker.timestamp,
            original_end=end_marker.timestamp
        )
        
        # Extract segments for validation
        if grace_start >= len(audio_buffer) or grace_end > len(audio_buffer):
            logger.warning("Boundary extends beyond audio buffer")
            return None
            
        # Validate continuity
        boundary_result.continuity_valid = self._validate_boundary_continuity(
            audio_buffer, boundary_result
        )
        
        # Apply correction if needed
        if not boundary_result.continuity_valid:
            corrected_result = self.correct_boundary(audio_buffer, boundary_result)
            if corrected_result:
                boundary_result = corrected_result
                
        # Record frame accounting
        self.frame_accounting.record_frames(
            boundary_result.start_frame, 
            boundary_result.end_frame, 
            'expected'
        )
        
        return boundary_result
    
    def apply_grace_period(self, start_timestamp: int, end_timestamp: int) -> Tuple[int, int]:
        """
        Apply grace period to track boundaries.
        
        Args:
            start_timestamp: Original start timestamp
            end_timestamp: Original end timestamp
            
        Returns:
            Tuple of (adjusted_start, adjusted_end)
        """
        # Apply grace period - extend start backward and end forward
        grace_frames = self.grace_period_ms  # Assuming 1ms = 1 frame for simplicity
        
        adjusted_start = max(0, start_timestamp - grace_frames // 2)
        adjusted_end = end_timestamp + grace_frames // 2
        
        logger.debug(f"Applied grace period: {start_timestamp}-{end_timestamp} -> {adjusted_start}-{adjusted_end}")
        
        return adjusted_start, adjusted_end
    
    def correct_boundary(self, audio_buffer: AudioSegment, 
                        boundary: BoundaryResult) -> Optional[BoundaryResult]:
        """
        Attempt to correct boundary timing mismatches.
        
        Args:
            audio_buffer: Audio buffer containing the track
            boundary: Original boundary result
            
        Returns:
            Corrected BoundaryResult or None if correction failed
        """
        max_correction_frames = self.max_correction_ms
        
        # Try different correction offsets
        best_result = None
        best_confidence = 0.0
        
        for offset in range(-max_correction_frames, max_correction_frames + 1, 100):
            test_start = max(0, boundary.start_frame + offset)
            test_end = min(len(audio_buffer), boundary.end_frame + offset)
            
            if test_start >= test_end:
                continue
                
            test_result = BoundaryResult(
                start_frame=test_start,
                end_frame=test_end,
                confidence=boundary.confidence,
                continuity_valid=False,
                grace_period_applied=boundary.grace_period_applied,
                correction_applied=True,
                original_start=boundary.original_start,
                original_end=boundary.original_end
            )
            
            # Test continuity with correction
            if self._validate_boundary_continuity(audio_buffer, test_result):
                test_result.continuity_valid = True
                confidence = self._calculate_boundary_confidence(audio_buffer, test_result)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = test_result
                    best_result.confidence = confidence
        
        if best_result:
            logger.debug(f"Boundary corrected with offset: {best_result.start_frame - boundary.start_frame}")
            
        return best_result
    
    def _validate_boundary_continuity(self, audio_buffer: AudioSegment, 
                                    boundary: BoundaryResult) -> bool:
        """Validate continuity at boundary points."""
        # Check start boundary
        if boundary.start_frame > 0:
            before_start = audio_buffer[max(0, boundary.start_frame - 200):boundary.start_frame]
            after_start = audio_buffer[boundary.start_frame:boundary.start_frame + 200]
            
            if not self.continuity_validator.validate_continuity(before_start, after_start):
                return False
        
        # Check end boundary
        if boundary.end_frame < len(audio_buffer):
            before_end = audio_buffer[boundary.end_frame - 200:boundary.end_frame]
            after_end = audio_buffer[boundary.end_frame:boundary.end_frame + 200]
            
            if not self.continuity_validator.validate_continuity(before_end, after_end):
                return False
                
        return True
    
    def _calculate_boundary_confidence(self, audio_buffer: AudioSegment, 
                                     boundary: BoundaryResult) -> float:
        """Calculate confidence score for boundary detection."""
        # Simple confidence based on audio characteristics
        segment = audio_buffer[boundary.start_frame:boundary.end_frame]
        
        if not segment:
            return 0.0
            
        # Convert to numpy for analysis
        samples = np.array(segment.get_array_of_samples())
        if segment.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples[:, 0]  # Use left channel
            
        # Calculate RMS level
        rms = np.sqrt(np.mean(samples.astype(float) ** 2))
        
        # Higher RMS generally indicates better signal
        confidence = min(1.0, rms / 1000.0)  # Normalize to 0-1 range
        
        return confidence
    
    def validate_frame_integrity(self) -> Tuple[bool, str]:
        """Validate frame accounting integrity."""
        return self.frame_accounting.validate_frame_integrity()
    
    def reset_accounting(self):
        """Reset frame accounting system."""
        self.frame_accounting.reset()