"""
Tests for TrackBoundaryDetector and related classes.
"""

import pytest
import numpy as np
from pydub import AudioSegment
from unittest.mock import Mock, patch

from spotify_splitter.track_boundary_detector import (
    TrackBoundaryDetector,
    AudioContinuityValidator,
    FrameAccountingSystem,
    BoundaryResult,
    TrackMarker
)
from collections import namedtuple

# Create TrackInfo namedtuple to match the one in mpris module
TrackInfo = namedtuple(
    "TrackInfo",
    "artist title album art_uri id track_number position duration_ms",
)


class TestAudioContinuityValidator:
    """Test AudioContinuityValidator functionality."""
    
    def setup_method(self):
        self.validator = AudioContinuityValidator(window_size_ms=100, threshold=0.3)
    
    def create_test_audio(self, duration_ms: int, frequency: float = 440.0, 
                         amplitude: float = 0.5) -> AudioSegment:
        """Create test audio segment with specified parameters."""
        sample_rate = 44100
        samples = int(sample_rate * duration_ms / 1000)
        
        # Generate sine wave
        t = np.linspace(0, duration_ms / 1000, samples, False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integers
        audio_data = (wave * 32767).astype(np.int16)
        
        return AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
    
    def test_validate_continuity_similar_levels(self):
        """Test continuity validation with similar audio levels."""
        before = self.create_test_audio(200, frequency=440, amplitude=0.5)
        after = self.create_test_audio(200, frequency=440, amplitude=0.5)
        
        result = self.validator.validate_continuity(before, after)
        assert result is True
    
    def test_validate_continuity_different_levels(self):
        """Test continuity validation with different audio levels."""
        before = self.create_test_audio(200, frequency=440, amplitude=0.5)
        after = self.create_test_audio(200, frequency=440, amplitude=0.1)  # Much quieter
        
        # The level ratio should be (0.5-0.1)/0.5 = 0.8, which is > 0.3 threshold
        result = self.validator.validate_continuity(before, after)
        assert result is False
    
    def test_validate_continuity_silence_mismatch(self):
        """Test continuity validation with silence mismatch."""
        before = self.create_test_audio(200, frequency=440, amplitude=0.5)
        after = self.create_test_audio(200, frequency=440, amplitude=0.001)  # Near silence
        
        result = self.validator.validate_continuity(before, after)
        assert result is False
    
    def test_validate_continuity_empty_segments(self):
        """Test continuity validation with empty segments."""
        before = AudioSegment.empty()
        after = self.create_test_audio(200)
        
        result = self.validator.validate_continuity(before, after)
        assert result is False
    
    def test_validate_continuity_short_segments(self):
        """Test continuity validation with segments shorter than window."""
        before = self.create_test_audio(50)  # Shorter than 100ms window
        after = self.create_test_audio(50)
        
        result = self.validator.validate_continuity(before, after)
        assert result is True  # Should assume valid for short segments


class TestFrameAccountingSystem:
    """Test FrameAccountingSystem functionality."""
    
    def setup_method(self):
        self.accounting = FrameAccountingSystem()
    
    def test_record_frames_processed(self):
        """Test recording processed frames."""
        self.accounting.record_frames(0, 1000, 'processed')
        assert self.accounting.processed_frames == 1000
        assert len(self.accounting.frame_history) == 1
    
    def test_record_frames_expected(self):
        """Test recording expected frames."""
        self.accounting.record_frames(0, 1000, 'expected')
        assert self.accounting.expected_frames == 1000
        assert len(self.accounting.frame_history) == 1
    
    def test_validate_frame_integrity_valid(self):
        """Test frame integrity validation when frames match."""
        self.accounting.record_frames(0, 1000, 'expected')
        self.accounting.record_frames(0, 1000, 'processed')
        
        is_valid, message = self.accounting.validate_frame_integrity()
        assert is_valid is True
        assert "1000 frames" in message
    
    def test_validate_frame_integrity_loss(self):
        """Test frame integrity validation when frames are lost."""
        self.accounting.record_frames(0, 1000, 'expected')
        self.accounting.record_frames(0, 800, 'processed')
        
        is_valid, message = self.accounting.validate_frame_integrity()
        assert is_valid is False
        assert "200 frames missing" in message
    
    def test_validate_frame_integrity_duplication(self):
        """Test frame integrity validation when frames are duplicated."""
        self.accounting.record_frames(0, 1000, 'expected')
        self.accounting.record_frames(0, 1200, 'processed')
        
        is_valid, message = self.accounting.validate_frame_integrity()
        assert is_valid is False
        assert "200 extra frames" in message
    
    def test_reset(self):
        """Test resetting frame accounting."""
        self.accounting.record_frames(0, 1000, 'expected')
        self.accounting.record_frames(0, 1000, 'processed')
        
        self.accounting.reset()
        
        assert self.accounting.processed_frames == 0
        assert self.accounting.expected_frames == 0
        assert len(self.accounting.frame_history) == 0


class TestTrackBoundaryDetector:
    """Test TrackBoundaryDetector functionality."""
    
    def setup_method(self):
        self.detector = TrackBoundaryDetector(grace_period_ms=500, max_correction_ms=2000)
        
        # Create mock track info
        self.track_info_1 = TrackInfo(
            artist="Test Artist",
            title="Test Track 1", 
            album="Test Album",
            art_uri="",
            id="spotify:track:1",
            track_number=1,
            position=0,
            duration_ms=180000
        )
        
        self.track_info_2 = TrackInfo(
            artist="Test Artist",
            title="Test Track 2",
            album="Test Album", 
            art_uri="",
            id="spotify:track:2",
            track_number=2,
            position=0,
            duration_ms=200000
        )
    
    def create_test_audio_buffer(self, duration_ms: int) -> AudioSegment:
        """Create a test audio buffer."""
        sample_rate = 44100
        samples = int(sample_rate * duration_ms / 1000)
        
        # Generate test audio with varying frequency to simulate track changes
        t = np.linspace(0, duration_ms / 1000, samples, False)
        wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        audio_data = (wave * 32767).astype(np.int16)
        
        return AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
    
    def test_apply_grace_period(self):
        """Test grace period application."""
        start_timestamp = 1000
        end_timestamp = 2000
        
        adjusted_start, adjusted_end = self.detector.apply_grace_period(
            start_timestamp, end_timestamp
        )
        
        # Grace period should extend boundaries
        assert adjusted_start < start_timestamp
        assert adjusted_end > end_timestamp
        assert adjusted_start >= 0  # Should not go negative
    
    def test_detect_boundary_insufficient_markers(self):
        """Test boundary detection with insufficient markers."""
        audio_buffer = self.create_test_audio_buffer(5000)
        markers = [TrackMarker(0, self.track_info_1)]
        
        result = self.detector.detect_boundary(audio_buffer, markers)
        assert result is None
    
    def test_detect_boundary_valid_case(self):
        """Test boundary detection with valid markers."""
        audio_buffer = self.create_test_audio_buffer(5000)
        markers = [
            TrackMarker(1000, self.track_info_1, confidence=0.9),
            TrackMarker(3000, self.track_info_2, confidence=0.8)
        ]
        
        result = self.detector.detect_boundary(audio_buffer, markers)
        
        assert result is not None
        assert result.grace_period_applied is True
        assert result.original_start == 1000
        assert result.original_end == 3000
        assert result.start_frame < 1000  # Grace period applied
        assert result.end_frame > 3000    # Grace period applied
        assert result.confidence == 0.8   # Min of marker confidences
    
    def test_detect_boundary_out_of_bounds(self):
        """Test boundary detection when markers exceed buffer length."""
        audio_buffer = self.create_test_audio_buffer(2000)  # Short buffer
        markers = [
            TrackMarker(1000, self.track_info_1),
            TrackMarker(5000, self.track_info_2)  # Beyond buffer length
        ]
        
        result = self.detector.detect_boundary(audio_buffer, markers)
        assert result is None
    
    @patch.object(TrackBoundaryDetector, '_validate_boundary_continuity')
    def test_detect_boundary_with_correction(self, mock_validate):
        """Test boundary detection with correction applied."""
        # First call returns False (needs correction), second returns True
        mock_validate.side_effect = [False, True]
        
        audio_buffer = self.create_test_audio_buffer(5000)
        markers = [
            TrackMarker(1000, self.track_info_1),
            TrackMarker(3000, self.track_info_2)
        ]
        
        with patch.object(self.detector, 'correct_boundary') as mock_correct:
            corrected_result = BoundaryResult(
                start_frame=950, end_frame=3050,
                confidence=0.9, continuity_valid=True,
                grace_period_applied=True, correction_applied=True
            )
            mock_correct.return_value = corrected_result
            
            result = self.detector.detect_boundary(audio_buffer, markers)
            
            assert result is not None
            assert result.correction_applied is True
            mock_correct.assert_called_once()
    
    def test_correct_boundary_success(self):
        """Test successful boundary correction."""
        audio_buffer = self.create_test_audio_buffer(5000)
        
        original_boundary = BoundaryResult(
            start_frame=1000, end_frame=3000,
            confidence=0.8, continuity_valid=False,
            grace_period_applied=True, correction_applied=False
        )
        
        with patch.object(self.detector, '_validate_boundary_continuity') as mock_validate:
            with patch.object(self.detector, '_calculate_boundary_confidence') as mock_confidence:
                mock_validate.return_value = True
                mock_confidence.return_value = 0.9
                
                result = self.detector.correct_boundary(audio_buffer, original_boundary)
                
                assert result is not None
                assert result.correction_applied is True
                assert result.continuity_valid is True
                assert result.confidence == 0.9
    
    def test_correct_boundary_failure(self):
        """Test boundary correction when no valid correction found."""
        audio_buffer = self.create_test_audio_buffer(5000)
        
        original_boundary = BoundaryResult(
            start_frame=1000, end_frame=3000,
            confidence=0.8, continuity_valid=False,
            grace_period_applied=True, correction_applied=False
        )
        
        with patch.object(self.detector, '_validate_boundary_continuity') as mock_validate:
            mock_validate.return_value = False  # Always fails validation
            
            result = self.detector.correct_boundary(audio_buffer, original_boundary)
            assert result is None
    
    def test_validate_frame_integrity(self):
        """Test frame integrity validation."""
        # Record some frames
        self.detector.frame_accounting.record_frames(0, 1000, 'expected')
        self.detector.frame_accounting.record_frames(0, 1000, 'processed')
        
        is_valid, message = self.detector.validate_frame_integrity()
        assert is_valid is True
        assert "1000 frames" in message
    
    def test_reset_accounting(self):
        """Test resetting frame accounting."""
        self.detector.frame_accounting.record_frames(0, 1000, 'expected')
        
        self.detector.reset_accounting()
        
        assert self.detector.frame_accounting.processed_frames == 0
        assert self.detector.frame_accounting.expected_frames == 0
    
    def test_calculate_boundary_confidence(self):
        """Test boundary confidence calculation."""
        audio_buffer = self.create_test_audio_buffer(5000)
        
        boundary = BoundaryResult(
            start_frame=1000, end_frame=3000,
            confidence=0.0, continuity_valid=True,
            grace_period_applied=True, correction_applied=False
        )
        
        confidence = self.detector._calculate_boundary_confidence(audio_buffer, boundary)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0  # Should have some confidence for non-empty segment
    
    def test_calculate_boundary_confidence_empty_segment(self):
        """Test boundary confidence calculation with empty segment."""
        audio_buffer = self.create_test_audio_buffer(5000)
        
        boundary = BoundaryResult(
            start_frame=1000, end_frame=1000,  # Empty segment
            confidence=0.0, continuity_valid=True,
            grace_period_applied=True, correction_applied=False
        )
        
        confidence = self.detector._calculate_boundary_confidence(audio_buffer, boundary)
        assert confidence == 0.0


class TestIntegration:
    """Integration tests for TrackBoundaryDetector."""
    
    def setup_method(self):
        self.detector = TrackBoundaryDetector(grace_period_ms=500)
    
    def create_realistic_audio_buffer(self) -> AudioSegment:
        """Create a more realistic audio buffer with multiple tracks."""
        # Create three different "tracks" with different characteristics
        track1 = self._create_track_audio(3000, 440, 0.7)  # 3 seconds, 440Hz, loud
        silence = AudioSegment.silent(duration=100)        # 100ms silence
        track2 = self._create_track_audio(4000, 880, 0.5)  # 4 seconds, 880Hz, medium
        
        return track1 + silence + track2
    
    def _create_track_audio(self, duration_ms: int, frequency: float, 
                           amplitude: float) -> AudioSegment:
        """Create audio segment representing a track."""
        sample_rate = 44100
        samples = int(sample_rate * duration_ms / 1000)
        
        t = np.linspace(0, duration_ms / 1000, samples, False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add some fade in/out for realism
        fade_samples = int(sample_rate * 0.1)  # 100ms fade
        if len(wave) > 2 * fade_samples:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        audio_data = (wave * 32767).astype(np.int16)
        
        return AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
    
    def test_end_to_end_boundary_detection(self):
        """Test complete boundary detection workflow."""
        audio_buffer = self.create_realistic_audio_buffer()
        
        # Create markers at track boundaries
        track_info_1 = TrackInfo(
            artist="Artist",
            title="Track 1",
            album="Album",
            art_uri="",
            id="spotify:track:1",
            track_number=1,
            position=0,
            duration_ms=3000
        )
        track_info_2 = TrackInfo(
            artist="Artist",
            title="Track 2", 
            album="Album",
            art_uri="",
            id="spotify:track:2",
            track_number=2,
            position=0,
            duration_ms=4000
        )
        
        markers = [
            TrackMarker(0, track_info_1, confidence=0.9),
            TrackMarker(3100, track_info_2, confidence=0.8)  # After silence
        ]
        
        result = self.detector.detect_boundary(audio_buffer, markers)
        
        assert result is not None
        assert result.grace_period_applied is True
        assert result.start_frame >= 0
        assert result.end_frame <= len(audio_buffer)
        assert result.confidence > 0
        
        # Validate frame integrity - we only recorded expected frames, so should show missing processed frames
        is_valid, message = self.detector.validate_frame_integrity()
        assert not is_valid and "missing" in message  # Should show frames missing since we didn't process them