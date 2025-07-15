"""
Tests for enhanced SegmentManager with improved boundary handling.

This module tests the integration of TrackBoundaryDetector into SegmentManager,
including grace period handling, audio continuity validation, and boundary correction.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import queue
import numpy as np
from pathlib import Path
from pydub import AudioSegment

from spotify_splitter.segmenter import SegmentManager, TrackMarker
from spotify_splitter.mpris import TrackInfo
from spotify_splitter.track_boundary_detector import BoundaryResult, TrackBoundaryDetector


class TestEnhancedSegmentProcessing(unittest.TestCase):
    """Test enhanced segment processing with boundary detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audio_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.ui_callback = Mock()
        
        # Create test audio segments
        self.test_audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
        self.test_audio_short = AudioSegment.silent(duration=1000)  # 1 second
        
        # Create test track info
        self.track_info = TrackInfo(
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            art_uri="http://example.com/art.jpg",
            id="spotify:track:test123",
            track_number=1,
            position=0,
            duration_ms=5000
        )
        
        self.segment_manager = SegmentManager(
            samplerate=44100,
            output_dir=Path("/tmp/test_music"),
            audio_queue=self.audio_queue,
            event_queue=self.event_queue,
            ui_callback=self.ui_callback,
            grace_period_ms=500,
            max_correction_ms=2000
        )
    
    def test_boundary_detector_initialization(self):
        """Test that TrackBoundaryDetector is properly initialized."""
        self.assertIsInstance(self.segment_manager.boundary_detector, TrackBoundaryDetector)
        self.assertEqual(self.segment_manager.boundary_detector.grace_period_ms, 500)
        self.assertEqual(self.segment_manager.boundary_detector.max_correction_ms, 2000)
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_process_segments_with_grace_period(self, mock_export):
        """Test segment processing with grace period applied."""
        # Set up test data
        self.segment_manager.continuous_buffer = self.test_audio
        self.segment_manager.track_markers = [
            TrackMarker(0, self.track_info),
            TrackMarker(3000, self.track_info)  # 3 second mark
        ]
        
        # Mock boundary detection to return result with grace period
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=3500,  # Extended by grace period
            confidence=0.9,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=False
        )
        
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=boundary_result):
            self.segment_manager.process_segments()
        
        # Verify export was called with corrected audio
        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args
        exported_audio, track_info = args
        
        self.assertEqual(len(exported_audio), 3500)  # Should use grace period boundary
        self.assertEqual(track_info, self.track_info)
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_process_segments_with_boundary_correction(self, mock_export):
        """Test segment processing with boundary correction applied."""
        # Set up test data
        self.segment_manager.continuous_buffer = self.test_audio
        self.segment_manager.track_markers = [
            TrackMarker(0, self.track_info),
            TrackMarker(3000, self.track_info)
        ]
        
        # Mock boundary detection to return result with correction
        boundary_result = BoundaryResult(
            start_frame=50,  # Corrected start
            end_frame=3050,  # Corrected end
            confidence=0.8,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=True,
            original_start=0,
            original_end=3000
        )
        
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=boundary_result):
            self.segment_manager.process_segments()
        
        # Verify export was called with corrected boundaries
        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args
        exported_audio, track_info = args
        
        self.assertEqual(len(exported_audio), 3000)  # 3050 - 50 = 3000ms
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_process_segments_continuity_validation_failed(self, mock_export):
        """Test segment processing when continuity validation fails."""
        # Set up test data
        self.segment_manager.continuous_buffer = self.test_audio
        self.segment_manager.track_markers = [
            TrackMarker(0, self.track_info),
            TrackMarker(3000, self.track_info)
        ]
        
        # Mock boundary detection to return result with failed continuity
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=3000,
            confidence=0.5,
            continuity_valid=False,  # Continuity validation failed
            grace_period_applied=True,
            correction_applied=False
        )
        
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=boundary_result):
            with patch('spotify_splitter.segmenter.logger') as mock_logger:
                self.segment_manager.process_segments()
                
                # Verify warning was logged
                mock_logger.warning.assert_called_with(
                    "Audio continuity validation failed - potential audio quality issues"
                )
        
        # Export should still proceed despite continuity warning
        mock_export.assert_called_once()
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_process_segments_boundary_detection_failed(self, mock_export):
        """Test fallback to original processing when boundary detection fails."""
        # Set up test data
        self.segment_manager.continuous_buffer = self.test_audio
        self.segment_manager.track_markers = [
            TrackMarker(0, self.track_info),
            TrackMarker(3000, self.track_info)
        ]
        
        # Mock boundary detection to return None (failure)
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=None):
            with patch('spotify_splitter.segmenter.logger') as mock_logger:
                self.segment_manager.process_segments()
                
                # Verify warning was logged
                mock_logger.warning.assert_called_with(
                    "Boundary detection failed for track: %s", self.track_info.title
                )
        
        # Export should still proceed with original timestamp-based cutting
        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args
        exported_audio, track_info = args
        
        self.assertEqual(len(exported_audio), 3000)  # Original duration
    
    def test_apply_final_boundary_correction_with_fade(self):
        """Test final boundary correction with fade-in/fade-out."""
        # Create test audio chunk
        audio_chunk = AudioSegment.silent(duration=2000)  # 2 seconds
        
        # Create boundary result with grace period applied
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=2000,
            confidence=0.9,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=False
        )
        
        # Apply final correction
        corrected_chunk = self.segment_manager._apply_final_boundary_correction(
            audio_chunk, self.track_info, boundary_result
        )
        
        # Verify fade was applied (duration should remain the same)
        self.assertEqual(len(corrected_chunk), len(audio_chunk))
        self.assertIsInstance(corrected_chunk, AudioSegment)
    
    def test_apply_final_boundary_correction_no_boundary_result(self):
        """Test final boundary correction when no boundary result is provided."""
        audio_chunk = AudioSegment.silent(duration=2000)
        
        # Apply correction with no boundary result
        corrected_chunk = self.segment_manager._apply_final_boundary_correction(
            audio_chunk, self.track_info, None
        )
        
        # Should return original chunk unchanged
        self.assertEqual(corrected_chunk, audio_chunk)
    
    def test_apply_final_boundary_correction_empty_chunk(self):
        """Test final boundary correction with empty audio chunk."""
        empty_chunk = AudioSegment.empty()
        
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=0,
            confidence=0.9,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=False
        )
        
        # Apply correction to empty chunk
        corrected_chunk = self.segment_manager._apply_final_boundary_correction(
            empty_chunk, self.track_info, boundary_result
        )
        
        # Should return original empty chunk
        self.assertEqual(corrected_chunk, empty_chunk)
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_enhanced_duration_tolerance(self, mock_export):
        """Test enhanced duration tolerance when grace period or correction is applied."""
        # Set up test data with duration mismatch
        self.segment_manager.continuous_buffer = self.test_audio
        track_info_long = TrackInfo(
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            art_uri="http://example.com/art.jpg",
            id="spotify:track:test123",
            track_number=1,
            position=0,
            duration_ms=8000  # Expected 8 seconds, but we have 5
        )
        
        self.segment_manager.track_markers = [
            TrackMarker(0, track_info_long),
            TrackMarker(5000, track_info_long)
        ]
        
        # Mock boundary detection with grace period (should increase tolerance)
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=5000,
            confidence=0.9,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=False
        )
        
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=boundary_result):
            self.segment_manager.process_segments()
        
        # With enhanced tolerance (6000ms), the 3000ms difference should be acceptable
        mock_export.assert_called_once()
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_frame_integrity_validation(self, mock_export):
        """Test frame integrity validation when duration mismatch is too large."""
        # Set up test data with large duration mismatch
        self.segment_manager.continuous_buffer = self.test_audio
        track_info_very_long = TrackInfo(
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            art_uri="http://example.com/art.jpg",
            id="spotify:track:test123",
            track_number=1,
            position=0,
            duration_ms=15000  # Expected 15 seconds, but we have 5
        )
        
        self.segment_manager.track_markers = [
            TrackMarker(0, track_info_very_long),
            TrackMarker(5000, track_info_very_long)
        ]
        
        # Mock boundary detection
        boundary_result = BoundaryResult(
            start_frame=0,
            end_frame=5000,
            confidence=0.9,
            continuity_valid=True,
            grace_period_applied=True,
            correction_applied=False
        )
        
        # Mock frame integrity validation to return failure
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', return_value=boundary_result):
            with patch.object(self.segment_manager.boundary_detector, 'validate_frame_integrity', 
                            return_value=(False, "Frame loss detected: 100 frames missing")):
                with patch('spotify_splitter.segmenter.logger') as mock_logger:
                    self.segment_manager.process_segments()
                    
                    # Verify frame integrity warning was logged
                    mock_logger.warning.assert_any_call("Frame integrity issue detected: %s", 
                                                      "Frame loss detected: 100 frames missing")
        
        # Export should not be called due to large duration mismatch
        mock_export.assert_not_called()
    
    def test_enhanced_track_marker_conversion(self):
        """Test conversion from TrackMarker to EnhancedTrackMarker."""
        # Set up test data
        self.segment_manager.continuous_buffer = self.test_audio
        self.segment_manager.track_markers = [
            TrackMarker(0, self.track_info),
            TrackMarker(3000, self.track_info)
        ]
        
        # Mock boundary detection to capture the enhanced markers
        captured_markers = None
        
        def capture_markers(audio_buffer, markers):
            nonlocal captured_markers
            captured_markers = markers
            return None  # Return None to trigger fallback
        
        with patch.object(self.segment_manager.boundary_detector, 'detect_boundary', side_effect=capture_markers):
            with patch('spotify_splitter.segmenter.SegmentManager._export'):
                self.segment_manager.process_segments()
        
        # Verify enhanced markers were created correctly
        self.assertIsNotNone(captured_markers)
        self.assertEqual(len(captured_markers), 2)
        
        # Check first marker
        self.assertEqual(captured_markers[0].timestamp, 0)
        self.assertEqual(captured_markers[0].track_info, self.track_info)
        self.assertEqual(captured_markers[0].confidence, 1.0)
        
        # Check second marker
        self.assertEqual(captured_markers[1].timestamp, 3000)
        self.assertEqual(captured_markers[1].track_info, self.track_info)
        self.assertEqual(captured_markers[1].confidence, 1.0)


class TestSegmentManagerIntegration(unittest.TestCase):
    """Integration tests for enhanced SegmentManager functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.audio_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.ui_callback = Mock()
        
        self.segment_manager = SegmentManager(
            samplerate=44100,
            output_dir=Path("/tmp/test_music"),
            audio_queue=self.audio_queue,
            event_queue=self.event_queue,
            ui_callback=self.ui_callback
        )
    
    def test_flush_cache_resets_boundary_detector(self):
        """Test that cache flush properly resets boundary detector state."""
        # Add some test data
        self.segment_manager.track_markers = [TrackMarker(0, Mock())]
        self.segment_manager.continuous_buffer = AudioSegment.silent(duration=1000)
        
        # Simulate some boundary detector state
        self.segment_manager.boundary_detector.frame_accounting.processed_frames = 100
        
        # Flush cache
        self.segment_manager.flush_cache()
        
        # Verify state is reset
        self.assertEqual(len(self.segment_manager.track_markers), 0)
        self.assertEqual(len(self.segment_manager.continuous_buffer), 0)
        self.assertFalse(self.segment_manager.first_track_seen)
    
    @patch('spotify_splitter.segmenter.SegmentManager._export')
    def test_end_to_end_enhanced_processing(self, mock_export):
        """Test complete end-to-end enhanced segment processing."""
        # Create realistic test audio data
        test_audio = AudioSegment.silent(duration=10000)  # 10 seconds
        
        # Create test track info
        track_info = TrackInfo(
            artist="Test Artist",
            title="Integration Test Track",
            album="Test Album",
            art_uri="http://example.com/art.jpg",
            id="spotify:track:test123",
            track_number=1,
            position=0,
            duration_ms=4500  # Slightly different from actual
        )
        
        # Set up segment manager state with 3 markers to test complete processing
        self.segment_manager.continuous_buffer = test_audio
        self.segment_manager.first_track_seen = True
        self.segment_manager.track_markers = [
            TrackMarker(1000, track_info),  # Start at 1 second
            TrackMarker(5000, track_info),  # End at 5 seconds
            TrackMarker(8000, track_info)   # Third marker for complete cleanup
        ]
        
        # Process segments
        self.segment_manager.process_segments()
        
        # Verify export was called
        mock_export.assert_called_once()
        
        # Verify buffer cleanup (accounting for grace period adjustments)
        # The exact remaining length may vary due to grace period and boundary detection
        self.assertLessEqual(len(self.segment_manager.continuous_buffer), 5000)  # Should be <= original remaining
        self.assertGreaterEqual(len(self.segment_manager.continuous_buffer), 2000)  # Should be reasonable
        self.assertEqual(len(self.segment_manager.track_markers), 2)  # Two markers remaining after processing first segment


if __name__ == '__main__':
    unittest.main()