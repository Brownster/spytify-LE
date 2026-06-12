"""
End-to-end tests for segment processing and export failure handling.

Covers the single recovery policy that survived Pass 3: segment preparation
failures are reported once and skipped, exports retry only transient I/O errors.
"""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil
from pydub import AudioSegment

from spotify_splitter.segmenter import SegmentManager, TrackMarker
from spotify_splitter.mpris import TrackInfo


class TestSegmentFailureHandling:
    """Test segment-processing and export failure handling."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_track_info(self):
        """Create mock track info for testing."""
        return TrackInfo(
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            art_uri="https://example.com/art.jpg",
            id="spotify:track:test123",
            track_number=1,
            position=0,
            duration_ms=3000  # 3 seconds to match test audio
        )
    
    @pytest.fixture
    def audio_queue(self):
        """Create audio queue for testing."""
        return queue.Queue(maxsize=100)
    
    @pytest.fixture
    def event_queue(self):
        """Create event queue for testing."""
        return queue.Queue()
    
    @pytest.fixture
    def ui_callback_mock(self):
        """Create mock UI callback for testing."""
        return Mock()
    
    @pytest.fixture
    def segment_manager(self, temp_output_dir, audio_queue, event_queue,
                       ui_callback_mock):
        """Create SegmentManager for testing."""
        return SegmentManager(
            samplerate=44100,
            output_dir=temp_output_dir,
            fmt="mp3",
            audio_queue=audio_queue,
            event_queue=event_queue,
            ui_callback=ui_callback_mock,
        )
    
    def test_segment_processing_error_is_reported_without_retry(
        self, segment_manager, mock_track_info, ui_callback_mock
    ):
        """Test segment preparation failures are reported once and skipped."""
        # Create test audio data
        test_audio = AudioSegment.silent(duration=3000)  # 3 seconds
        segment_manager.continuous_buffer = test_audio
        
        # Create track markers
        start_marker = TrackMarker(0, mock_track_info)
        end_marker = TrackMarker(3000, mock_track_info)
        segment_manager.track_markers = [start_marker, end_marker]
        
        call_count = 0
        def mock_detect_boundary(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Simulated boundary detection error")
        
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', side_effect=mock_detect_boundary):
            with patch.object(segment_manager, '_export') as mock_export:
                segment_manager.process_segments()
                segment_manager.wait_for_exports()
                
                assert call_count == 1
                mock_export.assert_not_called()
                assert len(segment_manager.track_markers) == 1
                assert segment_manager.processing_errors == 1
                assert segment_manager.recovery_attempts == 0
                
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                error_calls = [call for call in ui_callback_calls if call[0] == "processing_error"]
                failure_calls = [call for call in ui_callback_calls if call[0] == "processing_failure"]
                recovery_calls = [call for call in ui_callback_calls if call[0] == "recovery_success"]
                
                assert len(error_calls) == 1
                assert len(failure_calls) == 1
                assert len(recovery_calls) == 0
    
    def test_segment_processing_failure_does_not_degrade(
        self, segment_manager, mock_track_info, ui_callback_mock
    ):
        """Test segment failures do not use the removed degraded export path."""
        # Create test audio data
        test_audio = AudioSegment.silent(duration=3000)
        segment_manager.continuous_buffer = test_audio
        
        # Create track markers
        start_marker = TrackMarker(0, mock_track_info)
        end_marker = TrackMarker(3000, mock_track_info)
        segment_manager.track_markers = [start_marker, end_marker]
        
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', 
                         side_effect=MemoryError("Out of memory during boundary detection")):
            segment_manager.process_segments()

        ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
        degraded_calls = [call for call in ui_callback_calls if call[0] == "degraded_export"]
        failure_calls = [call for call in ui_callback_calls if call[0] == "processing_failure"]

        assert not hasattr(segment_manager, "_export_with_minimal_processing")
        assert not hasattr(segment_manager, "degraded_exports")
        assert len(degraded_calls) == 0
        assert len(failure_calls) == 1
    
    def test_export_error_recovery(self, segment_manager, mock_track_info, ui_callback_mock):
        """Test error recovery during export operations."""
        test_audio = AudioSegment.silent(duration=3000)
        
        # Mock export to fail first time, succeed second time
        call_count = 0
        def mock_export_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise IOError("Simulated export error")
            return None  # Success
        
        with patch.object(segment_manager, '_export', side_effect=mock_export_side_effect):
            # Test export with error handling
            success = segment_manager._export_with_error_handling(test_audio, mock_track_info)
            
            assert success, "Export should succeed after transient retry"
            assert call_count == 2, "Export should be retried once"
            assert segment_manager.export_errors == 1
            assert segment_manager.recovery_attempts == 1
            assert segment_manager.successful_recoveries == 1
            
            # Verify UI callbacks for export error
            ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
            export_error_calls = [call for call in ui_callback_calls if call[0] == "export_error"]
            
            assert len(export_error_calls) >= 1, "Should have called UI with export error"
    
    def test_export_non_transient_failure_does_not_retry(
        self, segment_manager, mock_track_info, ui_callback_mock
    ):
        """Test non-transient export failures fail once and notify."""
        test_audio = AudioSegment.silent(duration=3000)
        call_count = 0

        def mock_export_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent export error")

        with patch.object(segment_manager, '_export', side_effect=mock_export_side_effect):
            success = segment_manager._export_with_error_handling(test_audio, mock_track_info)

        assert not success
        assert call_count == 1
        assert segment_manager.export_errors == 1
        assert segment_manager.recovery_attempts == 0

        ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
        export_error_calls = [call for call in ui_callback_calls if call[0] == "export_error"]
        failure_calls = [call for call in ui_callback_calls if call[0] == "processing_failure"]

        assert len(export_error_calls) == 1
        assert export_error_calls[0][1]["will_retry"] is False
        assert len(failure_calls) == 1
    
    def test_end_to_end_error_recovery_scenario(self, segment_manager, mock_track_info, 
                                               ui_callback_mock, audio_queue, event_queue):
        """Test complete end-to-end error recovery scenario."""
        # Simulate a complete recording session with various errors
        
        # Add some test audio data
        test_audio = AudioSegment.silent(duration=5000)  # 5 seconds
        segment_manager.continuous_buffer = test_audio
        
        # Create multiple track markers
        track1 = mock_track_info
        track2 = TrackInfo(
            artist="Test Artist 2",
            title="Test Track 2",
            album="Test Album 2",
            art_uri="https://example.com/art2.jpg",
            id="spotify:track:test456",
            track_number=2,
            position=0,
            duration_ms=2500
        )
        
        markers = [
            TrackMarker(0, track1),
            TrackMarker(2500, track2),
            TrackMarker(5000, track1)  # End marker
        ]
        segment_manager.track_markers = markers
        
        # Mock various failure scenarios
        boundary_call_count = 0
        export_call_count = 0
        
        def mock_boundary_detect(*args, **kwargs):
            nonlocal boundary_call_count
            boundary_call_count += 1
            if boundary_call_count == 1:
                raise ValueError("Boundary detection failed")
            from spotify_splitter.track_boundary_detector import BoundaryResult
            return BoundaryResult(
                start_frame=0,
                end_frame=2500,
                confidence=0.8,
                continuity_valid=True,
                grace_period_applied=True,
                correction_applied=False
            )
        
        def mock_export(*args, **kwargs):
            nonlocal export_call_count
            export_call_count += 1
            if export_call_count == 1:
                raise IOError("Export failed")
            return None
        
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', 
                         side_effect=mock_boundary_detect):
            with patch.object(segment_manager, '_export', side_effect=mock_export):
                # Process first segment: boundary failure is reported and skipped.
                segment_manager.process_segments()

                # Process second segment: export fails transiently and retries on worker.
                if len(segment_manager.track_markers) >= 2:
                    segment_manager.process_segments()

                segment_manager.wait_for_exports()
        
        assert segment_manager.processing_errors == 1
        assert segment_manager.export_errors == 1
        assert segment_manager.recovery_attempts == 1
        assert segment_manager.successful_recoveries == 1

        # Verify UI was notified of various events
        ui_callback_calls = [call.args[0] for call in ui_callback_mock.call_args_list]
        
        expected_events = ["processing_error", "processing_failure", "export_error"]
        found_events = [event for event in expected_events if event in ui_callback_calls]
        
        assert found_events == expected_events
    
    def test_stat_counters_are_thread_safe_increments(self, segment_manager):
        """The surviving diagnostic counters increment under the stats lock."""
        segment_manager._increment_stat("processing_errors", 2)
        segment_manager._increment_stat("export_errors")
        segment_manager._increment_stat("recovery_attempts", 3)
        segment_manager._increment_stat("successful_recoveries")

        assert segment_manager.processing_errors == 2
        assert segment_manager.export_errors == 1
        assert segment_manager.recovery_attempts == 3
        assert segment_manager.successful_recoveries == 1

    def test_concurrent_error_handling(self, segment_manager, mock_track_info, ui_callback_mock):
        """Test error handling under concurrent processing scenarios."""
        # Create multiple threads that process segments simultaneously
        test_audio = AudioSegment.silent(duration=10000)  # 10 seconds
        segment_manager.continuous_buffer = test_audio
        
        # Create multiple track markers
        markers = []
        for i in range(5):
            track = TrackInfo(
                artist="Test Artist",
                title=f"Test Track {i}",
                album="Test Album",
                art_uri="https://example.com/art.jpg",
                id=f"spotify:track:test{i}",
                track_number=i+1,
                position=0,
                duration_ms=2000
            )
            markers.append(TrackMarker(i * 2000, track))
        
        segment_manager.track_markers = markers
        
        # Mock processing to occasionally fail
        call_count = 0
        def mock_process_internal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every third call fails
                raise RuntimeError(f"Simulated error {call_count}")
            return True
        
        with patch.object(segment_manager, '_process_segment_internal', side_effect=mock_process_internal):
            # Process multiple segments
            threads = []
            for _ in range(3):
                if len(segment_manager.track_markers) >= 2:
                    thread = threading.Thread(target=segment_manager.process_segments)
                    threads.append(thread)
                    thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5.0)
        
        # Verify error recovery handled concurrent scenarios
        assert segment_manager.processing_errors > 0 or segment_manager.recovery_attempts > 0
        
        # Verify no race conditions in UI callbacks
        ui_callback_calls = ui_callback_mock.call_args_list
        assert len(ui_callback_calls) > 0, "Should have UI callback calls from concurrent processing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
