"""
End-to-end tests for comprehensive error handling and recovery scenarios.

This module tests the integration of ErrorRecoveryManager with AudioStream and SegmentManager,
including progressive error escalation, graceful degradation, and user notifications.
"""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
from pydub import AudioSegment
import numpy as np

from spotify_splitter.error_recovery import ErrorRecoveryManager, RecoveryAction, ErrorSeverity
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.segmenter import SegmentManager, TrackMarker
from spotify_splitter.mpris import TrackInfo
from spotify_splitter.buffer_management import AdaptiveBufferManager


class TestErrorRecoveryIntegration:
    """Test comprehensive error recovery integration."""
    
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
    def error_recovery_manager(self):
        """Create error recovery manager for testing."""
        return ErrorRecoveryManager(
            max_retries=3,
            backoff_factor=1.2,
            max_backoff=5.0
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
                       ui_callback_mock, error_recovery_manager):
        """Create SegmentManager with error recovery for testing."""
        return SegmentManager(
            samplerate=44100,
            output_dir=temp_output_dir,
            fmt="mp3",
            audio_queue=audio_queue,
            event_queue=event_queue,
            ui_callback=ui_callback_mock,
            error_recovery=error_recovery_manager,
            enable_error_recovery=True,
            max_processing_retries=3,
            enable_graceful_degradation=True
        )
    
    def test_segment_processing_error_recovery(self, segment_manager, mock_track_info, ui_callback_mock):
        """Test error recovery during segment processing."""
        # Create test audio data
        test_audio = AudioSegment.silent(duration=3000)  # 3 seconds
        segment_manager.continuous_buffer = test_audio
        
        # Create track markers
        start_marker = TrackMarker(0, mock_track_info)
        end_marker = TrackMarker(3000, mock_track_info)
        segment_manager.track_markers = [start_marker, end_marker]
        
        # Mock boundary detector to raise an error on first call, succeed on second
        call_count = 0
        def mock_detect_boundary(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated boundary detection error")
            # Return successful boundary result on retry
            from spotify_splitter.track_boundary_detector import BoundaryResult
            return BoundaryResult(
                start_frame=0,
                end_frame=3000,
                confidence=0.9,
                continuity_valid=True,
                grace_period_applied=False,
                correction_applied=False
            )
        
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', side_effect=mock_detect_boundary):
            with patch.object(segment_manager, '_export') as mock_export:
                # Process segments - should recover from first error
                segment_manager.process_segments()
                
                # Verify recovery was attempted and succeeded
                assert call_count == 2  # First call failed, second succeeded
                mock_export.assert_called_once()
                
                # Verify UI callbacks for error and recovery
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                error_calls = [call for call in ui_callback_calls if call[0] == "processing_error"]
                recovery_calls = [call for call in ui_callback_calls if call[0] == "recovery_success"]
                
                assert len(error_calls) >= 1, "Should have called UI with processing error"
                assert len(recovery_calls) >= 1, "Should have called UI with recovery success"
    
    def test_segment_processing_graceful_degradation(self, segment_manager, mock_track_info, ui_callback_mock):
        """Test graceful degradation when processing repeatedly fails."""
        # Create test audio data
        test_audio = AudioSegment.silent(duration=3000)
        segment_manager.continuous_buffer = test_audio
        
        # Create track markers
        start_marker = TrackMarker(0, mock_track_info)
        end_marker = TrackMarker(3000, mock_track_info)
        segment_manager.track_markers = [start_marker, end_marker]
        
        # Mock boundary detector to always fail with a memory error that should trigger graceful degradation
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', 
                         side_effect=MemoryError("Out of memory during boundary detection")):
            with patch.object(segment_manager, '_export_with_minimal_processing', return_value=True) as mock_minimal_export:
                # Process segments - should fall back to graceful degradation
                segment_manager.process_segments()
                
                # Verify graceful degradation was used
                mock_minimal_export.assert_called_once()
                
                # Verify UI callback for degraded export
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                degraded_calls = [call for call in ui_callback_calls if call[0] == "degraded_export"]
                
                assert len(degraded_calls) >= 1, "Should have called UI with degraded export notification"
                assert segment_manager.degraded_exports == 1
    
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
            
            assert success, "Export should succeed after recovery"
            assert call_count == 2, "Export should be retried once"
            
            # Verify UI callbacks for export error
            ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
            export_error_calls = [call for call in ui_callback_calls if call[0] == "export_error"]
            
            assert len(export_error_calls) >= 1, "Should have called UI with export error"
    
    def test_progressive_error_escalation(self, segment_manager, mock_track_info, ui_callback_mock):
        """Test progressive error escalation when all recovery attempts fail."""
        # Create test audio data
        test_audio = AudioSegment.silent(duration=3000)
        segment_manager.continuous_buffer = test_audio
        
        # Create track markers
        start_marker = TrackMarker(0, mock_track_info)
        end_marker = TrackMarker(3000, mock_track_info)
        segment_manager.track_markers = [start_marker, end_marker]
        
        # Mock all recovery attempts to fail
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', 
                         side_effect=RuntimeError("Persistent error")):
            with patch.object(segment_manager, '_export_with_minimal_processing', return_value=False):
                # Process segments - should escalate after all attempts fail
                segment_manager.process_segments()
                
                # Verify processing failure was handled
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                failure_calls = [call for call in ui_callback_calls if call[0] == "processing_failure"]
                
                assert len(failure_calls) >= 1, "Should have called UI with processing failure"
                assert segment_manager.processing_errors > 0
    
    def test_audio_stream_error_recovery(self, error_recovery_manager, ui_callback_mock):
        """Test error recovery in EnhancedAudioStream."""
        buffer_manager = AdaptiveBufferManager()
        
        # Create enhanced audio stream with mocked sounddevice
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_sd.InputStream.return_value = mock_stream
            
            stream = EnhancedAudioStream(
                monitor_name="test_monitor",
                buffer_manager=buffer_manager,
                error_recovery=error_recovery_manager,
                ui_callback=ui_callback_mock,
                enable_error_recovery=True
            )
            
            # Test stream error handling with an OSError that should trigger reconnection
            test_error = OSError("PortAudioError: Device not found")
            
            with patch.object(stream, '_attempt_reconnection', return_value=True) as mock_reconnect:
                success = stream.handle_stream_error(test_error)
                
                assert success, "Stream error recovery should succeed"
                mock_reconnect.assert_called_once()
                
                # Verify UI callbacks for stream error and recovery
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                stream_error_calls = [call for call in ui_callback_calls if call[0] == "stream_error"]
                recovery_result_calls = [call for call in ui_callback_calls if call[0] == "stream_recovery_result"]
                
                assert len(stream_error_calls) >= 1, "Should have called UI with stream error"
                assert len(recovery_result_calls) >= 1, "Should have called UI with recovery result"
    
    def test_audio_stream_graceful_degradation(self, error_recovery_manager, ui_callback_mock):
        """Test graceful degradation in EnhancedAudioStream."""
        buffer_manager = AdaptiveBufferManager()
        
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_sd.InputStream.return_value = mock_stream
            
            stream = EnhancedAudioStream(
                monitor_name="test_monitor",
                buffer_manager=buffer_manager,
                error_recovery=error_recovery_manager,
                ui_callback=ui_callback_mock,
                enable_error_recovery=True
            )
            
            # Test graceful degradation for memory error
            memory_error = MemoryError("Simulated memory error")
            
            with patch.object(stream, 'stream') as mock_stream_obj:
                success = stream._attempt_graceful_degradation(memory_error)
                
                assert success, "Graceful degradation should succeed"
                
                # Verify UI callback for degraded mode
                ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
                degraded_mode_calls = [call for call in ui_callback_calls if call[0] == "degraded_mode"]
                
                assert len(degraded_mode_calls) >= 1, "Should have called UI with degraded mode notification"
    
    def test_error_escalation_with_recommendations(self, error_recovery_manager, ui_callback_mock):
        """Test error escalation with user-actionable recommendations."""
        buffer_manager = AdaptiveBufferManager()
        
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_sd.InputStream.return_value = mock_stream
            
            stream = EnhancedAudioStream(
                monitor_name="test_monitor",
                buffer_manager=buffer_manager,
                error_recovery=error_recovery_manager,
                ui_callback=ui_callback_mock,
                enable_error_recovery=True
            )
            
            # Test critical error escalation
            critical_error = Exception("PortAudioError: Device not found")
            
            stream._escalate_stream_error(critical_error, "test_context")
            
            # Verify UI callback for critical error with recommendations
            ui_callback_calls = [call.args for call in ui_callback_mock.call_args_list]
            critical_error_calls = [call for call in ui_callback_calls if call[0] == "critical_error"]
            
            assert len(critical_error_calls) >= 1, "Should have called UI with critical error"
            
            # Check that recommendations were provided
            critical_call_data = critical_error_calls[0][1]
            assert "recommendations" in critical_call_data
            assert len(critical_call_data["recommendations"]) > 0
    
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
            duration_ms=240000
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
                # First track fails boundary detection
                raise ValueError("Boundary detection failed")
            # Second track succeeds
            from spotify_splitter.track_boundary_detector import BoundaryResult
            return BoundaryResult(
                start_frame=2500,
                end_frame=5000,
                confidence=0.8,
                continuity_valid=True,
                grace_period_applied=True,
                correction_applied=False
            )
        
        def mock_export(*args, **kwargs):
            nonlocal export_call_count
            export_call_count += 1
            if export_call_count == 1:
                # First export attempt fails
                raise IOError("Export failed")
            # Subsequent exports succeed
            return None
        
        with patch.object(segment_manager.boundary_detector, 'detect_boundary', 
                         side_effect=mock_boundary_detect):
            with patch.object(segment_manager, '_export', side_effect=mock_export):
                with patch.object(segment_manager, '_export_with_minimal_processing', return_value=True):
                    # Process first segment (should use graceful degradation)
                    segment_manager.process_segments()
                    
                    # Process second segment (should recover from export error)
                    if len(segment_manager.track_markers) >= 2:
                        segment_manager.process_segments()
        
        # Verify comprehensive error handling occurred
        assert segment_manager.processing_errors > 0 or segment_manager.export_errors > 0
        assert segment_manager.recovery_attempts > 0
        
        # Verify UI was notified of various events
        ui_callback_calls = [call.args[0] for call in ui_callback_mock.call_args_list]
        
        # Should have various error recovery events
        expected_events = ["processing_error", "export_error", "degraded_export", "recovery_success"]
        found_events = [event for event in expected_events if event in ui_callback_calls]
        
        assert len(found_events) > 0, f"Should have found error recovery events, got: {ui_callback_calls}"
    
    def test_error_statistics_and_diagnostics(self, segment_manager, error_recovery_manager):
        """Test error statistics collection and diagnostic reporting."""
        # Simulate some errors
        test_errors = [
            ValueError("Test error 1"),
            IOError("Test error 2"),
            RuntimeError("Test error 3")
        ]
        
        for error in test_errors:
            error_recovery_manager.handle_error(error, "test_context")
        
        # Get statistics
        stats = segment_manager.get_error_statistics()
        
        assert "processing_errors" in stats
        assert "export_errors" in stats
        assert "recovery_attempts" in stats
        assert "error_recovery_enabled" in stats
        assert stats["error_recovery_enabled"] is True
        
        # Test diagnostic report generation
        report = segment_manager.generate_error_report()
        
        assert "SegmentManager Error Report" in report
        assert "Error Recovery Manager Diagnostics" in report
        assert len(report) > 100  # Should be a substantial report
    
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


class TestErrorRecoveryManagerStandalone:
    """Test ErrorRecoveryManager functionality in isolation."""
    
    @pytest.fixture
    def error_recovery(self):
        """Create error recovery manager for testing."""
        return ErrorRecoveryManager(max_retries=3, backoff_factor=1.5)
    
    def test_error_classification(self, error_recovery):
        """Test error severity classification."""
        # Test critical errors
        critical_error = MemoryError("Out of memory")
        action = error_recovery.handle_error(critical_error, "test")
        assert action in [RecoveryAction.ESCALATE, RecoveryAction.GRACEFUL_DEGRADE]
        
        # Test high severity errors - use a more specific error type
        high_error = OSError("PortAudioError: Device unavailable")
        action = error_recovery.handle_error(high_error, "test")
        assert action == RecoveryAction.RECONNECT
        
        # Test medium severity errors
        medium_error = IOError("File not found")
        action = error_recovery.handle_error(medium_error, "test")
        assert action in [RecoveryAction.RETRY, RecoveryAction.RECONNECT]
    
    def test_recovery_attempt_with_backoff(self, error_recovery):
        """Test recovery attempts with exponential backoff."""
        # Test multiple recovery attempts to see backoff behavior
        call_times = []
        
        def mock_recovery_func():
            call_times.append(time.time())
            return True  # Always succeed
        
        # First attempt - should have no delay
        start_time = time.time()
        success1 = error_recovery.attempt_recovery(
            RecoveryAction.RETRY,
            mock_recovery_func,
            "test_recovery_1"
        )
        
        # Second attempt - should have backoff delay
        success2 = error_recovery.attempt_recovery(
            RecoveryAction.RETRY,
            mock_recovery_func,
            "test_recovery_2"
        )
        
        assert success1, "First recovery should succeed"
        assert success2, "Second recovery should succeed"
        assert len(call_times) == 2, "Should make 2 recovery function calls"
        
        # Verify backoff delay was applied between attempts
        if len(call_times) >= 2:
            delay = call_times[1] - call_times[0]
            # The delay should be at least the backoff time (1.5^1 = 1.5 seconds)
            assert delay >= 1.0, f"Should have backoff delay between attempts, got {delay:.2f}s"
    
    def test_error_frequency_escalation(self, error_recovery):
        """Test escalation when same error occurs frequently."""
        # Generate multiple same errors quickly
        for _ in range(5):
            action = error_recovery.handle_error(ValueError("Frequent error"), "test")
        
        # Should escalate after frequent occurrences
        final_action = error_recovery.handle_error(ValueError("Frequent error"), "test")
        assert final_action == RecoveryAction.ESCALATE
    
    def test_diagnostics_generation(self, error_recovery):
        """Test comprehensive diagnostics generation."""
        # Generate various errors
        errors = [
            ValueError("Error 1"),
            IOError("Error 2"),
            RuntimeError("Error 3"),
            ValueError("Error 1"),  # Repeat for frequency testing
        ]
        
        for error in errors:
            error_recovery.handle_error(error, "test")
        
        # Generate diagnostics
        diagnostics = error_recovery.get_diagnostics()
        
        assert diagnostics.total_errors == 4
        assert len(diagnostics.most_common_errors) > 0
        assert diagnostics.most_common_errors[0][0] == "ValueError"  # Most common
        assert diagnostics.most_common_errors[0][1] == 2  # Occurred twice
        assert len(diagnostics.recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])