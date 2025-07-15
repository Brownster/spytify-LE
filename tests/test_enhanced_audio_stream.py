"""
Integration tests for EnhancedAudioStream with adaptive capabilities.

Tests the integration between AdaptiveBufferManager, BufferHealthMonitor,
ErrorRecoveryManager, and the enhanced audio streaming functionality.
"""

import pytest
import numpy as np
import time
import threading
from queue import Queue, Full
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.buffer_management import (
    AdaptiveBufferManager, BufferMetrics, BufferHealth, HealthStatus
)
from spotify_splitter.buffer_health_monitor import BufferHealthMonitor, AlertLevel
from spotify_splitter.error_recovery import ErrorRecoveryManager, RecoveryAction


class TestEnhancedAudioStreamIntegration:
    """Integration tests for EnhancedAudioStream with all adaptive components."""
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice module for testing."""
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_stream.active = True
            mock_sd.InputStream.return_value = mock_stream
            mock_sd.query_devices.return_value = [
                {'name': 'Test Audio Device'}
            ]
            yield mock_sd
    
    @pytest.fixture
    def buffer_manager(self):
        """Create a test buffer manager."""
        return AdaptiveBufferManager(
            initial_queue_size=100,
            min_size=50,
            max_size=500,
            adjustment_threshold=0.7,
            emergency_threshold=0.9
        )
    
    @pytest.fixture
    def error_recovery(self):
        """Create a test error recovery manager."""
        return ErrorRecoveryManager(max_retries=2, backoff_factor=1.2)
    
    @pytest.fixture
    def health_monitor(self, buffer_manager):
        """Create a test health monitor."""
        return BufferHealthMonitor(
            buffer_manager=buffer_manager,
            monitoring_interval=0.1
        )
    
    @pytest.fixture
    def enhanced_stream(self, mock_sounddevice, buffer_manager, error_recovery, health_monitor):
        """Create an enhanced audio stream for testing."""
        return EnhancedAudioStream(
            monitor_name="test_device",
            buffer_manager=buffer_manager,
            error_recovery=error_recovery,
            health_monitor=health_monitor,
            queue_size=100,
            enable_adaptive_management=True,
            enable_health_monitoring=True
        )
    
    def test_enhanced_stream_initialization(self, enhanced_stream):
        """Test that enhanced stream initializes correctly with all components."""
        assert enhanced_stream.buffer_manager is not None
        assert enhanced_stream.error_recovery is not None
        assert enhanced_stream.health_monitor is not None
        assert enhanced_stream.enable_adaptive_management is True
        assert enhanced_stream.enable_health_monitoring is True
        assert enhanced_stream.reconnection_attempts == 0
        assert enhanced_stream.emergency_expansions == 0
    
    def test_adaptive_callback_basic_functionality(self, enhanced_stream):
        """Test that adaptive callback processes audio data correctly."""
        # Mock audio data
        indata = np.random.random((1024, 2)).astype(np.float32)
        frames = 1024
        time_info = Mock()
        status = None
        
        # Call the adaptive callback
        enhanced_stream._adaptive_callback(indata, frames, time_info, status)
        
        # Verify audio was queued
        assert enhanced_stream.q.qsize() == 1
        assert enhanced_stream.callback_count == 1
        assert enhanced_stream.metrics['total_frames'] == frames
    
    def test_adaptive_callback_with_status_warning(self, enhanced_stream):
        """Test adaptive callback handling of status warnings."""
        indata = np.random.random((1024, 2)).astype(np.float32)
        frames = 1024
        time_info = Mock()
        status = Mock()
        status.__str__ = Mock(return_value="input overflow")
        
        # Mock UI callback
        ui_callback = Mock()
        enhanced_stream.ui_callback = ui_callback
        
        # Call the adaptive callback
        enhanced_stream._adaptive_callback(indata, frames, time_info, status)
        
        # Verify overflow was handled
        assert enhanced_stream.metrics['buffer_overflows'] == 1
        assert enhanced_stream.buffer_manager.overflow_count == 1
        ui_callback.assert_called()
    
    def test_buffer_overflow_handling(self, enhanced_stream):
        """Test buffer overflow handling and emergency expansion."""
        # Fill the queue to capacity
        for _ in range(enhanced_stream.q.maxsize):
            enhanced_stream.q.put(np.zeros((1024, 2)))
        
        # Mock audio data that will cause overflow
        indata = np.random.random((1024, 2)).astype(np.float32)
        frames = 1024
        
        # Process audio data (should trigger overflow handling)
        enhanced_stream._process_audio_data(indata, frames)
        
        # Verify overflow was handled
        assert enhanced_stream.metrics['buffer_overflows'] >= 1
        assert enhanced_stream.buffer_manager.overflow_count >= 1
    
    def test_emergency_buffer_expansion(self, enhanced_stream):
        """Test emergency buffer expansion mechanism."""
        initial_size = enhanced_stream.buffer_manager.current_queue_size
        
        # Trigger emergency expansion
        success = enhanced_stream._attempt_emergency_expansion()
        
        # Verify expansion occurred
        assert success is True
        assert enhanced_stream.buffer_manager.current_queue_size > initial_size
        assert enhanced_stream.emergency_expansions == 1
        assert enhanced_stream.metrics['emergency_expansions'] == 1
    
    def test_adaptive_management_integration(self, enhanced_stream):
        """Test integration between adaptive management components."""
        # Simulate high buffer utilization
        for _ in range(int(enhanced_stream.q.maxsize * 0.8)):
            enhanced_stream.q.put(np.zeros((1024, 2)))
        
        # Perform adaptive management
        enhanced_stream._perform_adaptive_management()
        
        # Verify metrics were collected
        assert len(enhanced_stream.buffer_manager.utilization_history) > 0
        
        # Get buffer health
        health = enhanced_stream.get_buffer_health()
        assert health is not None
        assert isinstance(health, BufferHealth)
    
    def test_error_recovery_integration(self, enhanced_stream):
        """Test error recovery integration with stream operations."""
        # Simulate a stream error
        test_error = Exception("Test stream error")
        
        # Handle the error
        recovery_successful = enhanced_stream.handle_stream_error(test_error)
        
        # Verify error was recorded
        assert len(enhanced_stream.error_recovery.error_history) > 0
        
        # Check that appropriate recovery action was determined
        last_error = enhanced_stream.error_recovery.error_history[-1]
        assert last_error.error_type == "Exception"
        assert last_error.recovery_action is not None
    
    @patch('spotify_splitter.audio.time.sleep')
    def test_stream_reconnection(self, mock_sleep, enhanced_stream, mock_sounddevice):
        """Test automatic stream reconnection functionality."""
        # Mock stream recreation
        mock_stream = Mock()
        mock_sounddevice.InputStream.return_value = mock_stream
        
        # Attempt reconnection
        success = enhanced_stream._attempt_reconnection()
        
        # Verify reconnection was attempted
        assert enhanced_stream.reconnection_attempts == 1
        assert enhanced_stream.metrics['reconnections'] == 1
        mock_sleep.assert_called_once_with(0.1)
    
    def test_performance_metrics_collection(self, enhanced_stream):
        """Test comprehensive performance metrics collection."""
        # Simulate some activity
        enhanced_stream.metrics['total_frames'] = 10000
        enhanced_stream.metrics['dropped_frames'] = 5
        enhanced_stream.reconnection_attempts = 1
        
        # Get performance metrics
        metrics = enhanced_stream.get_performance_metrics()
        
        # Verify all metric categories are present
        assert 'total_frames' in metrics
        assert 'dropped_frames' in metrics
        assert 'reconnection_attempts' in metrics
        assert 'buffer_stats' in metrics
        assert 'error_recovery' in metrics
        assert 'current_queue_size' in metrics
    
    def test_health_monitoring_integration(self, enhanced_stream):
        """Test integration with buffer health monitoring."""
        # Start monitoring (should already be started in fixture)
        assert enhanced_stream.health_monitor._monitoring is True
        
        # Simulate some buffer activity
        for _ in range(10):
            enhanced_stream.q.put(np.zeros((1024, 2)))
        
        # Wait for monitoring to collect data
        time.sleep(0.2)
        
        # Check that health data was collected
        current_health = enhanced_stream.health_monitor.get_current_health()
        if current_health:  # May be None if monitoring hasn't collected data yet
            assert isinstance(current_health, BufferHealth)
    
    def test_context_manager_functionality(self, enhanced_stream, mock_sounddevice):
        """Test enhanced context manager with error handling."""
        mock_stream = enhanced_stream.stream
        
        # Test context manager entry
        with enhanced_stream as stream:
            assert stream is enhanced_stream
            mock_stream.start.assert_called()
        
        # Test context manager exit
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()
    
    def test_context_manager_with_startup_error(self, enhanced_stream, mock_sounddevice):
        """Test context manager handling of startup errors."""
        # Mock stream start to raise an error
        enhanced_stream.stream.start.side_effect = Exception("Startup error")
        
        # Mock error recovery to return success
        with patch.object(enhanced_stream, 'handle_stream_error', return_value=True):
            with enhanced_stream as stream:
                assert stream is enhanced_stream
    
    def test_callback_error_handling(self, enhanced_stream):
        """Test error handling within the adaptive callback."""
        # Mock an error in adaptive management
        with patch.object(enhanced_stream, '_perform_adaptive_management', 
                         side_effect=Exception("Management error")):
            
            indata = np.random.random((1024, 2)).astype(np.float32)
            frames = 1024
            time_info = Mock()
            status = None
            
            # Callback should handle the error gracefully
            enhanced_stream._adaptive_callback(indata, frames, time_info, status)
            
            # Verify error was recorded
            assert len(enhanced_stream.error_recovery.error_history) > 0


class TestEnhancedAudioStreamStressTests:
    """Stress tests for enhanced audio stream under high load conditions."""
    
    @pytest.fixture
    def stress_test_stream(self, mock_sounddevice):
        """Create a stream configured for stress testing."""
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=50,
            min_size=25,
            max_size=200,
            adjustment_threshold=0.6,
            emergency_threshold=0.85
        )
        
        return EnhancedAudioStream(
            monitor_name="stress_test_device",
            buffer_manager=buffer_manager,
            queue_size=50,
            enable_adaptive_management=True,
            enable_health_monitoring=False  # Disable for stress test
        )
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice for stress tests."""
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_stream.active = True
            mock_sd.InputStream.return_value = mock_stream
            mock_sd.query_devices.return_value = [
                {'name': 'Stress Test Device'}
            ]
            yield mock_sd
    
    def test_high_frequency_callbacks(self, stress_test_stream):
        """Test stream handling of high-frequency audio callbacks."""
        callback_count = 100
        frames_per_callback = 512
        
        # Simulate rapid callbacks
        for i in range(callback_count):
            indata = np.random.random((frames_per_callback, 2)).astype(np.float32)
            stress_test_stream._adaptive_callback(indata, frames_per_callback, Mock(), None)
        
        # Verify all callbacks were processed
        assert stress_test_stream.callback_count == callback_count
        assert stress_test_stream.metrics['total_frames'] == callback_count * frames_per_callback
    
    def test_buffer_overflow_recovery_cycle(self, stress_test_stream):
        """Test repeated buffer overflow and recovery cycles."""
        overflow_cycles = 5
        
        for cycle in range(overflow_cycles):
            # Fill buffer to capacity
            while not stress_test_stream.q.full():
                try:
                    stress_test_stream.q.put_nowait(np.zeros((1024, 2)))
                except Full:
                    break
            
            # Trigger overflow handling
            stress_test_stream._handle_buffer_overflow()
            
            # Clear some space
            for _ in range(10):
                if not stress_test_stream.q.empty():
                    stress_test_stream.q.get_nowait()
        
        # Verify overflow handling occurred
        assert stress_test_stream.metrics['buffer_overflows'] >= overflow_cycles
        assert stress_test_stream.buffer_manager.overflow_count >= overflow_cycles
    
    def test_concurrent_callback_processing(self, stress_test_stream):
        """Test thread safety under concurrent callback processing."""
        def simulate_callback(stream, callback_id):
            """Simulate a single callback in a separate thread."""
            indata = np.random.random((1024, 2)).astype(np.float32)
            stream._adaptive_callback(indata, 1024, Mock(), None)
        
        # Create multiple threads to simulate concurrent callbacks
        threads = []
        thread_count = 10
        
        for i in range(thread_count):
            thread = threading.Thread(
                target=simulate_callback,
                args=(stress_test_stream, i)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify all callbacks were processed
        assert stress_test_stream.callback_count == thread_count
        assert stress_test_stream.metrics['total_frames'] == thread_count * 1024
    
    def test_memory_usage_under_load(self, stress_test_stream):
        """Test memory usage remains stable under sustained load."""
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Simulate sustained load
        for _ in range(200):
            indata = np.random.random((2048, 2)).astype(np.float32)
            stress_test_stream._adaptive_callback(indata, 2048, Mock(), None)
            
            # Periodically clear the queue to prevent unbounded growth
            if stress_test_stream.q.qsize() > 40:
                for _ in range(20):
                    if not stress_test_stream.q.empty():
                        stress_test_stream.q.get_nowait()
        
        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not have grown excessively
        # Allow for some growth but not unbounded
        memory_growth_ratio = final_objects / initial_objects
        assert memory_growth_ratio < 2.0, f"Memory usage grew by {memory_growth_ratio:.2f}x"


class TestEnhancedAudioStreamErrorScenarios:
    """Test error scenarios and recovery mechanisms."""
    
    @pytest.fixture
    def error_prone_stream(self, mock_sounddevice):
        """Create a stream for error scenario testing."""
        error_recovery = ErrorRecoveryManager(max_retries=1, backoff_factor=1.1)
        
        return EnhancedAudioStream(
            monitor_name="error_test_device",
            error_recovery=error_recovery,
            enable_adaptive_management=True,
            enable_health_monitoring=False
        )
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice for error tests."""
        with patch('spotify_splitter.audio.sd') as mock_sd:
            mock_stream = Mock()
            mock_stream.active = True
            mock_sd.InputStream.return_value = mock_stream
            mock_sd.query_devices.return_value = [
                {'name': 'Error Test Device'}
            ]
            yield mock_sd
    
    def test_stream_device_error_recovery(self, error_prone_stream):
        """Test recovery from audio device errors."""
        # Simulate device error
        device_error = OSError("Audio device disconnected")
        
        # Handle the error
        recovery_success = error_prone_stream.handle_stream_error(device_error)
        
        # Verify error was handled and recovery attempted
        assert len(error_prone_stream.error_recovery.error_history) > 0
        last_error = error_prone_stream.error_recovery.error_history[-1]
        assert last_error.error_type == "OSError"
        assert last_error.recovery_action == RecoveryAction.RECONNECT
    
    def test_multiple_consecutive_errors(self, error_prone_stream):
        """Test handling of multiple consecutive errors."""
        errors = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3")
        ]
        
        for error in errors:
            error_prone_stream.handle_stream_error(error)
        
        # Verify all errors were recorded
        assert len(error_prone_stream.error_recovery.error_history) == len(errors)
        
        # Check error escalation for repeated errors
        error_types = [e.error_type for e in error_prone_stream.error_recovery.error_history]
        assert all(error_type == "Exception" for error_type in error_types)
    
    def test_callback_exception_handling(self, error_prone_stream):
        """Test exception handling within callback processing."""
        # Mock queue to raise exception
        with patch.object(error_prone_stream.q, 'put_nowait', side_effect=Exception("Queue error")):
            indata = np.random.random((1024, 2)).astype(np.float32)
            
            # Callback should handle exception gracefully
            error_prone_stream._adaptive_callback(indata, 1024, Mock(), None)
            
            # Verify error was handled
            assert error_prone_stream.metrics['dropped_frames'] > 0
    
    def test_reconnection_attempt_limit(self, error_prone_stream):
        """Test that reconnection attempts are limited."""
        # Set reconnection attempts to near limit
        error_prone_stream.reconnection_attempts = error_prone_stream.max_reconnection_attempts - 1
        
        # Attempt one more reconnection (should succeed)
        success1 = error_prone_stream.reconnect_stream()
        
        # Attempt another reconnection (should fail due to limit)
        success2 = error_prone_stream.reconnect_stream()
        
        assert success2 is False
        assert error_prone_stream.reconnection_attempts == error_prone_stream.max_reconnection_attempts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])