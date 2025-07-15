"""
Tests for the error recovery management system.

This module tests automatic error recovery, stream reconnection,
device change detection, and comprehensive error diagnostics.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from queue import Queue, Full

from spotify_splitter.error_recovery import (
    ErrorRecoveryManager,
    RecoveryAction,
    ErrorSeverity,
    ErrorEvent,
    DeviceInfo,
    ErrorDiagnostics
)


# Custom exception classes for testing
class PortAudioError(Exception):
    """Mock PortAudio error for testing."""
    pass


class BufferError(Exception):
    """Mock buffer error for testing."""
    pass


class CustomError(Exception):
    """Custom error for testing."""
    pass


class TestErrorRecoveryManager:
    """Test cases for ErrorRecoveryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ErrorRecoveryManager(
            max_retries=3,
            backoff_factor=1.5,
            max_backoff=10.0,
            error_history_size=50
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.manager, 'device_monitoring_enabled') and self.manager.device_monitoring_enabled:
            self.manager.stop_device_monitoring()
    
    def test_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        assert self.manager.max_retries == 3
        assert self.manager.backoff_factor == 1.5
        assert self.manager.max_backoff == 10.0
        assert len(self.manager.error_history) == 0
        assert len(self.manager.error_counts) == 0
        assert not self.manager.device_monitoring_enabled
    
    def test_error_classification(self):
        """Test error severity classification."""
        # Test critical errors
        critical_error = SystemError("System failure")
        action = self.manager.handle_error(critical_error, "test_context")
        assert action == RecoveryAction.ESCALATE
        
        # Test high severity errors
        high_error = PortAudioError("Device unavailable")
        action = self.manager.handle_error(high_error, "test_context")
        assert action == RecoveryAction.RECONNECT
        
        # Test medium severity errors
        medium_error = ConnectionError("Connection timeout")
        action = self.manager.handle_error(medium_error, "test_context")
        assert action == RecoveryAction.RETRY
    
    def test_error_recording(self):
        """Test error event recording and statistics."""
        # Record some errors
        error1 = ValueError("Test error 1")
        error2 = ValueError("Test error 2")
        error3 = ConnectionError("Connection failed")
        
        self.manager.handle_error(error1, "context1")
        self.manager.handle_error(error2, "context2")
        self.manager.handle_error(error3, "context3")
        
        # Check error history
        assert len(self.manager.error_history) == 3
        assert self.manager.error_counts["ValueError"] == 2
        assert self.manager.error_counts["ConnectionError"] == 1
        
        # Check error events
        last_error = self.manager.error_history[-1]
        assert last_error.error_type == "ConnectionError"
        assert last_error.context == "context3"
        assert last_error.recovery_action == RecoveryAction.RETRY
    
    def test_frequent_error_escalation(self):
        """Test escalation of frequently occurring errors."""
        # Create multiple instances of the same error type
        for i in range(4):
            error = ValueError(f"Frequent error {i}")
            action = self.manager.handle_error(error, "test_context")
            
            # First few should be retries, but frequent ones should escalate
            if i < 3:
                assert action == RecoveryAction.RETRY
            else:
                assert action == RecoveryAction.ESCALATE
    
    def test_recovery_attempt_success(self):
        """Test successful recovery attempt."""
        # Mock successful recovery function
        recovery_func = Mock(return_value=True)
        
        success = self.manager.attempt_recovery(
            RecoveryAction.RETRY,
            recovery_func,
            "test_recovery"
        )
        
        assert success is True
        recovery_func.assert_called_once()
        
        # Check statistics
        stats = self.manager.recovery_stats[RecoveryAction.RETRY]
        assert stats["attempted"] == 1
        assert stats["successful"] == 1
        assert stats["failed"] == 0
    
    def test_recovery_attempt_failure(self):
        """Test failed recovery attempt."""
        # Mock failing recovery function
        recovery_func = Mock(return_value=False)
        
        success = self.manager.attempt_recovery(
            RecoveryAction.RECONNECT,
            recovery_func,
            "test_recovery"
        )
        
        assert success is False
        recovery_func.assert_called_once()
        
        # Check statistics
        stats = self.manager.recovery_stats[RecoveryAction.RECONNECT]
        assert stats["attempted"] == 1
        assert stats["successful"] == 0
        assert stats["failed"] == 1
    
    def test_recovery_attempt_exception(self):
        """Test recovery attempt that raises an exception."""
        # Mock recovery function that raises exception
        recovery_func = Mock(side_effect=Exception("Recovery failed"))
        
        success = self.manager.attempt_recovery(
            RecoveryAction.BUFFER_EXPAND,
            recovery_func,
            "test_recovery"
        )
        
        assert success is False
        recovery_func.assert_called_once()
        
        # Check statistics
        stats = self.manager.recovery_stats[RecoveryAction.BUFFER_EXPAND]
        assert stats["attempted"] == 1
        assert stats["successful"] == 0
        assert stats["failed"] == 1
    
    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep):
        """Test exponential backoff for retry attempts."""
        recovery_func = Mock(return_value=True)
        
        # First attempt - should have delay based on attempt count (starts at 0, so 1.5^1 = 1.5)
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test1")
        mock_sleep.assert_called_with(1.5)
        
        # Second attempt - longer delay (1.5^2 = 2.25)
        mock_sleep.reset_mock()
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test2")
        mock_sleep.assert_called_with(2.25)
        
        # Third attempt - even longer delay (1.5^3 = 3.375)
        mock_sleep.reset_mock()
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test3")
        mock_sleep.assert_called_with(3.375)
    
    def test_error_escalation(self):
        """Test error escalation functionality."""
        error = RuntimeError("Critical system error")
        
        # Record an error first
        self.manager.handle_error(error, "test_context")
        
        # Escalate the error
        self.manager.escalate_error(error, "escalation_context")
        
        # Check that the last error event was marked as escalated
        last_error = self.manager.error_history[-1]
        assert last_error.recovery_action == RecoveryAction.ESCALATE
        assert last_error.recovery_successful is False
    
    def test_custom_recovery_strategy(self):
        """Test setting custom recovery strategies."""
        # Set custom strategy
        self.manager.set_recovery_strategy("CustomError", RecoveryAction.BUFFER_EXPAND)
        
        # Create error with custom type
        error = CustomError("Custom error message")
        
        action = self.manager.handle_error(error, "test_context")
        assert action == RecoveryAction.BUFFER_EXPAND
    
    def test_error_history_limit(self):
        """Test that error history respects size limit."""
        # Create manager with small history size
        small_manager = ErrorRecoveryManager(error_history_size=5)
        
        # Add more errors than the limit
        for i in range(10):
            error = ValueError(f"Error {i}")
            small_manager.handle_error(error, f"context_{i}")
        
        # Should only keep the last 5 errors
        assert len(small_manager.error_history) == 5
        assert small_manager.error_history[-1].error_message == "Error 9"
        assert small_manager.error_history[0].error_message == "Error 5"
    
    def test_clear_error_history(self):
        """Test clearing error history and statistics."""
        # Add some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            self.manager.handle_error(error, f"context_{i}")
        
        # Add some recovery attempts
        recovery_func = Mock(return_value=True)
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test")
        
        # Clear history
        self.manager.clear_error_history()
        
        # Check that everything is cleared
        assert len(self.manager.error_history) == 0
        assert len(self.manager.error_counts) == 0
        assert len(self.manager.recovery_stats) == 0
    
    def test_diagnostics_generation(self):
        """Test comprehensive diagnostics generation."""
        # Add various errors and recovery attempts
        errors = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            ConnectionError("Connection failed"),
            RuntimeError("Runtime error")
        ]
        
        for error in errors:
            self.manager.handle_error(error, "test_context")
        
        # Add some recovery attempts
        recovery_func = Mock(return_value=True)
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test1")
        
        recovery_func = Mock(return_value=False)
        self.manager.attempt_recovery(RecoveryAction.RECONNECT, recovery_func, "test2")
        
        # Generate diagnostics
        diagnostics = self.manager.get_diagnostics()
        
        assert diagnostics.total_errors == 4
        assert len(diagnostics.most_common_errors) > 0
        assert diagnostics.most_common_errors[0] == ("ValueError", 2)
        assert len(diagnostics.recent_errors) == 4
        assert diagnostics.recovery_success_rate == 0.5  # 1 success out of 2 attempts
        assert len(diagnostics.recommendations) >= 0
    
    def test_statistics_collection(self):
        """Test comprehensive statistics collection."""
        # Add some errors and recovery attempts
        error = ValueError("Test error")
        self.manager.handle_error(error, "test_context")
        
        recovery_func = Mock(return_value=True)
        self.manager.attempt_recovery(RecoveryAction.RETRY, recovery_func, "test")
        
        # Get statistics
        stats = self.manager.get_statistics()
        
        assert stats["total_errors"] == 1
        assert "ValueError" in stats["error_types"]
        assert stats["error_types"]["ValueError"] == 1
        assert "retry" in stats["recovery_stats"]
        assert stats["recovery_stats"]["retry"]["attempted"] == 1
        assert stats["recovery_stats"]["retry"]["successful"] == 1
        assert "configuration" in stats
        assert stats["configuration"]["max_retries"] == 3


class TestDeviceChangeDetection:
    """Test cases for device change detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ErrorRecoveryManager()
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.manager.device_monitoring_enabled:
            self.manager.stop_device_monitoring()
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_get_available_devices(self, mock_query):
        """Test getting available audio devices."""
        # Mock device list
        mock_devices = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
            {'name': 'Device 2', 'max_input_channels': 0, 'default_samplerate': 48000},  # Output only
            {'name': 'Device 3', 'max_input_channels': 1, 'default_samplerate': 22050},
        ]
        mock_query.return_value = mock_devices
        
        devices = self.manager.get_available_devices()
        
        # Should only return input devices (max_input_channels > 0)
        assert len(devices) == 2
        assert devices[0].name == 'Device 1'
        assert devices[0].channels == 2
        assert devices[0].samplerate == 44100
        assert devices[1].name == 'Device 3'
        assert devices[1].channels == 1
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_find_device_by_name(self, mock_query):
        """Test finding devices by name."""
        mock_devices = [
            {'name': 'USB Audio Device', 'max_input_channels': 2, 'default_samplerate': 44100},
            {'name': 'Built-in Microphone', 'max_input_channels': 1, 'default_samplerate': 48000},
        ]
        mock_query.return_value = mock_devices
        
        # Test exact match
        device = self.manager.find_device_by_name('USB Audio Device')
        assert device is not None
        assert device.name == 'USB Audio Device'
        
        # Test partial match
        device = self.manager.find_device_by_name('USB')
        assert device is not None
        assert device.name == 'USB Audio Device'
        
        # Test no match
        device = self.manager.find_device_by_name('Nonexistent Device')
        assert device is None
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_device_monitoring_start_stop(self, mock_query):
        """Test starting and stopping device monitoring."""
        mock_query.return_value = []
        
        # Start monitoring
        assert not self.manager.device_monitoring_enabled
        self.manager.start_device_monitoring()
        assert self.manager.device_monitoring_enabled
        assert self.manager._device_monitor_thread is not None
        assert self.manager._device_monitor_thread.is_alive()
        
        # Stop monitoring
        self.manager.stop_device_monitoring()
        assert not self.manager.device_monitoring_enabled
        
        # Wait a moment for thread to stop
        time.sleep(0.1)
        assert not self.manager._device_monitor_thread.is_alive()
    
    def test_device_change_callbacks(self):
        """Test device change callback registration and removal."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        self.manager.add_device_change_callback(callback1)
        self.manager.add_device_change_callback(callback2)
        assert len(self.manager.device_change_callbacks) == 2
        
        # Remove callback
        self.manager.remove_device_change_callback(callback1)
        assert len(self.manager.device_change_callbacks) == 1
        assert callback2 in self.manager.device_change_callbacks
        assert callback1 not in self.manager.device_change_callbacks
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_device_change_detection(self, mock_query):
        """Test detection of device changes."""
        callback = Mock()
        self.manager.add_device_change_callback(callback)
        
        # Initial device list
        initial_devices = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
        ]
        mock_query.return_value = initial_devices
        
        # Initialize device list
        self.manager._update_device_list()
        assert len(self.manager.current_devices) == 1
        
        # Simulate device addition
        updated_devices = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
            {'name': 'Device 2', 'max_input_channels': 1, 'default_samplerate': 48000},
        ]
        mock_query.return_value = updated_devices
        
        # Check for changes
        self.manager._check_device_changes()
        
        # Callback should be called with added device
        callback.assert_called_once()
        added_devices, removed_devices = callback.call_args[0]
        assert len(added_devices) == 1
        assert len(removed_devices) == 0
        assert added_devices[0].name == 'Device 2'
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_device_removal_detection(self, mock_query):
        """Test detection of device removal."""
        callback = Mock()
        self.manager.add_device_change_callback(callback)
        
        # Initial device list with two devices
        initial_devices = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
            {'name': 'Device 2', 'max_input_channels': 1, 'default_samplerate': 48000},
        ]
        mock_query.return_value = initial_devices
        self.manager._update_device_list()
        
        # Simulate device removal
        updated_devices = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
        ]
        mock_query.return_value = updated_devices
        
        # Check for changes
        self.manager._check_device_changes()
        
        # Callback should be called with removed device
        callback.assert_called_once()
        added_devices, removed_devices = callback.call_args[0]
        assert len(added_devices) == 0
        assert len(removed_devices) == 1
        assert removed_devices[0].name == 'Device 2'
    
    @patch('spotify_splitter.error_recovery.sd.query_devices')
    def test_device_monitoring_error_handling(self, mock_query):
        """Test error handling in device monitoring."""
        # Make query_devices raise an exception
        mock_query.side_effect = Exception("Device query failed")
        
        # This should not raise an exception
        devices = self.manager.get_available_devices()
        assert devices == []  # Should return empty list on error
        
        # Device change checking should also handle errors gracefully
        self.manager._check_device_changes()  # Should not raise exception
    
    def test_callback_error_handling(self):
        """Test error handling in device change callbacks."""
        # Add a callback that raises an exception
        bad_callback = Mock(side_effect=Exception("Callback error"))
        good_callback = Mock()
        
        self.manager.add_device_change_callback(bad_callback)
        self.manager.add_device_change_callback(good_callback)
        
        # Notify callbacks - should handle the exception gracefully
        added = [DeviceInfo(0, "Test Device", 2, 44100)]
        removed = []
        
        self.manager._notify_device_change_callbacks(added, removed)
        
        # Both callbacks should be called despite the error in the first one
        bad_callback.assert_called_once_with(added, removed)
        good_callback.assert_called_once_with(added, removed)


class TestErrorRecoveryIntegration:
    """Integration tests for error recovery with audio components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ErrorRecoveryManager()
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.manager, 'device_monitoring_enabled') and self.manager.device_monitoring_enabled:
            self.manager.stop_device_monitoring()
    
    def test_buffer_overflow_recovery(self):
        """Test recovery from buffer overflow scenarios."""
        # Use the BufferError class defined at the top of the file
        buffer_error = BufferError("Buffer overflow")
        
        action = self.manager.handle_error(buffer_error, "audio_callback")
        assert action == RecoveryAction.BUFFER_EXPAND
        
        # Test recovery attempt
        recovery_func = Mock(return_value=True)
        success = self.manager.attempt_recovery(action, recovery_func, "buffer_expansion")
        assert success is True
    
    def test_stream_reconnection_scenario(self):
        """Test stream reconnection recovery scenario."""
        # Use the PortAudioError class defined at the top of the file
        stream_error = PortAudioError("Stream disconnected")
        
        action = self.manager.handle_error(stream_error, "audio_stream")
        assert action == RecoveryAction.RECONNECT
        
        # Test reconnection attempt
        reconnect_func = Mock(return_value=True)
        success = self.manager.attempt_recovery(action, reconnect_func, "stream_reconnection")
        assert success is True
        
        # Check statistics
        stats = self.manager.get_statistics()
        assert stats["recovery_stats"]["reconnect"]["successful"] == 1
    
    def test_multiple_error_recovery_sequence(self):
        """Test handling multiple errors in sequence."""
        errors = [
            (ConnectionError("Network timeout"), RecoveryAction.RETRY),
            (PortAudioError("Stream disconnected"), RecoveryAction.RECONNECT),
            (BufferError("Buffer overflow"), RecoveryAction.BUFFER_EXPAND),
        ]
        
        recovery_results = []
        
        for error, expected_action in errors:
            action = self.manager.handle_error(error, "test_sequence")
            assert action == expected_action
            
            # Attempt recovery
            recovery_func = Mock(return_value=True)
            success = self.manager.attempt_recovery(action, recovery_func, f"recovery_{action.value}")
            recovery_results.append(success)
        
        # All recoveries should succeed
        assert all(recovery_results)
        
        # Check final statistics
        stats = self.manager.get_statistics()
        assert stats["total_errors"] == 3
        assert len(stats["error_types"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])