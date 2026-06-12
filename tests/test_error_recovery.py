"""
Tests for the error recovery management system.

This module tests automatic error recovery, stream reconnection,
and comprehensive error diagnostics.
"""

import pytest
from unittest.mock import Mock, patch

from spotify_splitter.error_recovery import (
    ErrorRecoveryManager,
    RecoveryAction,
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
    
    def test_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        assert self.manager.max_retries == 3
        assert self.manager.backoff_factor == 1.5
        assert self.manager.max_backoff == 10.0
        assert len(self.manager.error_history) == 0
        assert len(self.manager.error_counts) == 0
    
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

class TestErrorRecoveryIntegration:
    """Integration tests for error recovery with audio components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ErrorRecoveryManager()
    
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
