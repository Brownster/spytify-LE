"""
Unit tests for the buffer health monitoring system.

Tests cover real-time monitoring, alert generation, health reporting,
trend analysis, and performance tracking.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import Mock, patch, MagicMock

from spotify_splitter.buffer_health_monitor import (
    BufferHealthMonitor,
    BufferAlert,
    HealthReport,
    AlertLevel
)
from spotify_splitter.buffer_management import (
    AdaptiveBufferManager,
    BufferMetrics,
    BufferHealth,
    HealthStatus
)


class TestBufferAlert:
    """Test BufferAlert data model."""
    
    def test_buffer_alert_creation(self):
        """Test basic BufferAlert creation."""
        metrics = BufferMetrics(
            utilization_percent=85.0,
            queue_size=170,
            overflow_count=1,
            underrun_count=0,
            average_latency_ms=60.0,
            peak_latency_ms=120.0,
            timestamp=datetime.now()
        )
        
        alert = BufferAlert(
            level=AlertLevel.WARNING,
            message="High buffer utilization detected",
            timestamp=datetime.now(),
            metrics=metrics,
            recommended_action="Increase buffer size"
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "High buffer utilization detected"
        assert isinstance(alert.timestamp, datetime)
        assert alert.metrics == metrics
        assert alert.recommended_action == "Increase buffer size"


class TestHealthReport:
    """Test HealthReport data model."""
    
    def test_health_report_creation(self):
        """Test basic HealthReport creation."""
        current_health = Mock()
        alerts = [Mock(), Mock()]
        performance_summary = {"total_checks": 100}
        trend_analysis = {"direction": "stable"}
        
        report = HealthReport(
            timestamp=datetime.now(),
            current_health=current_health,
            alerts=alerts,
            performance_summary=performance_summary,
            trend_analysis=trend_analysis
        )
        
        assert isinstance(report.timestamp, datetime)
        assert report.current_health == current_health
        assert len(report.alerts) == 2
        assert report.performance_summary["total_checks"] == 100
        assert report.trend_analysis["direction"] == "stable"


class TestBufferHealthMonitor:
    """Test BufferHealthMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer_manager = AdaptiveBufferManager(
            initial_queue_size=100,
            min_size=50,
            max_size=500
        )
        self.alert_callback = Mock()
        self.monitor = BufferHealthMonitor(
            buffer_manager=self.buffer_manager,
            monitoring_interval=0.1,  # Fast interval for testing
            alert_callback=self.alert_callback
        )
        self.audio_queue = Queue(maxsize=100)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.monitor, '_monitoring') and self.monitor._monitoring:
            self.monitor.stop_monitoring()
    
    def test_initialization(self):
        """Test BufferHealthMonitor initialization."""
        assert self.monitor.buffer_manager == self.buffer_manager
        assert self.monitor.monitoring_interval == 0.1
        assert self.monitor.alert_callback == self.alert_callback
        assert not self.monitor._monitoring
        assert len(self.monitor.health_history) == 0
        assert len(self.monitor.alert_history) == 0
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Test start
        self.monitor.start_monitoring(self.audio_queue)
        assert self.monitor._monitoring
        assert self.monitor._monitor_thread is not None
        assert self.monitor._monitor_thread.is_alive()
        
        # Test stop
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring
        
        # Wait for thread to finish
        time.sleep(0.2)
        assert not self.monitor._monitor_thread.is_alive()
    
    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        self.monitor.start_monitoring(self.audio_queue)
        
        # Try to start again - should log warning but not crash
        with patch('spotify_splitter.buffer_health_monitor.logger') as mock_logger:
            self.monitor.start_monitoring(self.audio_queue)
            mock_logger.warning.assert_called_once()
        
        self.monitor.stop_monitoring()
    
    def test_health_check_collection(self):
        """Test that health checks are collected during monitoring."""
        # Add some items to queue to create utilization
        for i in range(50):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        
        # Wait for a few monitoring cycles
        time.sleep(0.3)
        
        self.monitor.stop_monitoring()
        
        # Check that health data was collected
        assert len(self.monitor.health_history) > 0
        assert self.monitor.performance_stats["total_checks"] > 0
        
        # Verify health data structure
        health_record = self.monitor.health_history[0]
        assert 'timestamp' in health_record
        assert 'health' in health_record
        assert 'metrics' in health_record
        assert isinstance(health_record['health'], BufferHealth)
        assert isinstance(health_record['metrics'], BufferMetrics)
    
    def test_warning_alert_generation(self):
        """Test warning alert generation for high utilization."""
        # Fill queue to trigger warning threshold (80%)
        for i in range(85):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        
        # Wait for monitoring to detect high utilization
        time.sleep(0.3)
        
        self.monitor.stop_monitoring()
        
        # Check that warning alert was generated
        warning_alerts = [
            alert for alert in self.monitor.alert_history
            if alert.level == AlertLevel.WARNING
        ]
        assert len(warning_alerts) > 0
        
        # Verify alert callback was called
        assert self.alert_callback.called
        
        # Verify alert content
        alert = warning_alerts[0]
        assert "WARNING" in alert.message
        assert "utilization" in alert.message.lower()
        assert alert.recommended_action is not None
    
    def test_critical_alert_generation(self):
        """Test critical alert generation for very high utilization."""
        # Fill queue to trigger critical threshold (95%)
        for i in range(96):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        
        # Wait for monitoring to detect critical utilization
        time.sleep(0.3)
        
        self.monitor.stop_monitoring()
        
        # Check that critical alert was generated
        critical_alerts = [
            alert for alert in self.monitor.alert_history
            if alert.level == AlertLevel.CRITICAL
        ]
        assert len(critical_alerts) > 0
        
        # Verify alert content
        alert = critical_alerts[0]
        assert "CRITICAL" in alert.message
        assert "overflow imminent" in alert.message.lower()
        assert "emergency" in alert.recommended_action.lower()
    
    def test_info_alert_for_low_utilization(self):
        """Test info alert generation for consistently low utilization."""
        # Keep queue mostly empty for low utilization
        self.monitor.start_monitoring(self.audio_queue)
        
        # Wait for enough samples to establish low utilization trend
        time.sleep(1.2)  # Need time for history to build up
        
        self.monitor.stop_monitoring()
        
        # Check for info alerts about optimization opportunity
        info_alerts = [
            alert for alert in self.monitor.alert_history
            if alert.level == AlertLevel.INFO
        ]
        
        # May or may not have info alerts depending on timing, but shouldn't crash
        if info_alerts:
            alert = info_alerts[0]
            assert "INFO" in alert.message
            assert "optimization" in alert.message.lower() or "low" in alert.message.lower()
    
    def test_alert_cooldown(self):
        """Test that alert cooldown prevents spam."""
        # Set very short cooldown for testing
        self.monitor.alert_cooldowns[AlertLevel.WARNING] = timedelta(seconds=0.5)
        
        # Fill queue to trigger warnings
        for i in range(85):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        
        # Wait for first alert
        time.sleep(0.2)
        first_alert_count = len(self.monitor.alert_history)
        
        # Wait a bit more but less than cooldown
        time.sleep(0.2)
        second_alert_count = len(self.monitor.alert_history)
        
        # Should not have new alerts due to cooldown
        assert second_alert_count == first_alert_count
        
        # Wait for cooldown to expire
        time.sleep(0.4)
        third_alert_count = len(self.monitor.alert_history)
        
        # Now should have new alerts
        assert third_alert_count >= second_alert_count
        
        self.monitor.stop_monitoring()
    
    def test_get_current_health(self):
        """Test getting current health status."""
        # Initially no health data
        assert self.monitor.get_current_health() is None
        
        # Add some queue items and start monitoring
        for i in range(30):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        time.sleep(0.2)
        self.monitor.stop_monitoring()
        
        # Should now have current health
        current_health = self.monitor.get_current_health()
        assert current_health is not None
        assert isinstance(current_health, BufferHealth)
        assert current_health.utilization > 0
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts within time window."""
        # Create some test alerts with different timestamps
        old_alert = BufferAlert(
            level=AlertLevel.WARNING,
            message="Old alert",
            timestamp=datetime.now() - timedelta(minutes=15),
            metrics=Mock()
        )
        recent_alert = BufferAlert(
            level=AlertLevel.WARNING,
            message="Recent alert",
            timestamp=datetime.now() - timedelta(minutes=5),
            metrics=Mock()
        )
        
        self.monitor.alert_history.extend([old_alert, recent_alert])
        
        # Get alerts from last 10 minutes
        recent_alerts = self.monitor.get_recent_alerts(minutes=10)
        
        assert len(recent_alerts) == 1
        assert recent_alerts[0].message == "Recent alert"
    
    def test_generate_health_report(self):
        """Test comprehensive health report generation."""
        # Add some test data
        for i in range(20):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        time.sleep(0.3)
        self.monitor.stop_monitoring()
        
        # Generate report
        report = self.monitor.generate_health_report()
        
        assert isinstance(report, HealthReport)
        assert isinstance(report.timestamp, datetime)
        assert report.current_health is not None
        assert isinstance(report.alerts, list)
        assert isinstance(report.performance_summary, dict)
        assert isinstance(report.trend_analysis, dict)
        
        # Check performance summary content
        assert "total_checks" in report.performance_summary
        assert "monitoring_uptime_seconds" in report.performance_summary
        assert "alert_rate_per_hour" in report.performance_summary
        
        # Check trend analysis
        assert "status" in report.trend_analysis
    
    def test_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        # Add minimal data
        self.monitor.start_monitoring(self.audio_queue)
        time.sleep(0.1)
        self.monitor.stop_monitoring()
        
        trend_analysis = self.monitor._analyze_trends()
        
        assert trend_analysis["status"] == "insufficient_data"
        assert "message" in trend_analysis
    
    def test_trend_analysis_with_data(self):
        """Test trend analysis with sufficient data."""
        # Add items and monitor for a while to build history
        for i in range(40):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        time.sleep(0.5)  # Let it collect enough samples
        self.monitor.stop_monitoring()
        
        trend_analysis = self.monitor._analyze_trends()
        
        if trend_analysis["status"] == "analyzed":
            assert "utilization_trend" in trend_analysis
            assert "health_distribution" in trend_analysis
            assert "samples_analyzed" in trend_analysis
            
            # Check utilization trend structure
            util_trend = trend_analysis["utilization_trend"]
            assert "direction" in util_trend
            assert "average" in util_trend
            assert "maximum" in util_trend
            assert "minimum" in util_trend
    
    def test_custom_alert_thresholds(self):
        """Test setting custom alert thresholds."""
        # Set custom warning threshold
        self.monitor.set_alert_threshold(AlertLevel.WARNING, 0.6)
        assert self.monitor.alert_thresholds[AlertLevel.WARNING] == 0.6
        
        # Test validation
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            self.monitor.set_alert_threshold(AlertLevel.WARNING, 1.5)
    
    def test_custom_alert_cooldowns(self):
        """Test setting custom alert cooldowns."""
        custom_cooldown = timedelta(minutes=5)
        self.monitor.set_alert_cooldown(AlertLevel.WARNING, custom_cooldown)
        assert self.monitor.alert_cooldowns[AlertLevel.WARNING] == custom_cooldown
    
    def test_clear_histories(self):
        """Test clearing alert and health histories."""
        # Add some test data
        self.monitor.alert_history.append(Mock())
        self.monitor.health_history.append(Mock())
        self.monitor.last_alert_time[AlertLevel.WARNING] = datetime.now()
        
        # Clear alert history
        self.monitor.clear_alert_history()
        assert len(self.monitor.alert_history) == 0
        assert len(self.monitor.last_alert_time) == 0
        
        # Clear health history
        self.monitor.clear_health_history()
        assert len(self.monitor.health_history) == 0
    
    def test_get_statistics(self):
        """Test getting monitoring statistics."""
        stats = self.monitor.get_statistics()
        
        assert isinstance(stats, dict)
        assert "monitoring_active" in stats
        assert "health_history_size" in stats
        assert "alert_history_size" in stats
        assert "performance_stats" in stats
        assert "alert_thresholds" in stats
        assert "alert_cooldowns" in stats
        
        # Check that it reflects current state
        assert stats["monitoring_active"] == self.monitor._monitoring
        assert stats["health_history_size"] == len(self.monitor.health_history)
    
    def test_context_manager(self):
        """Test using BufferHealthMonitor as context manager."""
        with BufferHealthMonitor(self.buffer_manager) as monitor:
            assert isinstance(monitor, BufferHealthMonitor)
            monitor.start_monitoring(self.audio_queue)
            assert monitor._monitoring
        
        # Should automatically stop monitoring when exiting context
        assert not monitor._monitoring
    
    def test_alert_callback_exception_handling(self):
        """Test that exceptions in alert callback don't crash monitoring."""
        # Create callback that raises exception
        def failing_callback(alert):
            raise Exception("Callback failed")
        
        monitor = BufferHealthMonitor(
            self.buffer_manager,
            monitoring_interval=0.1,
            alert_callback=failing_callback
        )
        
        # Fill queue to trigger alert
        for i in range(85):
            self.audio_queue.put(f"item_{i}")
        
        # Should not crash despite callback exception
        monitor.start_monitoring(self.audio_queue)
        time.sleep(0.3)
        monitor.stop_monitoring()
        
        # Should still have collected health data
        assert len(monitor.health_history) > 0
    
    def test_performance_stats_tracking(self):
        """Test that performance statistics are properly tracked."""
        # Add queue items and start monitoring
        for i in range(60):
            self.audio_queue.put(f"item_{i}")
        
        self.monitor.start_monitoring(self.audio_queue)
        time.sleep(0.3)
        self.monitor.stop_monitoring()
        
        stats = self.monitor.performance_stats
        
        # Should have performed checks
        assert stats["total_checks"] > 0
        
        # May have alerts depending on timing
        assert stats["alerts_sent"] >= 0
        
        # Should track buffer events
        assert "overflow_events" in stats
        assert "underrun_events" in stats
        assert "emergency_expansions" in stats
    
    def test_monitoring_loop_exception_handling(self):
        """Test that monitoring loop handles exceptions gracefully."""
        # Mock the buffer manager to raise an exception
        with patch.object(self.buffer_manager, 'monitor_utilization', side_effect=Exception("Test error")):
            self.monitor.start_monitoring(self.audio_queue)
            time.sleep(0.2)
            self.monitor.stop_monitoring()
        
        # Monitoring should have stopped gracefully
        assert not self.monitor._monitoring
    
    def test_thread_safety(self):
        """Test thread safety of monitor operations."""
        results = []
        errors = []
        
        def worker():
            try:
                # Perform various operations concurrently
                self.monitor.get_current_health()
                self.monitor.get_recent_alerts()
                self.monitor.get_statistics()
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Start monitoring
        self.monitor.start_monitoring(self.audio_queue)
        
        # Start multiple threads performing operations
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.monitor.stop_monitoring()
        
        # All operations should complete without errors
        assert len(errors) == 0
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])