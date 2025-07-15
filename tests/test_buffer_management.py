"""
Unit tests for the adaptive buffer management system.

Tests cover buffer metrics, adaptive sizing, emergency expansion,
health monitoring, and edge cases.
"""

import pytest
import time
from datetime import datetime
from queue import Queue
from unittest.mock import Mock, patch

from spotify_splitter.buffer_management import (
    AdaptiveBufferManager,
    BufferMetrics,
    AudioSettings,
    BufferHealth,
    HealthStatus,
    BufferStrategy
)


class TestBufferMetrics:
    """Test BufferMetrics data model."""
    
    def test_buffer_metrics_creation(self):
        """Test basic BufferMetrics creation and validation."""
        metrics = BufferMetrics(
            utilization_percent=75.5,
            queue_size=150,
            overflow_count=2,
            underrun_count=1,
            average_latency_ms=50.0,
            peak_latency_ms=100.0,
            timestamp=datetime.now()
        )
        
        assert metrics.utilization_percent == 75.5
        assert metrics.queue_size == 150
        assert metrics.overflow_count == 2
        assert metrics.underrun_count == 1
        assert metrics.average_latency_ms == 50.0
        assert metrics.peak_latency_ms == 100.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_buffer_metrics_utilization_clamping(self):
        """Test that utilization percentage is clamped to valid range."""
        # Test over 100%
        metrics = BufferMetrics(
            utilization_percent=150.0,
            queue_size=100,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=0.0,
            peak_latency_ms=0.0,
            timestamp=datetime.now()
        )
        assert metrics.utilization_percent == 100.0
        
        # Test under 0%
        metrics = BufferMetrics(
            utilization_percent=-25.0,
            queue_size=100,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=0.0,
            peak_latency_ms=0.0,
            timestamp=datetime.now()
        )
        assert metrics.utilization_percent == 0.0


class TestAudioSettings:
    """Test AudioSettings data model."""
    
    def test_audio_settings_creation(self):
        """Test basic AudioSettings creation."""
        settings = AudioSettings(
            queue_size=200,
            blocksize=2048,
            latency=0.1,
            channels=2,
            samplerate=44100,
            buffer_strategy=BufferStrategy.BALANCED
        )
        
        assert settings.queue_size == 200
        assert settings.blocksize == 2048
        assert settings.latency == 0.1
        assert settings.channels == 2
        assert settings.samplerate == 44100
        assert settings.buffer_strategy == BufferStrategy.BALANCED
    
    def test_audio_settings_validation(self):
        """Test AudioSettings validation rules."""
        # Test invalid queue size
        with pytest.raises(ValueError, match="Queue size must be at least 10"):
            AudioSettings(
                queue_size=5,
                blocksize=2048,
                latency=0.1,
                channels=2,
                samplerate=44100,
                buffer_strategy=BufferStrategy.BALANCED
            )
        
        # Test invalid blocksize
        with pytest.raises(ValueError, match="Blocksize must be at least 256"):
            AudioSettings(
                queue_size=200,
                blocksize=128,
                latency=0.1,
                channels=2,
                samplerate=44100,
                buffer_strategy=BufferStrategy.BALANCED
            )
        
        # Test invalid latency
        with pytest.raises(ValueError, match="Latency must be at least 1ms"):
            AudioSettings(
                queue_size=200,
                blocksize=2048,
                latency=0.0005,
                channels=2,
                samplerate=44100,
                buffer_strategy=BufferStrategy.BALANCED
            )


class TestAdaptiveBufferManager:
    """Test AdaptiveBufferManager functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        manager = AdaptiveBufferManager(
            initial_queue_size=200,
            min_size=50,
            max_size=1000
        )
        
        assert manager.current_queue_size == 200
        assert manager.min_size == 50
        assert manager.max_size == 1000
        assert len(manager.utilization_history) == 0
        assert manager.overflow_count == 0
        assert manager.underrun_count == 0
    
    def test_initialization_validation(self):
        """Test initialization parameter validation."""
        # Test invalid size range
        with pytest.raises(ValueError, match="initial_queue_size must be between min_size and max_size"):
            AdaptiveBufferManager(initial_queue_size=100, min_size=200, max_size=300)
        
        # Test invalid thresholds
        with pytest.raises(ValueError, match="Invalid threshold values"):
            AdaptiveBufferManager(adjustment_threshold=0.9, emergency_threshold=0.8)
    
    def test_monitor_utilization(self):
        """Test queue utilization monitoring."""
        manager = AdaptiveBufferManager(initial_queue_size=100)
        queue = Queue(maxsize=100)
        
        # Add some items to queue
        for i in range(75):
            queue.put(f"item_{i}")
        
        metrics = manager.monitor_utilization(queue)
        
        assert metrics.queue_size == 75
        assert metrics.utilization_percent == 75.0
        assert len(manager.utilization_history) == 1
        assert manager.utilization_history[0] == 75.0
    
    def test_monitor_utilization_empty_queue(self):
        """Test monitoring with empty queue."""
        manager = AdaptiveBufferManager()
        queue = Queue(maxsize=100)
        
        metrics = manager.monitor_utilization(queue)
        
        assert metrics.queue_size == 0
        assert metrics.utilization_percent == 0.0
    
    def test_adjust_buffer_size_increase(self):
        """Test buffer size increase on high utilization."""
        manager = AdaptiveBufferManager(
            initial_queue_size=100,
            adjustment_threshold=0.8,
            cooldown_seconds=0.1
        )
        
        # Create high utilization metrics
        metrics = BufferMetrics(
            utilization_percent=85.0,
            queue_size=85,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=50.0,
            peak_latency_ms=100.0,
            timestamp=datetime.now()
        )
        
        new_size = manager.adjust_buffer_size(metrics)
        
        assert new_size > 100
        assert new_size <= manager.max_size
        assert manager.current_queue_size == new_size
    
    def test_adjust_buffer_size_decrease(self):
        """Test buffer size decrease on consistently low utilization."""
        manager = AdaptiveBufferManager(initial_queue_size=200)
        
        # Fill history with low utilization values
        for _ in range(15):
            manager.utilization_history.append(25.0)
        
        # Create low utilization metrics
        metrics = BufferMetrics(
            utilization_percent=25.0,
            queue_size=50,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=20.0,
            peak_latency_ms=40.0,
            timestamp=datetime.now()
        )
        
        # Wait for cooldown
        time.sleep(0.1)
        new_size = manager.adjust_buffer_size(metrics)
        
        assert new_size < 200
        assert new_size >= manager.min_size
    
    def test_adjust_buffer_size_cooldown(self):
        """Test that adjustments respect cooldown period."""
        manager = AdaptiveBufferManager(cooldown_seconds=1.0)
        original_size = manager.current_queue_size
        
        # Create high utilization metrics
        metrics = BufferMetrics(
            utilization_percent=90.0,
            queue_size=180,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=50.0,
            peak_latency_ms=100.0,
            timestamp=datetime.now()
        )
        
        # First adjustment should work
        new_size1 = manager.adjust_buffer_size(metrics)
        assert new_size1 != original_size
        
        # Second adjustment immediately should be blocked by cooldown
        new_size2 = manager.adjust_buffer_size(metrics)
        assert new_size2 == new_size1
    
    def test_emergency_expansion(self):
        """Test emergency buffer expansion."""
        manager = AdaptiveBufferManager(
            initial_queue_size=100,
            max_size=500
        )
        
        original_size = manager.current_queue_size
        result = manager.emergency_expansion()
        
        assert result is True
        assert manager.current_queue_size > original_size
        assert manager.emergency_expansions == 1
    
    def test_emergency_expansion_at_max(self):
        """Test emergency expansion when already at maximum size."""
        manager = AdaptiveBufferManager(
            initial_queue_size=500,
            max_size=500
        )
        
        result = manager.emergency_expansion()
        
        assert result is False
        assert manager.current_queue_size == 500
        assert manager.emergency_expansions == 0
    
    def test_get_optimal_settings_conservative(self):
        """Test optimal settings for high system load (conservative strategy)."""
        manager = AdaptiveBufferManager()
        
        settings = manager.get_optimal_settings(system_load=0.9)
        
        assert settings.buffer_strategy == BufferStrategy.CONSERVATIVE
        assert settings.latency >= 0.2  # High latency for stability
        assert settings.blocksize >= 4096  # Large blocks
        assert settings.queue_size == manager.current_queue_size
    
    def test_get_optimal_settings_low_latency(self):
        """Test optimal settings for low system load (low latency strategy)."""
        manager = AdaptiveBufferManager()
        
        # Fill history with low utilization
        for _ in range(10):
            manager.utilization_history.append(30.0)
        
        settings = manager.get_optimal_settings(system_load=0.2)
        
        assert settings.buffer_strategy == BufferStrategy.LOW_LATENCY
        assert settings.latency <= 0.05  # Low latency
        assert settings.blocksize <= 1024  # Small blocks
    
    def test_get_optimal_settings_balanced(self):
        """Test optimal settings for moderate system load (balanced strategy)."""
        manager = AdaptiveBufferManager()
        
        settings = manager.get_optimal_settings(system_load=0.5)
        
        assert settings.buffer_strategy == BufferStrategy.BALANCED
        assert 0.05 < settings.latency < 0.2  # Moderate latency
        assert 1024 < settings.blocksize < 4096  # Moderate blocks
    
    def test_get_buffer_health_healthy(self):
        """Test buffer health assessment for healthy conditions."""
        manager = AdaptiveBufferManager(adjustment_threshold=0.8)
        
        metrics = BufferMetrics(
            utilization_percent=50.0,
            queue_size=100,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=30.0,
            peak_latency_ms=60.0,
            timestamp=datetime.now()
        )
        
        health = manager.get_buffer_health(metrics)
        
        assert health.status == HealthStatus.HEALTHY
        assert health.utilization == 0.5
        assert health.overflow_risk < 0.5
        assert health.recommended_action is None
    
    def test_get_buffer_health_warning(self):
        """Test buffer health assessment for warning conditions."""
        manager = AdaptiveBufferManager(adjustment_threshold=0.8)
        
        metrics = BufferMetrics(
            utilization_percent=85.0,
            queue_size=170,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=80.0,
            peak_latency_ms=150.0,
            timestamp=datetime.now()
        )
        
        health = manager.get_buffer_health(metrics)
        
        assert health.status == HealthStatus.WARNING
        assert health.utilization == 0.85
        assert health.overflow_risk > 0.5
        assert "increasing buffer size" in health.recommended_action.lower()
    
    def test_get_buffer_health_critical(self):
        """Test buffer health assessment for critical conditions."""
        manager = AdaptiveBufferManager(emergency_threshold=0.95)
        
        metrics = BufferMetrics(
            utilization_percent=96.0,
            queue_size=192,
            overflow_count=0,
            underrun_count=0,
            average_latency_ms=120.0,
            peak_latency_ms=200.0,
            timestamp=datetime.now()
        )
        
        health = manager.get_buffer_health(metrics)
        
        assert health.status == HealthStatus.CRITICAL
        assert health.utilization == 0.96
        assert health.overflow_risk >= 0.9
        assert "emergency" in health.recommended_action.lower()
    
    def test_record_overflow(self):
        """Test overflow event recording."""
        manager = AdaptiveBufferManager()
        
        assert manager.overflow_count == 0
        
        manager.record_overflow()
        assert manager.overflow_count == 1
        
        manager.record_overflow()
        assert manager.overflow_count == 2
    
    def test_record_underrun(self):
        """Test underrun event recording."""
        manager = AdaptiveBufferManager()
        
        assert manager.underrun_count == 0
        
        manager.record_underrun()
        assert manager.underrun_count == 1
        
        manager.record_underrun()
        assert manager.underrun_count == 2
    
    def test_record_latency(self):
        """Test latency measurement recording."""
        manager = AdaptiveBufferManager()
        
        assert len(manager.latency_history) == 0
        
        manager.record_latency(0.05)
        assert len(manager.latency_history) == 1
        assert manager.latency_history[0] == 0.05
        
        manager.record_latency(0.08)
        assert len(manager.latency_history) == 2
        assert manager.latency_history[1] == 0.08
    
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        manager = AdaptiveBufferManager()
        
        # Add some data
        manager.utilization_history.extend([50.0, 60.0, 70.0])
        manager.latency_history.extend([0.05, 0.06, 0.07])
        manager.overflow_count = 5
        manager.underrun_count = 3
        manager.emergency_expansions = 2
        
        manager.reset_stats()
        
        assert len(manager.utilization_history) == 0
        assert len(manager.latency_history) == 0
        assert manager.overflow_count == 0
        assert manager.underrun_count == 0
        assert manager.emergency_expansions == 0
    
    def test_get_stats(self):
        """Test statistics summary retrieval."""
        manager = AdaptiveBufferManager(initial_queue_size=150)
        
        # Add some test data
        manager.utilization_history.extend([40.0, 50.0, 60.0])
        manager.latency_history.extend([0.04, 0.05, 0.06])
        manager.overflow_count = 2
        manager.underrun_count = 1
        manager.emergency_expansions = 1
        
        stats = manager.get_stats()
        
        assert stats["current_queue_size"] == 150
        assert stats["average_utilization"] == 50.0  # (40+50+60)/3
        assert abs(stats["average_latency_ms"] - 50.0) < 0.001  # (0.04+0.05+0.06)/3 * 1000
        assert stats["overflow_count"] == 2
        assert stats["underrun_count"] == 1
        assert stats["emergency_expansions"] == 1
        assert stats["utilization_samples"] == 3
        assert stats["latency_samples"] == 3
    
    def test_thread_safety(self):
        """Test thread safety of buffer manager operations."""
        import threading
        import time
        
        manager = AdaptiveBufferManager()
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    manager.record_overflow()
                    manager.record_latency(0.05 + i * 0.01)
                    manager.utilization_history.append(50.0 + i)
                    time.sleep(0.001)  # Small delay to encourage race conditions
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check that all operations completed without errors
        assert len(errors) == 0
        assert len(results) == 5
        assert manager.overflow_count == 50  # 5 threads * 10 operations each
        assert len(manager.latency_history) == 50
        assert len(manager.utilization_history) == 50


class TestBufferHealth:
    """Test BufferHealth data model."""
    
    def test_buffer_health_creation(self):
        """Test BufferHealth creation with all fields."""
        metrics = BufferMetrics(
            utilization_percent=75.0,
            queue_size=150,
            overflow_count=1,
            underrun_count=0,
            average_latency_ms=50.0,
            peak_latency_ms=100.0,
            timestamp=datetime.now()
        )
        
        health = BufferHealth(
            status=HealthStatus.WARNING,
            utilization=0.75,
            overflow_risk=0.6,
            recommended_action="Consider increasing buffer size",
            metrics=metrics
        )
        
        assert health.status == HealthStatus.WARNING
        assert health.utilization == 0.75
        assert health.overflow_risk == 0.6
        assert health.recommended_action == "Consider increasing buffer size"
        assert health.metrics == metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])