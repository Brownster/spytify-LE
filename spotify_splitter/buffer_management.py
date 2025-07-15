"""
Core buffer management infrastructure for adaptive audio buffering.

This module provides dynamic buffer sizing, utilization monitoring, and
adaptive buffer management to prevent audio dropouts and buffer overruns.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Optional, Deque, Dict, Any
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Buffer health status indicators."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class BufferStrategy(Enum):
    """Buffer management strategies for different scenarios."""
    CONSERVATIVE = "conservative"  # Stability over latency
    BALANCED = "balanced"         # Balance of stability and latency
    LOW_LATENCY = "low_latency"   # Minimal latency, higher risk


@dataclass
class BufferMetrics:
    """Real-time buffer utilization and performance metrics."""
    utilization_percent: float
    queue_size: int
    overflow_count: int
    underrun_count: int
    average_latency_ms: float
    peak_latency_ms: float
    timestamp: datetime
    
    def __post_init__(self):
        # Ensure utilization is within valid range
        self.utilization_percent = max(0.0, min(100.0, self.utilization_percent))


@dataclass
class AudioSettings:
    """Audio configuration settings for adaptive buffer management."""
    queue_size: int
    blocksize: int
    latency: float
    channels: int
    samplerate: int
    buffer_strategy: BufferStrategy
    
    def __post_init__(self):
        # Validate settings
        if self.queue_size < 10:
            raise ValueError("Queue size must be at least 10")
        if self.blocksize < 256:
            raise ValueError("Blocksize must be at least 256")
        if self.latency < 0.001:
            raise ValueError("Latency must be at least 1ms")


@dataclass
class BufferHealth:
    """Comprehensive buffer health assessment."""
    status: HealthStatus
    utilization: float
    overflow_risk: float
    recommended_action: Optional[str]
    metrics: BufferMetrics


class AdaptiveBufferManager:
    """
    Manages dynamic buffer sizing and monitoring for audio streams.
    
    Provides real-time buffer utilization monitoring, adaptive sizing,
    and emergency buffer expansion to prevent audio dropouts.
    """
    
    def __init__(
        self,
        initial_queue_size: int = 200,
        min_size: int = 50,
        max_size: int = 1000,
        adjustment_threshold: float = 0.8,
        emergency_threshold: float = 0.95,
        cooldown_seconds: float = 2.0,
        metrics_collector=None
    ):
        """
        Initialize the adaptive buffer manager.
        
        Args:
            initial_queue_size: Starting buffer size
            min_size: Minimum allowed buffer size
            max_size: Maximum allowed buffer size
            adjustment_threshold: Utilization threshold for adjustments (0.0-1.0)
            emergency_threshold: Utilization threshold for emergency expansion
            cooldown_seconds: Minimum time between adjustments
        """
        if not (min_size <= initial_queue_size <= max_size):
            raise ValueError("initial_queue_size must be between min_size and max_size")
        if not (0.0 < adjustment_threshold < emergency_threshold <= 1.0):
            raise ValueError("Invalid threshold values")
            
        self.current_queue_size = initial_queue_size
        self.min_size = min_size
        self.max_size = max_size
        self.adjustment_threshold = adjustment_threshold
        self.emergency_threshold = emergency_threshold
        self.cooldown_seconds = cooldown_seconds
        self.metrics_collector = metrics_collector
        
        # Monitoring data
        self.utilization_history: Deque[float] = deque(maxlen=100)
        self.latency_history: Deque[float] = deque(maxlen=50)
        self.last_adjustment_time = 0.0
        self.overflow_count = 0
        self.underrun_count = 0
        self.emergency_expansions = 0
        self.adjustment_count = 0
        self.emergency_expansion_count = 0
        
        # Metrics collection
        self.metrics_collector = metrics_collector
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register with metrics collector if provided
        if self.metrics_collector:
            self.metrics_collector.register_component('buffer_manager', self._get_metrics_data)
        
        logger.debug(
            "AdaptiveBufferManager initialized: size=%d, min=%d, max=%d, metrics=%s",
            initial_queue_size, min_size, max_size, metrics_collector is not None
        )
    
    def monitor_utilization(self, queue: Queue) -> BufferMetrics:
        """
        Monitor current queue utilization and collect metrics.
        
        Args:
            queue: The audio queue to monitor
            
        Returns:
            BufferMetrics with current utilization data
        """
        with self._lock:
            current_time = time.time()
            queue_size = queue.qsize()
            max_size = queue.maxsize or self.current_queue_size
            
            # Calculate utilization percentage
            utilization = (queue_size / max_size) * 100.0 if max_size > 0 else 0.0
            
            # Update history
            self.utilization_history.append(utilization)
            
            # Calculate average and peak latency
            avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
            peak_latency = max(self.latency_history) if self.latency_history else 0.0
            
            metrics = BufferMetrics(
                utilization_percent=utilization,
                queue_size=queue_size,
                overflow_count=self.overflow_count,
                underrun_count=self.underrun_count,
                average_latency_ms=avg_latency * 1000,  # Convert to ms
                peak_latency_ms=peak_latency * 1000,
                timestamp=datetime.now()
            )
            
            return metrics
    
    def adjust_buffer_size(self, metrics: BufferMetrics) -> int:
        """
        Dynamically adjust buffer size based on utilization metrics.
        
        Args:
            metrics: Current buffer metrics
            
        Returns:
            New buffer size (may be unchanged)
        """
        with self._lock:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_adjustment_time < self.cooldown_seconds:
                return self.current_queue_size
            
            utilization = metrics.utilization_percent / 100.0
            old_size = self.current_queue_size
            new_size = old_size
            
            # Determine if adjustment is needed
            if utilization >= self.adjustment_threshold:
                # High utilization - increase buffer size
                increase_factor = 1.2 + (utilization - self.adjustment_threshold) * 2
                new_size = min(int(old_size * increase_factor), self.max_size)
                
                if new_size > old_size:
                    logger.info(
                        "Increasing buffer size: %d -> %d (utilization: %.1f%%)",
                        old_size, new_size, utilization * 100
                    )
                    
            elif utilization < 0.3 and len(self.utilization_history) >= 10:
                # Consistently low utilization - consider reducing buffer size
                recent_avg = sum(list(self.utilization_history)[-10:]) / 10
                if recent_avg < 30.0:  # 30% average over last 10 measurements
                    new_size = max(int(old_size * 0.9), self.min_size)
                    
                    if new_size < old_size:
                        logger.info(
                            "Reducing buffer size: %d -> %d (avg utilization: %.1f%%)",
                            old_size, new_size, recent_avg
                        )
            
            # Update if size changed
            if new_size != old_size:
                self.current_queue_size = new_size
                self.last_adjustment_time = current_time
                self.adjustment_count += 1
            
            return self.current_queue_size
    
    def get_optimal_settings(self, system_load: float = 0.5) -> AudioSettings:
        """
        Get optimal audio settings based on current system state.
        
        Args:
            system_load: Current system load factor (0.0-1.0)
            
        Returns:
            AudioSettings optimized for current conditions
        """
        with self._lock:
            # Determine strategy based on system load and history
            if system_load > 0.8 or self.emergency_expansions > 2:
                strategy = BufferStrategy.CONSERVATIVE
                latency = 0.2  # 200ms
                blocksize = 4096
            elif system_load < 0.3 and len(self.utilization_history) >= 5:
                avg_util = sum(list(self.utilization_history)[-5:]) / 5
                if avg_util < 50.0:
                    strategy = BufferStrategy.LOW_LATENCY
                    latency = 0.05  # 50ms
                    blocksize = 1024
                else:
                    strategy = BufferStrategy.BALANCED
                    latency = 0.1  # 100ms
                    blocksize = 2048
            else:
                strategy = BufferStrategy.BALANCED
                latency = 0.1  # 100ms
                blocksize = 2048
            
            return AudioSettings(
                queue_size=self.current_queue_size,
                blocksize=blocksize,
                latency=latency,
                channels=2,  # Stereo default
                samplerate=44100,  # CD quality default
                buffer_strategy=strategy
            )
    
    def emergency_expansion(self) -> bool:
        """
        Perform emergency buffer expansion when overflow is imminent.
        
        Returns:
            True if expansion was performed, False if at maximum
        """
        with self._lock:
            if self.current_queue_size >= self.max_size:
                logger.warning("Cannot expand buffer - already at maximum size (%d)", self.max_size)
                return False
            
            old_size = self.current_queue_size
            # Emergency expansion - increase by 50% or to max, whichever is smaller
            self.current_queue_size = min(int(old_size * 1.5), self.max_size)
            self.emergency_expansions += 1
            self.emergency_expansion_count += 1
            self.last_adjustment_time = time.time()
            
            logger.warning(
                "Emergency buffer expansion: %d -> %d (expansion #%d)",
                old_size, self.current_queue_size, self.emergency_expansions
            )
            
            return True
    
    def get_buffer_health(self, metrics: BufferMetrics) -> BufferHealth:
        """
        Assess overall buffer health and provide recommendations.
        
        Args:
            metrics: Current buffer metrics
            
        Returns:
            BufferHealth assessment with status and recommendations
        """
        utilization = metrics.utilization_percent / 100.0
        
        # Determine health status
        if utilization >= self.emergency_threshold:
            status = HealthStatus.CRITICAL
            overflow_risk = 0.9
            action = "Emergency buffer expansion recommended"
        elif utilization >= self.adjustment_threshold:
            status = HealthStatus.WARNING
            overflow_risk = utilization * 0.8
            action = "Consider increasing buffer size"
        else:
            status = HealthStatus.HEALTHY
            overflow_risk = utilization * 0.5
            action = None
        
        # Additional recommendations based on history
        if metrics.overflow_count > 5:
            action = f"{action or ''} - Multiple overflows detected, increase max buffer size".strip(" -")
        elif len(self.utilization_history) >= 20:
            avg_util = sum(self.utilization_history) / len(self.utilization_history)
            if avg_util < 20.0 and status == HealthStatus.HEALTHY:
                action = "Buffer size could be reduced for lower latency"
        
        return BufferHealth(
            status=status,
            utilization=utilization,
            overflow_risk=overflow_risk,
            recommended_action=action,
            metrics=metrics
        )
    
    def record_overflow(self):
        """Record a buffer overflow event."""
        with self._lock:
            self.overflow_count += 1
            logger.warning("Buffer overflow recorded (total: %d)", self.overflow_count)
    
    def record_underrun(self):
        """Record a buffer underrun event."""
        with self._lock:
            self.underrun_count += 1
            logger.debug("Buffer underrun recorded (total: %d)", self.underrun_count)
    
    def record_latency(self, latency_seconds: float):
        """Record a latency measurement."""
        with self._lock:
            self.latency_history.append(latency_seconds)
    
    def reset_stats(self):
        """Reset all statistics and counters."""
        with self._lock:
            self.utilization_history.clear()
            self.latency_history.clear()
            self.overflow_count = 0
            self.underrun_count = 0
            self.emergency_expansions = 0
            self.last_adjustment_time = 0.0
            logger.info("Buffer manager statistics reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        with self._lock:
            avg_util = sum(self.utilization_history) / len(self.utilization_history) if self.utilization_history else 0.0
            avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
            
            return {
                "current_queue_size": self.current_queue_size,
                "average_utilization": avg_util,
                "average_latency_ms": avg_latency * 1000,
                "overflow_count": self.overflow_count,
                "underrun_count": self.underrun_count,
                "emergency_expansions": self.emergency_expansions,
                "utilization_samples": len(self.utilization_history),
                "latency_samples": len(self.latency_history)
            }
    
    def _get_metrics_data(self) -> Dict[str, Any]:
        """
        Get current metrics data for the metrics collector.
        
        Returns:
            Dictionary of current metric values for collection
        """
        with self._lock:
            # Get basic statistics
            stats = self.get_stats()
            
            # Add real-time metrics
            current_metrics = {
                'current_queue_size': self.current_queue_size,
                'min_size': self.min_size,
                'max_size': self.max_size,
                'overflow_count': self.overflow_count,
                'underrun_count': self.underrun_count,
                'emergency_expansions': self.emergency_expansions,
                'adjustment_threshold': self.adjustment_threshold,
                'emergency_threshold': self.emergency_threshold,
                'utilization_samples': len(self.utilization_history),
                'latency_samples': len(self.latency_history)
            }
            
            # Add calculated averages
            if self.utilization_history:
                current_metrics['average_utilization_percent'] = sum(self.utilization_history) / len(self.utilization_history)
                current_metrics['peak_utilization_percent'] = max(self.utilization_history)
                current_metrics['recent_utilization_percent'] = self.utilization_history[-1] if self.utilization_history else 0.0
            
            if self.latency_history:
                current_metrics['average_latency_ms'] = (sum(self.latency_history) / len(self.latency_history)) * 1000
                current_metrics['peak_latency_ms'] = max(self.latency_history) * 1000
                current_metrics['recent_latency_ms'] = self.latency_history[-1] * 1000 if self.latency_history else 0.0
            
            # Add time since last adjustment
            current_metrics['seconds_since_last_adjustment'] = time.time() - self.last_adjustment_time
            
            return current_metrics