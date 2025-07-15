"""
Real-time buffer health monitoring and early warning system.

This module provides continuous monitoring of buffer health, early warning
notifications for potential overflow conditions, and comprehensive metrics
collection for buffer performance analysis.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Queue
from typing import Callable, Optional, List, Dict, Any
from enum import Enum

from .buffer_management import (
    AdaptiveBufferManager,
    BufferMetrics,
    BufferHealth,
    HealthStatus
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for buffer health notifications."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BufferAlert:
    """Buffer health alert notification."""
    level: AlertLevel
    message: str
    timestamp: datetime
    metrics: BufferMetrics
    recommended_action: Optional[str] = None


@dataclass
class HealthReport:
    """Comprehensive buffer health report."""
    timestamp: datetime
    current_health: BufferHealth
    alerts: List[BufferAlert]
    performance_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]


class BufferHealthMonitor:
    """
    Real-time buffer health monitoring and alerting system.
    
    Provides continuous monitoring of buffer utilization, early warning
    notifications, and comprehensive health reporting for audio buffer
    management systems.
    """
    
    def __init__(
        self,
        buffer_manager: AdaptiveBufferManager,
        monitoring_interval: float = 0.5,
        alert_callback: Optional[Callable[[BufferAlert], None]] = None,
        history_size: int = 1000
    ):
        """
        Initialize the buffer health monitor.
        
        Args:
            buffer_manager: The buffer manager to monitor
            monitoring_interval: Time between health checks in seconds
            alert_callback: Optional callback for alert notifications
            history_size: Number of health records to maintain in history
        """
        self.buffer_manager = buffer_manager
        self.monitoring_interval = monitoring_interval
        self.alert_callback = alert_callback
        self.history_size = history_size
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Health history and alerts
        self.health_history: deque = deque(maxlen=history_size)
        self.alert_history: deque = deque(maxlen=100)
        self.last_alert_time: Dict[AlertLevel, datetime] = {}
        
        # Alert thresholds and cooldowns
        self.alert_thresholds = {
            AlertLevel.WARNING: 0.8,  # 80% utilization
            AlertLevel.CRITICAL: 0.95  # 95% utilization
        }
        self.alert_cooldowns = {
            AlertLevel.INFO: timedelta(minutes=1),
            AlertLevel.WARNING: timedelta(minutes=2),
            AlertLevel.CRITICAL: timedelta(seconds=30)
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_checks": 0,
            "alerts_sent": 0,
            "overflow_events": 0,
            "underrun_events": 0,
            "emergency_expansions": 0
        }
        
        logger.debug("BufferHealthMonitor initialized with interval=%.2fs", monitoring_interval)
    
    def start_monitoring(self, audio_queue: Queue) -> None:
        """
        Start real-time buffer health monitoring.
        
        Args:
            audio_queue: The audio queue to monitor
        """
        if self._monitoring:
            logger.warning("Buffer health monitoring is already running")
            return
        
        self.audio_queue = audio_queue
        self._monitoring = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="BufferHealthMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Buffer health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time buffer health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            if self._monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully")
        
        logger.info("Buffer health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        logger.debug("Buffer health monitoring loop started")
        
        try:
            while self._monitoring and not self._stop_event.is_set():
                try:
                    self._perform_health_check()
                    self.performance_stats["total_checks"] += 1
                    
                except Exception as e:
                    logger.error("Error during health check: %s", e)
                
                # Wait for next check or stop signal
                if self._stop_event.wait(timeout=self.monitoring_interval):
                    break
                    
        except Exception as e:
            logger.error("Fatal error in monitoring loop: %s", e)
        finally:
            logger.debug("Buffer health monitoring loop ended")
    
    def _perform_health_check(self) -> None:
        """Perform a single health check and process results."""
        if not hasattr(self, 'audio_queue'):
            return
        
        # Get current metrics
        metrics = self.buffer_manager.monitor_utilization(self.audio_queue)
        health = self.buffer_manager.get_buffer_health(metrics)
        
        # Store in history
        self.health_history.append({
            'timestamp': datetime.now(),
            'health': health,
            'metrics': metrics
        })
        
        # Check for alerts
        self._check_and_send_alerts(health, metrics)
        
        # Update performance statistics
        self._update_performance_stats(metrics)
    
    def _check_and_send_alerts(self, health: BufferHealth, metrics: BufferMetrics) -> None:
        """Check health status and send alerts if necessary."""
        current_time = datetime.now()
        utilization = health.utilization
        
        # Determine alert level based on utilization
        alert_level = None
        if utilization >= self.alert_thresholds[AlertLevel.CRITICAL]:
            alert_level = AlertLevel.CRITICAL
        elif utilization >= self.alert_thresholds[AlertLevel.WARNING]:
            alert_level = AlertLevel.WARNING
        elif health.status == HealthStatus.HEALTHY and utilization < 0.3:
            # Info alert for very low utilization (potential optimization opportunity)
            if len(self.health_history) >= 10:
                recent_avg = sum(h['health'].utilization for h in list(self.health_history)[-10:]) / 10
                if recent_avg < 0.3:
                    alert_level = AlertLevel.INFO
        
        if alert_level is None:
            return
        
        # Check cooldown period
        last_alert = self.last_alert_time.get(alert_level)
        if last_alert and (current_time - last_alert) < self.alert_cooldowns[alert_level]:
            return
        
        # Create and send alert
        alert = self._create_alert(alert_level, health, metrics)
        self._send_alert(alert)
        
        # Update last alert time
        self.last_alert_time[alert_level] = current_time
    
    def _create_alert(self, level: AlertLevel, health: BufferHealth, metrics: BufferMetrics) -> BufferAlert:
        """Create an alert based on current health status."""
        utilization_pct = health.utilization * 100
        
        if level == AlertLevel.CRITICAL:
            message = f"CRITICAL: Buffer utilization at {utilization_pct:.1f}% - overflow imminent!"
            action = "Emergency buffer expansion recommended immediately"
        elif level == AlertLevel.WARNING:
            message = f"WARNING: High buffer utilization at {utilization_pct:.1f}%"
            action = health.recommended_action or "Consider increasing buffer size"
        else:  # INFO
            message = f"INFO: Low buffer utilization at {utilization_pct:.1f}% - optimization opportunity"
            action = "Buffer size could be reduced for lower latency"
        
        return BufferAlert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            recommended_action=action
        )
    
    def _send_alert(self, alert: BufferAlert) -> None:
        """Send alert notification."""
        self.alert_history.append(alert)
        self.performance_stats["alerts_sent"] += 1
        
        # Log the alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[alert.level]
        
        logger.log(log_level, "Buffer Health Alert: %s", alert.message)
        
        # Call external callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error("Error in alert callback: %s", e)
    
    def _update_performance_stats(self, metrics: BufferMetrics) -> None:
        """Update performance statistics based on current metrics."""
        # Track overflow and underrun events
        if metrics.overflow_count > self.performance_stats["overflow_events"]:
            self.performance_stats["overflow_events"] = metrics.overflow_count
        
        if metrics.underrun_count > self.performance_stats["underrun_events"]:
            self.performance_stats["underrun_events"] = metrics.underrun_count
        
        # Track emergency expansions
        current_expansions = self.buffer_manager.emergency_expansions
        if current_expansions > self.performance_stats["emergency_expansions"]:
            self.performance_stats["emergency_expansions"] = current_expansions
    
    def get_current_health(self) -> Optional[BufferHealth]:
        """Get the most recent buffer health assessment."""
        if not self.health_history:
            return None
        
        return self.health_history[-1]['health']
    
    def get_recent_alerts(self, minutes: int = 10) -> List[BufferAlert]:
        """Get alerts from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def generate_health_report(self) -> HealthReport:
        """Generate a comprehensive buffer health report."""
        current_time = datetime.now()
        current_health = self.get_current_health()
        recent_alerts = self.get_recent_alerts(minutes=30)
        
        # Performance summary
        performance_summary = {
            **self.performance_stats,
            "monitoring_uptime_seconds": (
                (current_time - self.health_history[0]['timestamp']).total_seconds()
                if self.health_history else 0
            ),
            "average_check_interval": (
                self.monitoring_interval if self.performance_stats["total_checks"] > 0 else 0
            ),
            "alert_rate_per_hour": (
                (self.performance_stats["alerts_sent"] / 
                 max(1, self.performance_stats["total_checks"])) * 3600 / self.monitoring_interval
                if self.performance_stats["total_checks"] > 0 else 0
            )
        }
        
        # Trend analysis
        trend_analysis = self._analyze_trends()
        
        return HealthReport(
            timestamp=current_time,
            current_health=current_health,
            alerts=recent_alerts,
            performance_summary=performance_summary,
            trend_analysis=trend_analysis
        )
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze buffer health trends from historical data."""
        if len(self.health_history) < 10:
            return {"status": "insufficient_data", "message": "Not enough data for trend analysis"}
        
        # Get recent data (last 50 samples or all if less)
        recent_data = list(self.health_history)[-50:]
        
        # Calculate utilization trend
        utilizations = [h['health'].utilization for h in recent_data]
        if len(utilizations) >= 2:
            # Simple linear trend calculation
            x_vals = list(range(len(utilizations)))
            n = len(utilizations)
            sum_x = sum(x_vals)
            sum_y = sum(utilizations)
            sum_xy = sum(x * y for x, y in zip(x_vals, utilizations))
            sum_x2 = sum(x * x for x in x_vals)
            
            # Calculate slope (trend)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        else:
            trend_direction = "unknown"
            slope = 0.0
        
        # Calculate statistics
        avg_utilization = sum(utilizations) / len(utilizations)
        max_utilization = max(utilizations)
        min_utilization = min(utilizations)
        
        # Health status distribution
        status_counts = {}
        for h in recent_data:
            status = h['health'].status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "status": "analyzed",
            "utilization_trend": {
                "direction": trend_direction,
                "slope": slope,
                "average": avg_utilization,
                "maximum": max_utilization,
                "minimum": min_utilization
            },
            "health_distribution": status_counts,
            "samples_analyzed": len(recent_data),
            "time_span_minutes": (
                (recent_data[-1]['timestamp'] - recent_data[0]['timestamp']).total_seconds() / 60
                if len(recent_data) > 1 else 0
            )
        }
    
    def set_alert_threshold(self, level: AlertLevel, threshold: float) -> None:
        """Set custom alert threshold for a specific level."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.alert_thresholds[level] = threshold
        logger.info("Alert threshold for %s set to %.2f", level.value, threshold)
    
    def set_alert_cooldown(self, level: AlertLevel, cooldown: timedelta) -> None:
        """Set custom alert cooldown for a specific level."""
        self.alert_cooldowns[level] = cooldown
        logger.info("Alert cooldown for %s set to %s", level.value, cooldown)
    
    def clear_alert_history(self) -> None:
        """Clear all alert history."""
        self.alert_history.clear()
        self.last_alert_time.clear()
        logger.info("Alert history cleared")
    
    def clear_health_history(self) -> None:
        """Clear all health history."""
        self.health_history.clear()
        logger.info("Health history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return {
            "monitoring_active": self._monitoring,
            "health_history_size": len(self.health_history),
            "alert_history_size": len(self.alert_history),
            "performance_stats": self.performance_stats.copy(),
            "alert_thresholds": {k.value: v for k, v in self.alert_thresholds.items()},
            "alert_cooldowns": {k.value: str(v) for k, v in self.alert_cooldowns.items()}
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure monitoring is stopped."""
        self.stop_monitoring()