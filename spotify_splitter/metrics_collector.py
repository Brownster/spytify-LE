"""
Comprehensive metrics collection and performance tracking system.

This module provides real-time audio pipeline statistics, detailed error
diagnostics, and performance monitoring for the Spotify Splitter application.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class DiagnosticLevel(Enum):
    """Diagnostic information detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class MetricValue:
    """Individual metric measurement."""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics snapshot."""
    timestamp: datetime
    audio_pipeline: Dict[str, Any]
    buffer_management: Dict[str, Any]
    error_recovery: Dict[str, Any]
    system_resources: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'audio_pipeline': self.audio_pipeline,
            'buffer_management': self.buffer_management,
            'error_recovery': self.error_recovery,
            'system_resources': self.system_resources
        }


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    timestamp: datetime
    level: DiagnosticLevel
    summary: Dict[str, Any]
    metrics: Dict[str, List[MetricValue]]
    performance_snapshots: List[PerformanceSnapshot]
    error_analysis: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'summary': self.summary,
            'metrics': {
                name: [m.to_dict() for m in values]
                for name, values in self.metrics.items()
            },
            'performance_snapshots': [s.to_dict() for s in self.performance_snapshots],
            'error_analysis': self.error_analysis,
            'recommendations': self.recommendations
        }


class MetricsCollector:
    """
    Comprehensive metrics collection and performance tracking system.
    
    Provides real-time collection of audio pipeline statistics, buffer
    performance metrics, error diagnostics, and system resource usage.
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 1000,
        enable_debug_mode: bool = False,
        metrics_file: Optional[Path] = None
    ):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: Time between metric collections in seconds
            history_size: Number of metric values to retain in memory
            enable_debug_mode: Enable detailed debug metrics collection
            metrics_file: Optional file path for persistent metrics storage
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_debug_mode = enable_debug_mode
        self.metrics_file = metrics_file
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Performance snapshots
        self.performance_snapshots: deque = deque(maxlen=100)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: deque = deque(maxlen=200)
        
        # Collection state
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Registered components for metric collection
        self._registered_components: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.collection_stats = {
            'total_collections': 0,
            'failed_collections': 0,
            'metrics_recorded': 0,
            'errors_recorded': 0,
            'start_time': None
        }
        
        logger.debug(
            "MetricsCollector initialized: interval=%.2fs, debug=%s",
            collection_interval, enable_debug_mode
        )
    
    def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self._collecting:
            logger.warning("Metrics collection is already running")
            return
        
        self._collecting = True
        self._stop_event.clear()
        self.collection_stats['start_time'] = datetime.now()
        
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        self._collection_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        if not self._collecting:
            return
        
        self._collecting = False
        self._stop_event.set()
        
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=2.0)
            if self._collection_thread.is_alive():
                logger.warning("Collection thread did not stop gracefully")
        
        # Save final metrics if file is configured
        if self.metrics_file:
            self._save_metrics_to_file()
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        logger.debug("Metrics collection loop started")
        
        try:
            while self._collecting and not self._stop_event.is_set():
                try:
                    self._collect_all_metrics()
                    self.collection_stats['total_collections'] += 1
                    
                except Exception as e:
                    logger.error("Error during metrics collection: %s", e)
                    self.collection_stats['failed_collections'] += 1
                
                # Wait for next collection or stop signal
                if self._stop_event.wait(timeout=self.collection_interval):
                    break
                    
        except Exception as e:
            logger.error("Fatal error in metrics collection loop: %s", e)
        finally:
            logger.debug("Metrics collection loop ended")
    
    def _collect_all_metrics(self) -> None:
        """Collect metrics from all registered components."""
        timestamp = datetime.now()
        
        # Collect from registered components
        component_metrics = {}
        for name, collector_func in self._registered_components.items():
            try:
                metrics = collector_func()
                component_metrics[name] = metrics
                
                # Store individual metrics
                for metric_name, value in metrics.items():
                    full_name = f"{name}.{metric_name}"
                    self._record_metric(full_name, value, timestamp)
                    
            except Exception as e:
                logger.error("Error collecting metrics from %s: %s", name, e)
                self.record_error(f"metrics_collection_{name}", str(e))
        
        # Create performance snapshot
        if component_metrics:
            snapshot = self._create_performance_snapshot(timestamp, component_metrics)
            with self._lock:
                self.performance_snapshots.append(snapshot)
        
        # Collect system metrics if debug mode is enabled
        if self.enable_debug_mode:
            self._collect_system_metrics(timestamp)
    
    def _record_metric(self, name: str, value: Union[int, float], timestamp: datetime) -> None:
        """Record a single metric value."""
        metric_value = MetricValue(timestamp=timestamp, value=value)
        
        with self._lock:
            self.metrics[name].append(metric_value)
            self.collection_stats['metrics_recorded'] += 1
    
    def _create_performance_snapshot(
        self, 
        timestamp: datetime, 
        component_metrics: Dict[str, Dict[str, Any]]
    ) -> PerformanceSnapshot:
        """Create a performance snapshot from collected metrics."""
        return PerformanceSnapshot(
            timestamp=timestamp,
            audio_pipeline=component_metrics.get('audio_stream', {}),
            buffer_management=component_metrics.get('buffer_manager', {}),
            error_recovery=component_metrics.get('error_recovery', {}),
            system_resources=component_metrics.get('system', {})
        )
    
    def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system resource metrics (debug mode only)."""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            self._record_metric('system.cpu_percent', cpu_percent, timestamp)
            self._record_metric('system.memory_percent', memory.percent, timestamp)
            self._record_metric('system.memory_available_mb', memory.available / 1024 / 1024, timestamp)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            self._record_metric('process.memory_rss_mb', process_memory.rss / 1024 / 1024, timestamp)
            self._record_metric('process.memory_vms_mb', process_memory.vms / 1024 / 1024, timestamp)
            self._record_metric('process.cpu_percent', process.cpu_percent(), timestamp)
            self._record_metric('process.num_threads', process.num_threads(), timestamp)
            
        except ImportError:
            # psutil not available - skip system metrics
            pass
        except Exception as e:
            logger.debug("Error collecting system metrics: %s", e)
    
    def register_component(self, name: str, collector_func: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a component for automatic metrics collection.
        
        Args:
            name: Component name (used as metric prefix)
            collector_func: Function that returns a dict of metric_name -> value
        """
        with self._lock:
            self._registered_components[name] = collector_func
        
        logger.debug("Registered metrics collector for component: %s", name)
    
    def unregister_component(self, name: str) -> None:
        """Unregister a component from metrics collection."""
        with self._lock:
            self._registered_components.pop(name, None)
        
        logger.debug("Unregistered metrics collector for component: %s", name)
    
    def record_counter(self, name: str, increment: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += increment
            
            metric_value = MetricValue(
                timestamp=datetime.now(),
                value=self.counters[name],
                labels=labels or {}
            )
            self.metrics[name].append(metric_value)
            self.collection_stats['metrics_recorded'] += 1
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            
            metric_value = MetricValue(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self.metrics[name].append(metric_value)
            self.collection_stats['metrics_recorded'] += 1
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration in seconds)."""
        with self._lock:
            self.timers[name].append(duration)
            
            # Keep only recent timer values
            if len(self.timers[name]) > 100:
                self.timers[name] = self.timers[name][-100:]
            
            metric_value = MetricValue(
                timestamp=datetime.now(),
                value=duration,
                labels=labels or {}
            )
            self.metrics[name].append(metric_value)
            self.collection_stats['metrics_recorded'] += 1
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error for diagnostic purposes."""
        with self._lock:
            self.error_counts[error_type] += 1
            
            error_record = {
                'timestamp': datetime.now(),
                'type': error_type,
                'message': error_message,
                'context': context or {},
                'count': self.error_counts[error_type]
            }
            
            self.error_history.append(error_record)
            self.collection_stats['errors_recorded'] += 1
        
        logger.debug("Recorded error: %s - %s", error_type, error_message)
    
    def get_metric_values(self, name: str, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric values, optionally filtered by time."""
        with self._lock:
            values = list(self.metrics.get(name, []))
        
        if since:
            values = [v for v in values if v.timestamp >= since]
        
        return values
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        values = self.get_metric_values(name, since)
        
        if not values:
            return {'count': 0, 'error': 'No data available'}
        
        numeric_values = [v.value for v in values]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'average': sum(numeric_values) / len(numeric_values),
            'latest': numeric_values[-1],
            'first_timestamp': values[0].timestamp.isoformat(),
            'latest_timestamp': values[-1].timestamp.isoformat()
        }
    
    def get_timer_statistics(self, name: str) -> Dict[str, float]:
        """Get statistical analysis of timer metrics."""
        with self._lock:
            durations = self.timers.get(name, [])
        
        if not durations:
            return {'count': 0, 'error': 'No timer data available'}
        
        durations_sorted = sorted(durations)
        count = len(durations)
        
        # Calculate median correctly for both odd and even length arrays
        if count % 2 == 0:
            # Even number of elements - average of middle two
            median = (durations_sorted[count // 2 - 1] + durations_sorted[count // 2]) / 2
        else:
            # Odd number of elements - middle element
            median = durations_sorted[count // 2]
        
        return {
            'count': count,
            'min': min(durations),
            'max': max(durations),
            'average': sum(durations) / count,
            'median': median,
            'p95': durations_sorted[int(count * 0.95)] if count > 20 else durations_sorted[-1],
            'p99': durations_sorted[int(count * 0.99)] if count > 100 else durations_sorted[-1]
        }
    
    def get_error_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of recorded errors."""
        with self._lock:
            errors = list(self.error_history)
        
        if since:
            errors = [e for e in errors if e['timestamp'] >= since]
        
        # Count by error type
        error_type_counts = defaultdict(int)
        for error in errors:
            error_type_counts[error['type']] += 1
        
        return {
            'total_errors': len(errors),
            'unique_error_types': len(error_type_counts),
            'error_type_counts': dict(error_type_counts),
            'recent_errors': errors[-10:] if errors else []
        }
    
    def generate_diagnostic_report(self, level: DiagnosticLevel = DiagnosticLevel.BASIC) -> DiagnosticReport:
        """Generate a comprehensive diagnostic report."""
        timestamp = datetime.now()
        
        # Summary statistics
        summary = {
            'collection_uptime_seconds': (
                (timestamp - self.collection_stats['start_time']).total_seconds()
                if self.collection_stats['start_time'] else 0
            ),
            'total_metrics': len(self.metrics),
            'total_snapshots': len(self.performance_snapshots),
            'collection_stats': self.collection_stats.copy()
        }
        
        # Get metrics based on detail level
        if level == DiagnosticLevel.BASIC:
            # Only key metrics for basic report
            key_metrics = ['audio_stream.total_frames', 'buffer_manager.overflow_count', 'error_recovery.total_errors']
            metrics = {name: list(self.metrics.get(name, [])) for name in key_metrics if name in self.metrics}
            snapshots = list(self.performance_snapshots)[-10:]  # Last 10 snapshots
        elif level == DiagnosticLevel.DETAILED:
            # More comprehensive metrics
            metrics = {name: list(values) for name, values in self.metrics.items()}
            snapshots = list(self.performance_snapshots)[-50:]  # Last 50 snapshots
        else:  # DEBUG
            # All metrics and snapshots
            metrics = {name: list(values) for name, values in self.metrics.items()}
            snapshots = list(self.performance_snapshots)
        
        # Error analysis
        error_analysis = self.get_error_summary(since=timestamp - timedelta(hours=1))
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return DiagnosticReport(
            timestamp=timestamp,
            level=level,
            summary=summary,
            metrics=metrics,
            performance_snapshots=snapshots,
            error_analysis=error_analysis,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on collected metrics."""
        recommendations = []
        
        # Check for high error rates
        error_summary = self.get_error_summary(since=datetime.now() - timedelta(minutes=30))
        if error_summary['total_errors'] > 10:
            recommendations.append(
                f"High error rate detected: {error_summary['total_errors']} errors in last 30 minutes. "
                "Consider investigating error recovery mechanisms."
            )
        
        # Check buffer overflow patterns
        overflow_metrics = self.get_metric_values('buffer_manager.overflow_count')
        if overflow_metrics and len(overflow_metrics) > 5:
            recent_overflows = overflow_metrics[-5:]
            if any(m.value > 0 for m in recent_overflows):
                recommendations.append(
                    "Buffer overflows detected. Consider increasing buffer sizes or "
                    "enabling adaptive buffer management."
                )
        
        # Check for performance degradation
        if len(self.performance_snapshots) > 10:
            recent_snapshots = list(self.performance_snapshots)[-10:]
            cpu_values = [s.system_resources.get('cpu_percent', 0) for s in recent_snapshots]
            if cpu_values and sum(cpu_values) / len(cpu_values) > 80:
                recommendations.append(
                    "High CPU usage detected. Consider optimizing audio processing "
                    "or reducing buffer collection frequency."
                )
        
        # Check collection health
        if self.collection_stats['failed_collections'] > self.collection_stats['total_collections'] * 0.1:
            recommendations.append(
                "High metrics collection failure rate. Consider reducing collection "
                "frequency or disabling debug mode."
            )
        
        return recommendations
    
    def _save_metrics_to_file(self) -> None:
        """Save current metrics to file."""
        if not self.metrics_file:
            return
        
        try:
            # Create directory if it doesn't exist
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate report and save as JSON
            report = self.generate_diagnostic_report(DiagnosticLevel.DETAILED)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info("Metrics saved to %s", self.metrics_file)
            
        except Exception as e:
            logger.error("Error saving metrics to file: %s", e)
    
    def export_metrics(self, filepath: Path, format: str = 'json') -> None:
        """Export metrics to file in specified format."""
        try:
            report = self.generate_diagnostic_report(DiagnosticLevel.DEBUG)
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info("Metrics exported to %s", filepath)
            
        except Exception as e:
            logger.error("Error exporting metrics: %s", e)
            raise
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics and reset counters."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()
            self.performance_snapshots.clear()
            self.error_history.clear()
            self.error_counts.clear()
            
            # Reset collection stats but preserve start time
            start_time = self.collection_stats.get('start_time')
            self.collection_stats = {
                'total_collections': 0,
                'failed_collections': 0,
                'metrics_recorded': 0,
                'errors_recorded': 0,
                'start_time': start_time
            }
        
        logger.info("All metrics cleared")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the metrics collector."""
        with self._lock:
            return {
                'collecting': self._collecting,
                'collection_interval': self.collection_interval,
                'debug_mode': self.enable_debug_mode,
                'registered_components': list(self._registered_components.keys()),
                'metrics_count': len(self.metrics),
                'snapshots_count': len(self.performance_snapshots),
                'errors_count': len(self.error_history),
                'collection_stats': self.collection_stats.copy(),
                'memory_usage': {
                    'metrics_memory_mb': sum(
                        len(values) * 100  # Rough estimate
                        for values in self.metrics.values()
                    ) / 1024 / 1024,
                    'snapshots_memory_mb': len(self.performance_snapshots) * 1024 / 1024 / 1024  # Rough estimate
                }
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_collection()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, timer_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics_collector = metrics_collector
        self.timer_name = timer_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(self.timer_name, duration, self.labels)


def timer(metrics_collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimerContext(metrics_collector, name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator