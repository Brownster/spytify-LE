"""
Real-time performance monitoring dashboard for audio pipeline.

This module provides a comprehensive performance monitoring interface
that displays real-time metrics, buffer health, and system performance.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.console import Console

from .metrics_collector import MetricsCollector, DiagnosticLevel

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class DashboardConfig:
    """Configuration for the performance dashboard."""
    update_interval: float = 1.0
    history_size: int = 100
    alert_thresholds: Dict[str, Dict[str, float]] = None
    enable_alerts: bool = True
    enable_recommendations: bool = True
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'buffer_utilization': {'warning': 80.0, 'critical': 95.0},
                'error_rate': {'warning': 5.0, 'critical': 10.0},
                'cpu_usage': {'warning': 80.0, 'critical': 95.0},
                'memory_usage': {'warning': 85.0, 'critical': 95.0},
                'latency_ms': {'warning': 100.0, 'critical': 200.0}
            }


class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard.
    
    Provides comprehensive monitoring of audio pipeline performance,
    buffer health, system resources, and error rates with real-time
    alerts and optimization recommendations.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        config: Optional[DashboardConfig] = None,
        ui_callback: Optional[Callable] = None
    ):
        """
        Initialize the performance dashboard.
        
        Args:
            metrics_collector: MetricsCollector instance for data source
            config: Dashboard configuration
            ui_callback: Optional callback for UI updates
        """
        self.metrics_collector = metrics_collector
        self.config = config or DashboardConfig()
        self.ui_callback = ui_callback
        
        # Dashboard state
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance data
        self.current_metrics: Dict[str, Any] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.history_size))
        self.alerts: deque = deque(maxlen=50)
        self.recommendations: List[str] = []
        
        # Performance tracking
        self.performance_trends: Dict[str, str] = {}  # improving, degrading, stable
        self.baseline_metrics: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.debug("PerformanceDashboard initialized")
    
    def start_monitoring(self) -> None:
        """Start the performance monitoring dashboard."""
        if self.is_running:
            logger.warning("Performance dashboard is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        self.update_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceDashboard",
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("Performance dashboard monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the performance monitoring dashboard."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            if self.update_thread.is_alive():
                logger.warning("Dashboard monitoring thread did not stop gracefully")
        
        logger.info("Performance dashboard monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for dashboard updates."""
        logger.debug("Performance dashboard monitoring loop started")
        
        try:
            while self.is_running and not self._stop_event.is_set():
                try:
                    self._update_dashboard()
                    
                except Exception as e:
                    logger.error("Error updating performance dashboard: %s", e)
                
                # Wait for next update or stop signal
                if self._stop_event.wait(timeout=self.config.update_interval):
                    break
                    
        except Exception as e:
            logger.error("Fatal error in dashboard monitoring loop: %s", e)
        finally:
            logger.debug("Performance dashboard monitoring loop ended")
    
    def _update_dashboard(self) -> None:
        """Update dashboard with latest performance metrics."""
        timestamp = datetime.now()
        
        # Collect current metrics
        current_metrics = self._collect_current_metrics()
        
        with self._lock:
            self.current_metrics = current_metrics
            
            # Update metric history
            for metric_name, value in current_metrics.items():
                if isinstance(value, (int, float)):
                    self.metric_history[metric_name].append({
                        'timestamp': timestamp,
                        'value': value
                    })
        
        # Check for alerts
        if self.config.enable_alerts:
            self._check_performance_alerts(current_metrics, timestamp)
        
        # Update performance trends
        self._update_performance_trends()
        
        # Generate recommendations
        if self.config.enable_recommendations:
            self._update_recommendations()
        
        # Notify UI callback if available
        if self.ui_callback:
            self.ui_callback("dashboard_update", {
                'metrics': current_metrics,
                'alerts': list(self.alerts)[-5:],  # Last 5 alerts
                'recommendations': self.recommendations[:3],  # Top 3 recommendations
                'trends': self.performance_trends
            })
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics from all sources."""
        metrics = {}
        
        try:
            # Get metrics from collector
            if self.metrics_collector:
                # Audio pipeline metrics
                audio_metrics = self.metrics_collector.get_metric_summary('audio_stream.total_frames')
                if 'error' not in audio_metrics:
                    metrics['total_frames'] = audio_metrics.get('latest', 0)
                    metrics['frames_per_second'] = self._calculate_rate('audio_stream.total_frames')
                
                # Buffer metrics
                buffer_util = self.metrics_collector.get_metric_summary('buffer_manager.utilization_percent')
                if 'error' not in buffer_util:
                    metrics['buffer_utilization'] = buffer_util.get('latest', 0)
                    metrics['buffer_utilization_avg'] = buffer_util.get('average', 0)
                
                overflow_count = self.metrics_collector.get_metric_summary('buffer_manager.overflow_count')
                if 'error' not in overflow_count:
                    metrics['buffer_overflows'] = overflow_count.get('latest', 0)
                    metrics['overflow_rate'] = self._calculate_rate('buffer_manager.overflow_count')
                
                # Error metrics
                error_count = self.metrics_collector.get_metric_summary('error_recovery.total_errors')
                if 'error' not in error_count:
                    metrics['total_errors'] = error_count.get('latest', 0)
                    metrics['error_rate'] = self._calculate_rate('error_recovery.total_errors')
                
                # Latency metrics
                latency_stats = self.metrics_collector.get_timer_statistics('audio_stream.callback_latency')
                if 'error' not in latency_stats:
                    metrics['latency_avg_ms'] = latency_stats.get('average', 0) * 1000
                    metrics['latency_p95_ms'] = latency_stats.get('p95', 0) * 1000
                    metrics['latency_max_ms'] = latency_stats.get('max', 0) * 1000
                
                # System metrics (if available)
                cpu_metric = self.metrics_collector.get_metric_summary('system.cpu_percent')
                if 'error' not in cpu_metric:
                    metrics['cpu_usage'] = cpu_metric.get('latest', 0)
                
                memory_metric = self.metrics_collector.get_metric_summary('system.memory_percent')
                if 'error' not in memory_metric:
                    metrics['memory_usage'] = memory_metric.get('latest', 0)
                
                # Collection health
                debug_info = self.metrics_collector.get_debug_info()
                metrics['metrics_collection_health'] = (
                    100.0 - (debug_info.get('collection_stats', {}).get('failed_collections', 0) /
                             max(1, debug_info.get('collection_stats', {}).get('total_collections', 1)) * 100)
                )
                
        except Exception as e:
            logger.error("Error collecting dashboard metrics: %s", e)
            metrics['collection_error'] = str(e)
        
        return metrics
    
    def _calculate_rate(self, metric_name: str, window_seconds: float = 60.0) -> float:
        """Calculate rate of change for a metric over a time window."""
        try:
            values = self.metrics_collector.get_metric_values(
                metric_name,
                since=datetime.now() - timedelta(seconds=window_seconds)
            )
            
            if len(values) < 2:
                return 0.0
            
            # Calculate rate as change per second
            first_value = values[0].value
            last_value = values[-1].value
            time_diff = (values[-1].timestamp - values[0].timestamp).total_seconds()
            
            if time_diff > 0:
                return (last_value - first_value) / time_diff
            else:
                return 0.0
                
        except Exception as e:
            logger.debug("Error calculating rate for %s: %s", metric_name, e)
            return 0.0
    
    def _check_performance_alerts(self, metrics: Dict[str, Any], timestamp: datetime) -> None:
        """Check for performance alerts based on current metrics."""
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Check if we have thresholds for this metric
            thresholds = self.config.alert_thresholds.get(metric_name)
            if not thresholds:
                continue
            
            # Check critical threshold
            if 'critical' in thresholds and value >= thresholds['critical']:
                alert = PerformanceAlert(
                    timestamp=timestamp,
                    level='CRITICAL',
                    component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                    message=f"{metric_name} is critically high: {value:.1f}",
                    metric_value=value,
                    threshold=thresholds['critical']
                )
                self._add_alert(alert)
                
            # Check warning threshold
            elif 'warning' in thresholds and value >= thresholds['warning']:
                alert = PerformanceAlert(
                    timestamp=timestamp,
                    level='WARNING',
                    component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                    message=f"{metric_name} is elevated: {value:.1f}",
                    metric_value=value,
                    threshold=thresholds['warning']
                )
                self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert) -> None:
        """Add a performance alert to the queue."""
        with self._lock:
            # Check if we already have a recent similar alert
            recent_similar = any(
                a.component == alert.component and 
                a.level == alert.level and
                (alert.timestamp - a.timestamp).total_seconds() < 30
                for a in list(self.alerts)[-5:]
            )
            
            if not recent_similar:
                self.alerts.append(alert)
                logger.warning("Performance alert: %s - %s", alert.level, alert.message)
    
    def _update_performance_trends(self) -> None:
        """Update performance trend analysis."""
        with self._lock:
            for metric_name, history in self.metric_history.items():
                if len(history) < 10:  # Need enough data points
                    continue
                
                try:
                    # Get recent values
                    recent_values = [point['value'] for point in list(history)[-10:]]
                    older_values = [point['value'] for point in list(history)[-20:-10]] if len(history) >= 20 else []
                    
                    if not older_values:
                        continue
                    
                    # Calculate trend
                    recent_avg = sum(recent_values) / len(recent_values)
                    older_avg = sum(older_values) / len(older_values)
                    
                    change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                    
                    if abs(change_percent) < 5:
                        trend = 'stable'
                    elif change_percent > 0:
                        trend = 'degrading' if metric_name in ['error_rate', 'buffer_utilization', 'latency_avg_ms'] else 'improving'
                    else:
                        trend = 'improving' if metric_name in ['error_rate', 'buffer_utilization', 'latency_avg_ms'] else 'degrading'
                    
                    self.performance_trends[metric_name] = trend
                    
                except Exception as e:
                    logger.debug("Error calculating trend for %s: %s", metric_name, e)
    
    def _update_recommendations(self) -> None:
        """Update performance optimization recommendations."""
        recommendations = []
        
        try:
            # Analyze current metrics for recommendations
            metrics = self.current_metrics
            
            # Buffer utilization recommendations
            buffer_util = metrics.get('buffer_utilization', 0)
            if buffer_util > 90:
                recommendations.append("Buffer utilization is very high - consider increasing buffer sizes")
            elif buffer_util > 80:
                recommendations.append("Buffer utilization is elevated - monitor for potential overflows")
            elif buffer_util < 20:
                recommendations.append("Buffer utilization is low - consider reducing buffer sizes for lower latency")
            
            # Error rate recommendations
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 1:
                recommendations.append("High error rate detected - check audio device connectivity and system resources")
            
            # Latency recommendations
            latency_avg = metrics.get('latency_avg_ms', 0)
            if latency_avg > 100:
                recommendations.append("High audio latency detected - consider reducing buffer sizes or blocksize")
            
            # CPU usage recommendations
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 85:
                recommendations.append("High CPU usage - consider reducing audio quality or closing other applications")
            
            # Memory usage recommendations
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 90:
                recommendations.append("High memory usage - consider reducing buffer sizes or restarting the application")
            
            # Trend-based recommendations
            trends = self.performance_trends
            if trends.get('error_rate') == 'degrading':
                recommendations.append("Error rate is increasing - investigate potential hardware or configuration issues")
            
            if trends.get('latency_avg_ms') == 'degrading':
                recommendations.append("Audio latency is increasing - check system load and audio configuration")
            
            # Collection health recommendations
            collection_health = metrics.get('metrics_collection_health', 100)
            if collection_health < 90:
                recommendations.append("Metrics collection is experiencing issues - consider reducing collection frequency")
            
            with self._lock:
                self.recommendations = recommendations[:5]  # Keep top 5 recommendations
                
        except Exception as e:
            logger.error("Error updating recommendations: %s", e)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for UI display."""
        with self._lock:
            return {
                'metrics': self.current_metrics.copy(),
                'alerts': list(self.alerts),
                'recommendations': self.recommendations.copy(),
                'trends': self.performance_trends.copy(),
                'is_running': self.is_running
            }
    
    def create_dashboard_panel(self) -> Panel:
        """Create a Rich panel for dashboard display."""
        try:
            dashboard_data = self.get_dashboard_data()
            
            # Create main table
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="cyan", min_width=20)
            table.add_column("Value", style="white", min_width=15)
            table.add_column("Trend", style="yellow", min_width=10)
            
            metrics = dashboard_data['metrics']
            trends = dashboard_data['trends']
            
            # Add key metrics
            if 'buffer_utilization' in metrics:
                trend_icon = self._get_trend_icon(trends.get('buffer_utilization', 'stable'))
                color = self._get_metric_color(metrics['buffer_utilization'], [80, 95])
                table.add_row(
                    "Buffer Usage",
                    Text(f"{metrics['buffer_utilization']:.1f}%", style=color),
                    trend_icon
                )
            
            if 'error_rate' in metrics:
                trend_icon = self._get_trend_icon(trends.get('error_rate', 'stable'))
                color = self._get_metric_color(metrics['error_rate'], [1, 5])
                table.add_row(
                    "Error Rate",
                    Text(f"{metrics['error_rate']:.2f}/s", style=color),
                    trend_icon
                )
            
            if 'latency_avg_ms' in metrics:
                trend_icon = self._get_trend_icon(trends.get('latency_avg_ms', 'stable'))
                color = self._get_metric_color(metrics['latency_avg_ms'], [50, 100])
                table.add_row(
                    "Avg Latency",
                    Text(f"{metrics['latency_avg_ms']:.1f}ms", style=color),
                    trend_icon
                )
            
            if 'cpu_usage' in metrics:
                trend_icon = self._get_trend_icon(trends.get('cpu_usage', 'stable'))
                color = self._get_metric_color(metrics['cpu_usage'], [80, 95])
                table.add_row(
                    "CPU Usage",
                    Text(f"{metrics['cpu_usage']:.1f}%", style=color),
                    trend_icon
                )
            
            if 'frames_per_second' in metrics:
                table.add_row(
                    "Audio Rate",
                    Text(f"{metrics['frames_per_second']:.0f} fps", style="green"),
                    ""
                )
            
            # Add alerts section if there are any
            alerts = dashboard_data['alerts']
            if alerts:
                table.add_row("", "", "")  # Separator
                table.add_row("Recent Alerts:", "", "")
                for alert in alerts[-3:]:  # Show last 3 alerts
                    alert_color = {
                        'CRITICAL': 'red',
                        'WARNING': 'yellow',
                        'INFO': 'blue'
                    }.get(alert.level, 'white')
                    
                    table.add_row(
                        f"  {alert.component}",
                        Text(alert.level, style=alert_color),
                        ""
                    )
            
            # Add recommendations section
            recommendations = dashboard_data['recommendations']
            if recommendations:
                table.add_row("", "", "")  # Separator
                table.add_row("Recommendations:", "", "")
                for rec in recommendations[:2]:  # Show top 2 recommendations
                    table.add_row(
                        f"  • {rec[:40]}...",
                        "",
                        ""
                    )
            
            return Panel(
                table,
                title="Performance Dashboard",
                border_style="blue",
                subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            logger.error("Error creating dashboard panel: %s", e)
            return Panel(
                Text(f"Dashboard Error: {e}", style="red"),
                title="Performance Dashboard",
                border_style="red"
            )
    
    def _get_trend_icon(self, trend: str) -> str:
        """Get icon for performance trend."""
        return {
            'improving': '↗️',
            'degrading': '↘️',
            'stable': '→'
        }.get(trend, '')
    
    def _get_metric_color(self, value: float, thresholds: List[float]) -> str:
        """Get color for metric value based on thresholds."""
        if len(thresholds) >= 2:
            if value >= thresholds[1]:
                return 'red'
            elif value >= thresholds[0]:
                return 'yellow'
        elif len(thresholds) >= 1:
            if value >= thresholds[0]:
                return 'yellow'
        
        return 'green'
    
    def export_performance_report(self, filepath: str) -> None:
        """Export comprehensive performance report."""
        try:
            # Generate diagnostic report from metrics collector
            report = self.metrics_collector.generate_diagnostic_report(DiagnosticLevel.DETAILED)
            
            # Add dashboard-specific data
            dashboard_data = self.get_dashboard_data()
            
            enhanced_report = {
                'timestamp': datetime.now().isoformat(),
                'dashboard_data': dashboard_data,
                'diagnostic_report': report.to_dict(),
                'performance_summary': {
                    'total_alerts': len(dashboard_data['alerts']),
                    'critical_alerts': len([a for a in dashboard_data['alerts'] if a.level == 'CRITICAL']),
                    'active_recommendations': len(dashboard_data['recommendations']),
                    'trending_metrics': {
                        metric: trend for metric, trend in dashboard_data['trends'].items()
                        if trend != 'stable'
                    }
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(enhanced_report, f, indent=2, default=str)
            
            logger.info("Performance report exported to %s", filepath)
            
        except Exception as e:
            logger.error("Error exporting performance report: %s", e)
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()