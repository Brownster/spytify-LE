"""
Automatic performance optimization system.

This module analyzes performance metrics and automatically suggests
or applies optimizations to improve audio pipeline performance.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import threading

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of performance optimizations."""
    BUFFER_SIZE_ADJUSTMENT = "buffer_size_adjustment"
    LATENCY_OPTIMIZATION = "latency_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    ERROR_REDUCTION = "error_reduction"
    QUALITY_ADJUSTMENT = "quality_adjustment"


class OptimizationPriority(Enum):
    """Priority levels for optimizations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""
    timestamp: datetime
    optimization_type: OptimizationType
    priority: OptimizationPriority
    title: str
    description: str
    expected_improvement: str
    implementation_steps: List[str]
    estimated_impact: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    prerequisites: List[str] = None
    risks: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.risks is None:
            self.risks = []


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    timestamp: datetime
    metrics: Dict[str, float]
    duration_hours: float
    conditions: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Result of an applied optimization."""
    suggestion: OptimizationSuggestion
    applied_timestamp: datetime
    success: bool
    actual_improvement: Optional[float]
    side_effects: List[str]
    rollback_needed: bool = False


class PerformanceOptimizer:
    """
    Automatic performance optimization system.
    
    Analyzes performance metrics, detects performance issues,
    and provides automatic optimization suggestions with
    impact estimation and risk assessment.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        auto_apply_safe_optimizations: bool = False,
        optimization_interval: float = 300.0,  # 5 minutes
        baseline_duration_hours: float = 1.0
    ):
        """
        Initialize the performance optimizer.
        
        Args:
            metrics_collector: MetricsCollector instance for data source
            auto_apply_safe_optimizations: Whether to automatically apply low-risk optimizations
            optimization_interval: Time between optimization analysis in seconds
            baseline_duration_hours: Duration for establishing performance baselines
        """
        self.metrics_collector = metrics_collector
        self.auto_apply_safe_optimizations = auto_apply_safe_optimizations
        self.optimization_interval = optimization_interval
        self.baseline_duration_hours = baseline_duration_hours
        
        # Optimization state
        self.is_running = False
        self.optimization_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.baselines: deque = deque(maxlen=24)  # Keep 24 baselines (24 hours if hourly)
        self.current_baseline: Optional[PerformanceBaseline] = None
        self.suggestions: deque = deque(maxlen=50)
        self.applied_optimizations: deque = deque(maxlen=100)
        
        # Performance analysis
        self.performance_patterns: Dict[str, Any] = {}
        self.optimization_history: Dict[str, List[OptimizationResult]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Optimization rules and thresholds
        self.optimization_rules = self._initialize_optimization_rules()
        
        logger.debug("PerformanceOptimizer initialized")
    
    def _initialize_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization rules and thresholds."""
        return {
            'buffer_utilization': {
                'high_threshold': 85.0,
                'low_threshold': 20.0,
                'optimization_types': [OptimizationType.BUFFER_SIZE_ADJUSTMENT],
                'priority_mapping': {
                    (90, 100): OptimizationPriority.CRITICAL,
                    (85, 90): OptimizationPriority.HIGH,
                    (80, 85): OptimizationPriority.MEDIUM,
                    (0, 20): OptimizationPriority.LOW
                }
            },
            'error_rate': {
                'high_threshold': 1.0,
                'critical_threshold': 5.0,
                'optimization_types': [OptimizationType.ERROR_REDUCTION, OptimizationType.BUFFER_SIZE_ADJUSTMENT],
                'priority_mapping': {
                    (5, float('inf')): OptimizationPriority.CRITICAL,
                    (2, 5): OptimizationPriority.HIGH,
                    (1, 2): OptimizationPriority.MEDIUM
                }
            },
            'latency_avg_ms': {
                'high_threshold': 100.0,
                'critical_threshold': 200.0,
                'optimization_types': [OptimizationType.LATENCY_OPTIMIZATION],
                'priority_mapping': {
                    (200, float('inf')): OptimizationPriority.CRITICAL,
                    (150, 200): OptimizationPriority.HIGH,
                    (100, 150): OptimizationPriority.MEDIUM
                }
            },
            'cpu_usage': {
                'high_threshold': 80.0,
                'critical_threshold': 95.0,
                'optimization_types': [OptimizationType.CPU_OPTIMIZATION],
                'priority_mapping': {
                    (95, 100): OptimizationPriority.CRITICAL,
                    (85, 95): OptimizationPriority.HIGH,
                    (80, 85): OptimizationPriority.MEDIUM
                }
            },
            'memory_usage': {
                'high_threshold': 85.0,
                'critical_threshold': 95.0,
                'optimization_types': [OptimizationType.MEMORY_OPTIMIZATION],
                'priority_mapping': {
                    (95, 100): OptimizationPriority.CRITICAL,
                    (90, 95): OptimizationPriority.HIGH,
                    (85, 90): OptimizationPriority.MEDIUM
                }
            }
        }
    
    def start_optimization(self) -> None:
        """Start automatic performance optimization."""
        if self.is_running:
            logger.warning("Performance optimizer is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="PerformanceOptimizer",
            daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self) -> None:
        """Stop automatic performance optimization."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=2.0)
            if self.optimization_thread.is_alive():
                logger.warning("Optimization thread did not stop gracefully")
        
        logger.info("Performance optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization analysis loop."""
        logger.debug("Performance optimization loop started")
        
        try:
            while self.is_running and not self._stop_event.is_set():
                try:
                    self._analyze_and_optimize()
                    
                except Exception as e:
                    logger.error("Error in optimization analysis: %s", e)
                
                # Wait for next analysis or stop signal
                if self._stop_event.wait(timeout=self.optimization_interval):
                    break
                    
        except Exception as e:
            logger.error("Fatal error in optimization loop: %s", e)
        finally:
            logger.debug("Performance optimization loop ended")
    
    def _analyze_and_optimize(self) -> None:
        """Analyze current performance and generate optimization suggestions."""
        timestamp = datetime.now()
        
        # Collect current metrics for analysis
        current_metrics = self._collect_optimization_metrics()
        
        # Generate optimization suggestions based on current metrics
        suggestions = []
        
        try:
            # Analyze buffer utilization
            buffer_util = current_metrics.get('buffer_utilization', 0)
            if buffer_util > 85:
                suggestions.append(OptimizationSuggestion(
                    timestamp=timestamp,
                    optimization_type=OptimizationType.BUFFER_SIZE_ADJUSTMENT,
                    priority=OptimizationPriority.HIGH,
                    title="Increase Buffer Size",
                    description=f"Buffer utilization is high ({buffer_util:.1f}%)",
                    expected_improvement="Reduced buffer overflows",
                    implementation_steps=["Increase queue_size parameter"],
                    estimated_impact=0.8,
                    confidence=0.9
                ))
            
            # Analyze error rate
            error_rate = current_metrics.get('error_rate', 0)
            if error_rate > 1.0:
                suggestions.append(OptimizationSuggestion(
                    timestamp=timestamp,
                    optimization_type=OptimizationType.ERROR_REDUCTION,
                    priority=OptimizationPriority.HIGH,
                    title="Reduce Error Rate",
                    description=f"Error rate is elevated ({error_rate:.2f}/min)",
                    expected_improvement="Improved stability",
                    implementation_steps=["Check system resources", "Increase buffer sizes"],
                    estimated_impact=0.7,
                    confidence=0.8
                ))
            
            # Analyze latency
            latency = current_metrics.get('latency_avg_ms', 0)
            if latency > 100:
                suggestions.append(OptimizationSuggestion(
                    timestamp=timestamp,
                    optimization_type=OptimizationType.LATENCY_OPTIMIZATION,
                    priority=OptimizationPriority.MEDIUM,
                    title="Optimize Latency",
                    description=f"Average latency is high ({latency:.1f}ms)",
                    expected_improvement="Reduced audio latency",
                    implementation_steps=["Reduce blocksize", "Lower buffer sizes"],
                    estimated_impact=0.6,
                    confidence=0.7
                ))
            
            with self._lock:
                # Add new suggestions
                for suggestion in suggestions:
                    self.suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error("Error generating optimization suggestions: %s", e)
    
    def _collect_optimization_metrics(self) -> Dict[str, float]:
        """Collect metrics relevant for optimization analysis."""
        metrics = {}
        
        try:
            # Buffer metrics
            buffer_util = self.metrics_collector.get_metric_summary('buffer_manager.utilization_percent')
            if 'error' not in buffer_util:
                metrics['buffer_utilization'] = buffer_util.get('average', 0)
            
            # Error metrics - calculate rate from total errors
            error_summary = self.metrics_collector.get_error_summary(
                since=datetime.now() - timedelta(minutes=30)
            )
            metrics['error_rate'] = error_summary.get('total_errors', 0) / 30.0  # errors per minute
            
            # Latency metrics
            latency_stats = self.metrics_collector.get_timer_statistics('audio_stream.callback_latency')
            if 'error' not in latency_stats:
                metrics['latency_avg_ms'] = latency_stats.get('average', 0) * 1000
            
            # System metrics
            cpu_metric = self.metrics_collector.get_metric_summary('system.cpu_percent')
            if 'error' not in cpu_metric:
                metrics['cpu_usage'] = cpu_metric.get('average', 0)
            
            memory_metric = self.metrics_collector.get_metric_summary('system.memory_percent')
            if 'error' not in memory_metric:
                metrics['memory_usage'] = memory_metric.get('average', 0)
                
        except Exception as e:
            logger.error("Error collecting optimization metrics: %s", e)
        
        return metrics
    
    def get_optimization_suggestions(self, limit: int = 10) -> List[OptimizationSuggestion]:
        """Get current optimization suggestions."""
        with self._lock:
            return list(self.suggestions)[-limit:]
    
    def get_optimization_history(self) -> Dict[str, List[OptimizationResult]]:
        """Get optimization application history."""
        with self._lock:
            return {k: list(v) for k, v in self.optimization_history.items()}
    
    def __enter__(self):
        """Context manager entry."""
        self.start_optimization()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_optimization()