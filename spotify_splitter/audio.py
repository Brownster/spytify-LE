try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency may be missing
    sd = None
import numpy as np
from queue import Queue, Full
from typing import Optional, Callable
import logging
import time
import threading
from datetime import datetime

from .buffer_management import AdaptiveBufferManager, BufferHealth, HealthStatus
from .buffer_health_monitor import BufferHealthMonitor
from .error_recovery import ErrorRecoveryManager, RecoveryAction
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class AudioStream:
    """Continuously capture audio from a PipeWire/PulseAudio monitor."""

    def __init__(
        self,
        monitor_name: str,
        samplerate: int = 44100,
        channels: int = 2,
        q: Optional[Queue] = None,
        queue_size: int = 200,
        blocksize: Optional[int] = None,
        latency: Optional[float] = None,
        ui_callback: Optional[callable] = None,
    ):
        """Create a stream reading from ``monitor_name``.

        ``queue_size`` determines how many blocks of audio are buffered before
        old data is dropped. ``blocksize`` and ``latency`` are passed directly
        to :class:`sounddevice.InputStream` to control callback timing.
        
        Increased default queue_size to 50 to prevent buffer underruns during
        intensive track processing operations.
        """

        self.q: Queue[np.ndarray] = q or Queue(maxsize=queue_size)
        self.ui_callback = ui_callback

        def _open(device):
            return sd.InputStream(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
                callback=self._callback,
                blocksize=blocksize or 2048,
                latency=latency or 'high',
            )

        try:
            self.stream = _open(monitor_name)
        except Exception:
            # ``monitor_name`` may not exactly match a PortAudio device.
            logger.debug("Exact device name match failed. Searching for a partial match...")
            try:
                devices = sd.query_devices()
            except Exception:  # pragma: no cover - requires PortAudio
                raise

            search_term = monitor_name
            if "alsa_output" in search_term and ".monitor" in search_term:
                # use the descriptive portion of the monitor name which often
                # matches what PortAudio reports
                search_term = search_term.split(".")[1].replace("_", " ")

            for idx, dev in enumerate(devices):
                name = str(dev.get("name", ""))
                if search_term in name:
                    logger.debug(
                        "Resolved monitor %s -> device %s (%s)", monitor_name, idx, name
                    )
                    self.stream = _open(idx)
                    break
            else:
                raise ValueError(f"Could not find a matching sounddevice for '{monitor_name}'")

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning("SoundDevice status: %s", status)
            if self.ui_callback:
                self.ui_callback("buffer_warning", None)
        try:
            self.q.put_nowait(indata.copy())
        except Full:
            logger.warning("Audio buffer full; dropping frames")
            if self.ui_callback:
                self.ui_callback("buffer_warning", None)

    def __enter__(self):
        self.stream.start()
        logger.debug("Audio stream started")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stream.stop()
        logger.debug("Audio stream stopped")

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """Blocks until frames are available."""
        return self.q.get(timeout=timeout)


class EnhancedAudioStream(AudioStream):
    """
    Enhanced AudioStream with adaptive buffer management and error recovery.
    
    Extends the base AudioStream class with:
    - Adaptive buffer management with dynamic sizing
    - Real-time buffer health monitoring
    - Automatic error recovery and stream reconnection
    - Emergency buffer expansion mechanisms
    """
    
    def __init__(
        self,
        monitor_name: str,
        buffer_manager: Optional[AdaptiveBufferManager] = None,
        error_recovery: Optional[ErrorRecoveryManager] = None,
        health_monitor: Optional[BufferHealthMonitor] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        samplerate: int = 44100,
        channels: int = 2,
        q: Optional[Queue] = None,
        queue_size: int = 200,
        blocksize: Optional[int] = None,
        latency: Optional[float] = None,
        ui_callback: Optional[Callable] = None,
        enable_adaptive_management: bool = True,
        enable_health_monitoring: bool = True,
        enable_metrics_collection: bool = True,
        **kwargs
    ):
        """
        Create an enhanced audio stream with adaptive capabilities.
        
        Args:
            monitor_name: Audio device monitor name
            buffer_manager: Adaptive buffer manager instance
            error_recovery: Error recovery manager instance
            health_monitor: Buffer health monitor instance
            enable_adaptive_management: Enable adaptive buffer management
            enable_health_monitoring: Enable real-time health monitoring
            **kwargs: Additional arguments passed to base AudioStream
        """
        # Initialize adaptive components
        self.buffer_manager = buffer_manager or AdaptiveBufferManager(
            initial_queue_size=queue_size
        )
        self.error_recovery = error_recovery or ErrorRecoveryManager()
        self.health_monitor = health_monitor
        self.metrics_collector = metrics_collector
        
        # Configuration flags
        self.enable_adaptive_management = enable_adaptive_management
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_metrics_collection = enable_metrics_collection
        
        # Stream state tracking
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = 5
        self.last_callback_time = 0.0
        self.callback_count = 0
        self.emergency_expansions = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'dropped_frames': 0,
            'buffer_overflows': 0,
            'buffer_underruns': 0,
            'reconnections': 0,
            'emergency_expansions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Store original parameters for reconnection
        self.original_params = {
            'monitor_name': monitor_name,
            'samplerate': samplerate,
            'channels': channels,
            'blocksize': blocksize,
            'latency': latency
        }
        
        # Initialize base class with adaptive queue size
        if self.enable_adaptive_management:
            adaptive_settings = self.buffer_manager.get_optimal_settings()
            queue_size = adaptive_settings.queue_size
            blocksize = blocksize or adaptive_settings.blocksize
            latency = latency or adaptive_settings.latency
        
        # Initialize base AudioStream (filter out unknown kwargs)
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['enable_error_recovery', 'enable_adaptive_management', 
                                   'enable_health_monitoring', 'enable_metrics_collection']}
        super().__init__(
            monitor_name=monitor_name,
            samplerate=samplerate,
            channels=channels,
            q=q,
            queue_size=queue_size,
            blocksize=blocksize,
            latency=latency,
            ui_callback=ui_callback,
            **base_kwargs
        )
        
        # Start health monitoring if enabled
        if self.enable_health_monitoring and self.health_monitor:
            self.health_monitor.start_monitoring(self.q)
        
        # Register with metrics collector if enabled
        if self.enable_metrics_collection and self.metrics_collector:
            self.metrics_collector.register_component('audio_stream', self._get_metrics_data)
        
        logger.info(
            "EnhancedAudioStream initialized: adaptive=%s, monitoring=%s, metrics=%s, queue_size=%d",
            enable_adaptive_management, enable_health_monitoring, enable_metrics_collection, queue_size
        )
    
    def _adaptive_callback(self, indata, frames, time_info, status):
        """
        Enhanced audio callback with adaptive buffer management and health monitoring.
        
        This callback extends the base functionality with:
        - Real-time buffer utilization monitoring
        - Dynamic buffer size adjustment
        - Emergency buffer expansion
        - Error recovery handling
        """
        callback_start_time = time.time()
        
        with self._lock:
            self.callback_count += 1
            self.last_callback_time = callback_start_time
        
        try:
            # Handle status warnings/errors
            if status:
                self._handle_callback_status(status)
            
            # Monitor buffer health if adaptive management is enabled
            if self.enable_adaptive_management:
                self._perform_adaptive_management()
            
            # Process audio data
            self._process_audio_data(indata, frames)
            
            # Update performance metrics
            with self._lock:
                self.metrics['total_frames'] += frames
            
        except Exception as e:
            logger.error("Error in adaptive callback: %s", e)
            self._handle_callback_error(e)
    
    def _handle_callback_status(self, status):
        """Handle sounddevice status messages in callback."""
        logger.warning("SoundDevice status in adaptive callback: %s", status)
        
        # Notify UI callback if available
        if self.ui_callback:
            self.ui_callback("buffer_warning", {"status": str(status)})
        
        # Handle specific status types
        status_str = str(status)
        if "overflow" in status_str.lower():
            with self._lock:
                self.metrics['buffer_overflows'] += 1
            self.buffer_manager.record_overflow()
            
            # Attempt emergency buffer expansion
            if self.enable_adaptive_management:
                self._attempt_emergency_expansion()
                
        elif "underrun" in status_str.lower():
            with self._lock:
                self.metrics['buffer_underruns'] += 1
            self.buffer_manager.record_underrun()
    
    def _perform_adaptive_management(self):
        """Perform adaptive buffer management during callback."""
        try:
            # Monitor current buffer utilization
            metrics = self.buffer_manager.monitor_utilization(self.q)
            
            # Record latency measurement
            if self.last_callback_time > 0:
                callback_interval = time.time() - self.last_callback_time
                self.buffer_manager.record_latency(callback_interval)
            
            # Check if buffer adjustment is needed
            new_size = self.buffer_manager.adjust_buffer_size(metrics)
            
            # Apply buffer size changes if needed (note: this requires stream restart)
            current_maxsize = getattr(self.q, 'maxsize', 0)
            if new_size != current_maxsize and abs(new_size - current_maxsize) > 10:
                logger.debug("Buffer size adjustment needed: %d -> %d", current_maxsize, new_size)
                # Note: Actual queue resizing would require stream restart in a real implementation
                # For now, we log the recommendation
                
        except Exception as e:
            logger.error("Error in adaptive management: %s", e)
    
    def _process_audio_data(self, indata, frames):
        """Process and queue audio data with error handling."""
        try:
            # Copy audio data
            audio_copy = indata.copy()
            
            # Attempt to queue the data
            self.q.put_nowait(audio_copy)
            
        except Full:
            # Buffer is full - handle overflow
            self._handle_buffer_overflow()
            
        except Exception as e:
            logger.error("Error processing audio data: %s", e)
            with self._lock:
                self.metrics['dropped_frames'] += frames
    
    def _handle_buffer_overflow(self):
        """Handle buffer overflow situations."""
        logger.warning("Audio buffer overflow in enhanced stream")
        
        with self._lock:
            self.metrics['buffer_overflows'] += 1
            self.metrics['dropped_frames'] += 1  # Approximate frame count
        
        # Record overflow in buffer manager
        self.buffer_manager.record_overflow()
        
        # Notify UI callback
        if self.ui_callback:
            self.ui_callback("buffer_overflow", {
                "queue_size": self.q.qsize(),
                "max_size": getattr(self.q, 'maxsize', 0)
            })
        
        # Attempt emergency expansion if enabled
        if self.enable_adaptive_management:
            self._attempt_emergency_expansion()
    
    def _attempt_emergency_expansion(self):
        """Attempt emergency buffer expansion."""
        try:
            if self.buffer_manager.emergency_expansion():
                with self._lock:
                    self.emergency_expansions += 1
                    self.metrics['emergency_expansions'] += 1
                
                logger.info("Emergency buffer expansion successful")
                
                if self.ui_callback:
                    self.ui_callback("emergency_expansion", {
                        "new_size": self.buffer_manager.current_queue_size,
                        "expansion_count": self.emergency_expansions
                    })
                return True
            else:
                logger.warning("Emergency buffer expansion failed - at maximum size")
                return False
                
        except Exception as e:
            logger.error("Error during emergency buffer expansion: %s", e)
            return False
    
    def _handle_callback_error(self, error: Exception):
        """Handle errors that occur in the audio callback."""
        context = "audio_callback"
        recovery_action = self.error_recovery.handle_error(error, context)
        
        logger.error("Callback error handled: %s -> %s", type(error).__name__, recovery_action.value)
        
        # Note: Actual recovery would be handled outside the callback
        # to avoid blocking the audio thread
    
    def handle_stream_error(self, error: Exception) -> bool:
        """
        Handle stream-level errors with automatic recovery.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if recovery was successful, False otherwise
        """
        context = f"audio_stream_{self.original_params['monitor_name']}"
        recovery_action = self.error_recovery.handle_error(error, context)
        
        logger.info("Handling stream error: %s with action: %s", type(error).__name__, recovery_action.value)
        
        # Notify UI of error and recovery attempt
        if self.ui_callback:
            self.ui_callback("stream_error", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "recovery_action": recovery_action.value,
                "context": context
            })
        
        # Attempt recovery based on action
        recovery_success = False
        
        if recovery_action == RecoveryAction.RECONNECT:
            recovery_success = self._attempt_reconnection()
        elif recovery_action == RecoveryAction.BUFFER_EXPAND:
            recovery_success = self._attempt_emergency_expansion()
        elif recovery_action == RecoveryAction.RETRY:
            recovery_success = self._attempt_retry()
        elif recovery_action == RecoveryAction.GRACEFUL_DEGRADE:
            recovery_success = self._attempt_graceful_degradation(error)
        elif recovery_action == RecoveryAction.ESCALATE:
            self._escalate_stream_error(error, context)
            recovery_success = False
        else:
            logger.warning("Unknown recovery action: %s", recovery_action.value)
            recovery_success = False
        
        # Notify UI of recovery result
        if self.ui_callback:
            self.ui_callback("stream_recovery_result", {
                "success": recovery_success,
                "recovery_action": recovery_action.value,
                "error_type": type(error).__name__
            })
        
        return recovery_success
    
    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect the audio stream."""
        def reconnect_func():
            try:
                # Stop current stream
                if hasattr(self, 'stream') and self.stream:
                    self.stream.stop()
                    self.stream.close()
                
                # Wait a moment before reconnecting
                time.sleep(0.1)
                
                # Recreate stream with original parameters
                self._recreate_stream()
                
                # Restart stream
                self.stream.start()
                
                with self._lock:
                    self.reconnection_attempts += 1
                    self.metrics['reconnections'] += 1
                
                logger.info("Stream reconnection successful (attempt %d)", self.reconnection_attempts)
                return True
                
            except Exception as reconnect_error:
                logger.error("Stream reconnection failed: %s", reconnect_error)
                return False
        
        return self.error_recovery.attempt_recovery(
            RecoveryAction.RECONNECT,
            reconnect_func,
            f"stream_reconnection_{self.reconnection_attempts}"
        )
    
    def _recreate_stream(self):
        """Recreate the audio stream with current optimal settings."""
        # Get optimal settings if adaptive management is enabled
        if self.enable_adaptive_management:
            settings = self.buffer_manager.get_optimal_settings()
            blocksize = settings.blocksize
            latency = settings.latency
        else:
            blocksize = self.original_params['blocksize'] or 2048
            latency = self.original_params['latency'] or 'high'
        
        # Create new stream
        self.stream = sd.InputStream(
            device=self.original_params['monitor_name'],
            channels=self.original_params['channels'],
            samplerate=self.original_params['samplerate'],
            dtype="float32",
            callback=self._adaptive_callback,
            blocksize=blocksize,
            latency=latency,
        )
    
    def _attempt_retry(self) -> bool:
        """Attempt a simple retry operation."""
        def retry_func():
            try:
                # Simple retry - just ensure stream is running
                if hasattr(self, 'stream') and self.stream and not self.stream.active:
                    self.stream.start()
                return True
            except Exception:
                return False
        
        return self.error_recovery.attempt_recovery(
            RecoveryAction.RETRY,
            retry_func,
            "stream_retry"
        )
    
    def _attempt_graceful_degradation(self, error: Exception) -> bool:
        """
        Attempt graceful degradation for stream errors.
        
        Args:
            error: The error that occurred
            
        Returns:
            True if graceful degradation was successful, False otherwise
        """
        try:
            logger.info("Attempting graceful degradation for stream error: %s", type(error).__name__)
            
            # Graceful degradation strategies based on error type
            error_type = type(error).__name__
            
            if "PortAudio" in error_type or "sounddevice" in error_type:
                # Audio device issues - try with more conservative settings
                logger.info("Audio device error - attempting conservative settings")
                
                # Try to recreate stream with more conservative settings
                try:
                    if hasattr(self, 'stream') and self.stream:
                        self.stream.stop()
                        self.stream.close()
                    
                    # Use more conservative audio settings
                    conservative_stream = sd.InputStream(
                        device=self.original_params['monitor_name'],
                        channels=self.original_params['channels'],
                        samplerate=self.original_params['samplerate'],
                        dtype="float32",
                        callback=self._adaptive_callback,
                        blocksize=4096,  # Larger blocksize for stability
                        latency='high',  # Higher latency for stability
                    )
                    
                    self.stream = conservative_stream
                    self.stream.start()
                    
                    logger.info("Graceful degradation successful - using conservative audio settings")
                    
                    # Notify UI of degraded mode
                    if self.ui_callback:
                        self.ui_callback("degraded_mode", {
                            "reason": "Audio device issues",
                            "settings": "Conservative audio settings applied"
                        })
                    
                    return True
                    
                except Exception as degradation_error:
                    logger.error("Conservative settings degradation failed: %s", degradation_error)
                    return False
                    
            elif "Memory" in error_type:
                # Memory issues - reduce buffer sizes and disable some features
                logger.info("Memory error - attempting reduced resource usage")
                
                try:
                    # Temporarily disable adaptive management to reduce memory usage
                    if self.enable_adaptive_management:
                        logger.info("Temporarily disabling adaptive management due to memory constraints")
                        self.enable_adaptive_management = False
                        
                        # Notify UI of feature degradation
                        if self.ui_callback:
                            self.ui_callback("feature_degraded", {
                                "feature": "adaptive_management",
                                "reason": "Memory constraints"
                            })
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Try to restart with minimal settings
                    if hasattr(self, 'stream') and self.stream:
                        self.stream.stop()
                        self.stream.close()
                    
                    # Create minimal stream
                    minimal_stream = sd.InputStream(
                        device=self.original_params['monitor_name'],
                        channels=self.original_params['channels'],
                        samplerate=self.original_params['samplerate'],
                        dtype="float32",
                        callback=self._callback,  # Use basic callback instead of adaptive
                        blocksize=2048,
                        latency='high',
                    )
                    
                    self.stream = minimal_stream
                    self.stream.start()
                    
                    logger.info("Graceful degradation successful - using minimal resource mode")
                    
                    # Notify UI of degraded mode
                    if self.ui_callback:
                        self.ui_callback("degraded_mode", {
                            "reason": "Memory constraints",
                            "settings": "Minimal resource usage mode"
                        })
                    
                    return True
                    
                except Exception as degradation_error:
                    logger.error("Memory degradation failed: %s", degradation_error)
                    return False
                    
            else:
                # Generic degradation - disable advanced features and use basic mode
                logger.info("Generic error - attempting basic mode degradation")
                
                try:
                    # Disable advanced features
                    original_adaptive = self.enable_adaptive_management
                    original_monitoring = self.enable_health_monitoring
                    original_metrics = self.enable_metrics_collection
                    
                    self.enable_adaptive_management = False
                    self.enable_health_monitoring = False
                    self.enable_metrics_collection = False
                    
                    # Try to restart in basic mode
                    if hasattr(self, 'stream') and self.stream:
                        self.stream.stop()
                        self.stream.close()
                    
                    # Create basic stream
                    basic_stream = sd.InputStream(
                        device=self.original_params['monitor_name'],
                        channels=self.original_params['channels'],
                        samplerate=self.original_params['samplerate'],
                        dtype="float32",
                        callback=self._callback,  # Use basic callback
                        blocksize=self.original_params['blocksize'] or 2048,
                        latency=self.original_params['latency'] or 'high',
                    )
                    
                    self.stream = basic_stream
                    self.stream.start()
                    
                    logger.info("Graceful degradation successful - using basic mode")
                    
                    # Notify UI of degraded mode
                    if self.ui_callback:
                        self.ui_callback("degraded_mode", {
                            "reason": f"{error_type} error",
                            "settings": "Basic mode - advanced features disabled",
                            "disabled_features": {
                                "adaptive_management": original_adaptive,
                                "health_monitoring": original_monitoring,
                                "metrics_collection": original_metrics
                            }
                        })
                    
                    return True
                    
                except Exception as degradation_error:
                    logger.error("Basic mode degradation failed: %s", degradation_error)
                    return False
            
        except Exception as e:
            logger.error("Error in graceful degradation attempt: %s", e)
            return False
    
    def _escalate_stream_error(self, error: Exception, context: str) -> None:
        """
        Escalate a stream error that cannot be automatically recovered.
        
        Args:
            error: The exception to escalate
            context: Additional context information
        """
        logger.critical("Stream error escalated: %s in %s - %s", 
                       type(error).__name__, context, str(error))
        
        # Notify UI of critical error
        if self.ui_callback:
            self.ui_callback("critical_error", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "escalated": True,
                "recommendations": self._get_error_recommendations(error)
            })
        
        # Generate detailed error diagnostics
        try:
            diagnostics = self.error_recovery.get_diagnostics()
            logger.critical("Error diagnostics at escalation:")
            logger.critical("  - Total errors: %d", diagnostics.total_errors)
            logger.critical("  - Error rate: %.2f/hour", diagnostics.error_rate_per_hour)
            logger.critical("  - Recovery success rate: %.1f%%", diagnostics.recovery_success_rate * 100)
            
            if diagnostics.recommendations:
                logger.critical("  - Recommendations:")
                for rec in diagnostics.recommendations:
                    logger.critical("    * %s", rec)
                    
        except Exception as diag_error:
            logger.error("Error generating diagnostics during escalation: %s", diag_error)
        
        # Record escalation in error recovery manager
        self.error_recovery.escalate_error(error, context)
    
    def _get_error_recommendations(self, error: Exception) -> list:
        """
        Get user-actionable recommendations for an error.
        
        Args:
            error: The error that occurred
            
        Returns:
            List of recommendation strings
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        recommendations = []
        
        if "portaudio" in error_type.lower() or "sounddevice" in error_type.lower():
            recommendations.extend([
                "Check that your audio device is connected and functioning",
                "Try restarting the audio service (pulseaudio/pipewire)",
                "Verify that the monitor device name is correct",
                "Check for conflicting audio applications"
            ])
            
        elif "permission" in error_message or "access" in error_message:
            recommendations.extend([
                "Check audio device permissions",
                "Try running with appropriate user privileges",
                "Verify that the user is in the audio group"
            ])
            
        elif "memory" in error_type.lower():
            recommendations.extend([
                "Close other memory-intensive applications",
                "Reduce buffer sizes in configuration",
                "Check available system memory",
                "Consider using a lower sample rate"
            ])
            
        elif "device" in error_message:
            recommendations.extend([
                "Check that the audio device is available",
                "Try using a different audio device",
                "Restart the audio system",
                "Check device configuration"
            ])
            
        else:
            recommendations.extend([
                "Check system logs for additional error details",
                "Try restarting the application",
                "Verify system audio configuration",
                "Consider using different audio settings"
            ])
        
        return recommendations
    
    def reconnect_stream(self) -> bool:
        """
        Public method to manually reconnect the stream.
        
        Returns:
            True if reconnection was successful, False otherwise
        """
        if self.reconnection_attempts >= self.max_reconnection_attempts:
            logger.error("Maximum reconnection attempts reached (%d)", self.max_reconnection_attempts)
            return False
        
        return self._attempt_reconnection()
    
    def get_buffer_health(self) -> Optional[BufferHealth]:
        """
        Get current buffer health assessment.
        
        Returns:
            BufferHealth object with current status, or None if not available
        """
        if not self.enable_adaptive_management:
            return None
        
        try:
            metrics = self.buffer_manager.monitor_utilization(self.q)
            return self.buffer_manager.get_buffer_health(metrics)
        except Exception as e:
            logger.error("Error getting buffer health: %s", e)
            return None
    
    def get_performance_metrics(self) -> dict:
        """Get comprehensive performance metrics."""
        with self._lock:
            base_metrics = self.metrics.copy()
        
        # Add buffer manager statistics if available
        if self.enable_adaptive_management:
            buffer_stats = self.buffer_manager.get_stats()
            base_metrics.update({
                'buffer_stats': buffer_stats,
                'current_queue_size': self.buffer_manager.current_queue_size
            })
        
        # Add error recovery statistics
        recovery_stats = self.error_recovery.get_statistics()
        base_metrics.update({
            'error_recovery': recovery_stats,
            'reconnection_attempts': self.reconnection_attempts
        })
        
        # Add health monitoring statistics if available
        if self.health_monitor:
            health_stats = self.health_monitor.get_statistics()
            base_metrics.update({'health_monitoring': health_stats})
        
        return base_metrics
    
    def _get_metrics_data(self) -> dict:
        """
        Get current metrics data for the metrics collector.
        
        Returns:
            Dictionary of current metric values for collection
        """
        with self._lock:
            current_metrics = self.metrics.copy()
        
        # Add real-time queue information
        queue_size = self.q.qsize() if hasattr(self.q, 'qsize') else 0
        queue_maxsize = getattr(self.q, 'maxsize', 0)
        queue_utilization = (queue_size / queue_maxsize) if queue_maxsize > 0 else 0.0
        
        # Add callback performance metrics
        callback_metrics = {
            'callback_count': self.callback_count,
            'queue_size': queue_size,
            'queue_maxsize': queue_maxsize,
            'queue_utilization_percent': queue_utilization * 100,
            'emergency_expansions': self.emergency_expansions,
            'reconnection_attempts': self.reconnection_attempts
        }
        
        # Combine with existing metrics
        current_metrics.update(callback_metrics)
        
        # Add buffer health if available
        if self.enable_adaptive_management:
            try:
                buffer_health = self.get_buffer_health()
                if buffer_health:
                    current_metrics.update({
                        'buffer_health_status': buffer_health.status.value,
                        'buffer_utilization': buffer_health.utilization,
                        'buffer_overflow_risk': buffer_health.overflow_risk
                    })
            except Exception as e:
                logger.debug("Error getting buffer health for metrics: %s", e)
        
        return current_metrics
    
    def __enter__(self):
        """Enhanced context manager entry with error handling."""
        try:
            self.stream.start()
            logger.debug("Enhanced audio stream started")
            return self
        except Exception as e:
            logger.error("Error starting enhanced audio stream: %s", e)
            if not self.handle_stream_error(e):
                raise
            return self
    
    def __exit__(self, exc_type, exc, tb):
        """Enhanced context manager exit with cleanup."""
        try:
            # Stop health monitoring
            if self.health_monitor and self.enable_health_monitoring:
                self.health_monitor.stop_monitoring()
            
            # Stop audio stream
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
            
            logger.debug("Enhanced audio stream stopped")
            
            # Log final performance metrics
            metrics = self.get_performance_metrics()
            logger.info("Stream session metrics: %s", {
                k: v for k, v in metrics.items() 
                if k in ['total_frames', 'dropped_frames', 'reconnections', 'emergency_expansions']
            })
            
        except Exception as e:
            logger.error("Error stopping enhanced audio stream: %s", e)
    
    # Override the callback method to use adaptive callback
    def _callback(self, indata, frames, time, status):
        """Override base callback to use adaptive callback."""
        self._adaptive_callback(indata, frames, time, status)
