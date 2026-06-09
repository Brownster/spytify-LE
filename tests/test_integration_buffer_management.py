"""
Integration tests for adaptive buffer management in main application.

Tests the integration between main.py, configuration profiles, and
adaptive buffer management components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from spotify_splitter.config_profiles import (
    ProfileManager, ProfileType, SystemCapabilityDetector, SystemCapabilities
)
from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferStrategy
from spotify_splitter.error_recovery import ErrorRecoveryManager


class TestBufferManagementIntegration:
    """Integration tests for buffer management components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_profile_to_buffer_manager_integration(self):
        """Test integration between configuration profiles and buffer manager."""
        # Test each profile type
        for profile_type in [ProfileType.HEADLESS, ProfileType.DESKTOP, ProfileType.HIGH_PERFORMANCE]:
            profile = ProfileManager.get_profile(profile_type)
            
            # Create buffer manager with profile settings
            buffer_manager = AdaptiveBufferManager(
                initial_queue_size=profile.queue_size,
                min_size=50,
                max_size=1000
            )
            
            # Verify buffer manager uses profile settings
            assert buffer_manager.current_queue_size == profile.queue_size
            
            # Get optimal settings from buffer manager
            settings = buffer_manager.get_optimal_settings()
            assert settings.queue_size == profile.queue_size
            assert settings.buffer_strategy in [BufferStrategy.CONSERVATIVE, BufferStrategy.BALANCED, BufferStrategy.LOW_LATENCY]
    
    def test_system_capability_to_profile_integration(self):
        """Test integration between system capabilities and profile selection."""
        # Test headless system
        headless_caps = SystemCapabilities(
            cpu_cores=2, memory_gb=4.0, is_headless=True,
            audio_backend="pulseaudio", has_gui=False, system_load=0.3
        )
        
        profile = ProfileManager.select_optimal_profile(headless_caps)
        assert "headless" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.CONSERVATIVE
        assert profile.queue_size >= 300  # Conservative buffer size
        
        # Test high-performance system
        high_perf_caps = SystemCapabilities(
            cpu_cores=16, memory_gb=32.0, is_headless=False,
            audio_backend="pipewire", has_gui=True, system_load=0.1
        )
        
        profile = ProfileManager.select_optimal_profile(high_perf_caps)
        assert "high_performance" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.LOW_LATENCY
        assert profile.latency <= 0.05  # Low latency
        
        # Test desktop system
        desktop_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.5
        )
        
        profile = ProfileManager.select_optimal_profile(desktop_caps)
        assert "desktop" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.BALANCED
    
    def test_error_recovery_integration(self):
        """Test integration between buffer manager and error recovery."""
        # Create error recovery manager
        error_recovery = ErrorRecoveryManager(max_retries=3)
        
        # Create buffer manager
        buffer_manager = AdaptiveBufferManager(initial_queue_size=200)
        
        # Simulate buffer overflow
        buffer_manager.record_overflow()
        assert buffer_manager.overflow_count == 1
        
        # Test emergency expansion
        old_size = buffer_manager.current_queue_size
        success = buffer_manager.emergency_expansion()
        assert success
        assert buffer_manager.current_queue_size > old_size
        
        # Test error handling
        test_error = Exception("Test buffer error")
        recovery_action = error_recovery.handle_error(test_error, "buffer_test")
        
        # Verify error was recorded
        stats = error_recovery.get_statistics()
        assert stats['total_errors'] == 1
    
    def test_profile_adjustment_integration(self):
        """Test profile adjustment based on system characteristics."""
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        
        # Test low memory adjustment
        low_memory_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=1.5, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        adjusted_profile = ProfileManager._adjust_profile_for_system(base_profile, low_memory_caps)
        
        # Should reduce buffer size and disable debug mode
        assert adjusted_profile.queue_size < base_profile.queue_size
        assert not adjusted_profile.enable_debug_mode
        
        # Create buffer manager with adjusted settings
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=adjusted_profile.queue_size,
            min_size=50,
            max_size=500  # Lower max for low memory
        )
        
        assert buffer_manager.current_queue_size == adjusted_profile.queue_size
        assert buffer_manager.max_size == 500
    
    def test_complete_workflow_integration(self):
        """Test complete workflow from system detection to buffer management."""
        # Step 1: Detect system capabilities (mocked for testing)
        capabilities = SystemCapabilities(
            cpu_cores=8, memory_gb=16.0, is_headless=False,
            audio_backend="pipewire", has_gui=True, system_load=0.2
        )
        
        # Step 2: Select optimal profile
        profile = ProfileManager.select_optimal_profile(capabilities)
        
        # Step 3: Create adaptive components
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=profile.queue_size,
            min_size=50,
            max_size=1000,
        )

        error_recovery = ErrorRecoveryManager(
            max_retries=profile.max_reconnection_attempts
        )

        # Step 4: Verify integration
        assert buffer_manager.current_queue_size == profile.queue_size
        assert error_recovery.max_retries == profile.max_reconnection_attempts

        # Step 5: Test runtime behavior
        import queue

        test_queue = queue.Queue(maxsize=profile.queue_size)

        # Simulate buffer usage
        for i in range(profile.queue_size // 2):
            test_queue.put(f"data_{i}")

        # Monitor and adjust
        metrics = buffer_manager.monitor_utilization(test_queue)
        buffer_manager.adjust_buffer_size(metrics)

        # Verify monitoring worked
        assert metrics.utilization_percent > 0
        assert metrics.queue_size == profile.queue_size // 2

    def test_cli_argument_override_integration(self):
        """Test that CLI arguments properly override profile defaults."""
        # Get base profile
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        
        # Simulate CLI overrides
        cli_queue_size = 500
        cli_latency = 0.05
        cli_debug_mode = True
        
        # Apply overrides (simulating main.py logic)
        effective_queue_size = cli_queue_size  # CLI override
        effective_latency = cli_latency  # CLI override
        effective_debug = cli_debug_mode  # CLI override
        effective_blocksize = base_profile.blocksize  # Use profile default
        
        # Create components with effective settings
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=effective_queue_size
        )

        # Verify CLI overrides were applied
        assert buffer_manager.current_queue_size == cli_queue_size
        assert buffer_manager.current_queue_size != base_profile.queue_size
    
    def test_profile_serialization_integration(self):
        """Test profile serialization for configuration persistence."""
        profile = ProfileManager.get_profile(ProfileType.HIGH_PERFORMANCE)
        
        # Convert to dictionary
        profile_dict = profile.to_dict()
        
        # Verify all required fields are present
        required_fields = [
            'name', 'description', 'buffer_strategy', 'queue_size',
            'blocksize', 'latency', 'enable_adaptive_management'
        ]
        
        for field in required_fields:
            assert field in profile_dict
        
        # Verify values are serializable
        import json
        json_str = json.dumps(profile_dict, default=str)
        assert len(json_str) > 0
        
        # Verify buffer strategy is serialized as string
        assert isinstance(profile_dict['buffer_strategy'], str)
        assert profile_dict['buffer_strategy'] in ['conservative', 'balanced', 'low_latency']


if __name__ == '__main__':
    pytest.main([__file__])