"""
Comprehensive tests for configuration profiles functionality.

Tests profile creation, selection, system capability detection,
and automatic profile adjustment based on environment.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import psutil

from spotify_splitter.config_profiles import (
    ProfileManager, ProfileType, SystemCapabilityDetector, 
    SystemCapabilities, ConfigProfile
)
from spotify_splitter.buffer_management import BufferStrategy


class TestProfileTypes:
    """Test profile type enumeration and basic functionality."""
    
    def test_profile_type_values(self):
        """Test that all expected profile types exist."""
        expected_types = {"headless", "desktop", "high_performance", "auto"}
        actual_types = {profile_type.value for profile_type in ProfileType}
        assert actual_types == expected_types
    
    def test_profile_type_string_conversion(self):
        """Test profile type string conversion."""
        assert ProfileType.HEADLESS.value == "headless"
        assert ProfileType.DESKTOP.value == "desktop"
        assert ProfileType.HIGH_PERFORMANCE.value == "high_performance"
        assert ProfileType.AUTO.value == "auto"


class TestConfigProfile:
    """Test ConfigProfile dataclass functionality."""
    
    def test_config_profile_creation(self):
        """Test creating a configuration profile."""
        profile = ConfigProfile(
            name="test_profile",
            description="Test profile for unit testing",
            buffer_strategy=BufferStrategy.BALANCED,
            queue_size=200,
            blocksize=2048,
            latency=0.1,
            collection_interval=1.0,
            enable_debug_mode=False,
            enable_adaptive_management=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            max_reconnection_attempts=5
        )
        
        assert profile.name == "test_profile"
        assert profile.buffer_strategy == BufferStrategy.BALANCED
        assert profile.queue_size == 200
        assert profile.enable_adaptive_management is True
    
    def test_config_profile_to_dict(self):
        """Test converting profile to dictionary."""
        profile = ConfigProfile(
            name="test_profile",
            description="Test profile",
            buffer_strategy=BufferStrategy.LOW_LATENCY,
            queue_size=150,
            blocksize=1024,
            latency=0.05,
            collection_interval=0.5,
            enable_debug_mode=True,
            enable_adaptive_management=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            max_reconnection_attempts=3
        )
        
        profile_dict = profile.to_dict()
        
        assert profile_dict["name"] == "test_profile"
        assert profile_dict["buffer_strategy"] == "low_latency"
        assert profile_dict["queue_size"] == 150
        assert profile_dict["enable_debug_mode"] is True


class TestSystemCapabilities:
    """Test SystemCapabilities dataclass and validation."""
    
    def test_system_capabilities_creation(self):
        """Test creating system capabilities."""
        caps = SystemCapabilities(
            cpu_cores=4,
            memory_gb=8.0,
            is_headless=False,
            audio_backend="pulseaudio",
            has_gui=True,
            system_load=0.3
        )
        
        assert caps.cpu_cores == 4
        assert caps.memory_gb == 8.0
        assert caps.is_headless is False
        assert caps.audio_backend == "pulseaudio"
        assert caps.has_gui is True
        assert caps.system_load == 0.3
    
    def test_system_capabilities_validation(self):
        """Test system capabilities validation."""
        # Test invalid CPU cores
        with pytest.raises(ValueError, match="CPU cores must be at least 1"):
            SystemCapabilities(
                cpu_cores=0,
                memory_gb=4.0,
                is_headless=False,
                audio_backend="pulseaudio",
                has_gui=True,
                system_load=0.3
            )
        
        # Test invalid memory
        with pytest.raises(ValueError, match="Memory GB cannot be negative"):
            SystemCapabilities(
                cpu_cores=2,
                memory_gb=-1.0,
                is_headless=False,
                audio_backend="pulseaudio",
                has_gui=True,
                system_load=0.3
            )


class TestSystemCapabilityDetector:
    """Test system capability detection functionality."""
    
    @patch('spotify_splitter.config_profiles.psutil.cpu_count')
    @patch('spotify_splitter.config_profiles.psutil.virtual_memory')
    @patch('spotify_splitter.config_profiles.psutil.cpu_percent')
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector._detect_headless')
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector._detect_audio_backend')
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector._detect_gui')
    def test_detect_capabilities(
        self, mock_detect_gui, mock_detect_audio, mock_detect_headless,
        mock_cpu_percent, mock_virtual_memory, mock_cpu_count
    ):
        """Test complete capability detection."""
        # Setup mocks
        mock_cpu_count.return_value = 8
        mock_memory = MagicMock()
        mock_memory.total = 16 * (1024 ** 3)  # 16GB in bytes
        mock_virtual_memory.return_value = mock_memory
        mock_cpu_percent.return_value = 25.0
        mock_detect_headless.return_value = False
        mock_detect_audio.return_value = "pipewire"
        mock_detect_gui.return_value = True
        
        capabilities = SystemCapabilityDetector.detect_capabilities()
        
        assert capabilities.cpu_cores == 8
        assert capabilities.memory_gb == 16.0
        assert capabilities.system_load == 0.25
        assert capabilities.is_headless is False
        assert capabilities.audio_backend == "pipewire"
        assert capabilities.has_gui is True
    
    @patch('spotify_splitter.config_profiles.os.environ.get')
    @patch('spotify_splitter.config_profiles.os.path.exists')
    def test_detect_headless(self, mock_path_exists, mock_env_get):
        """Test headless detection logic."""
        # Test with display environment
        mock_env_get.side_effect = lambda var, default=None: {
            'DISPLAY': ':0',
            'WAYLAND_DISPLAY': None,
            'XDG_SESSION_TYPE': None,
            'container': None,
            'SSH_CLIENT': None,
            'SSH_TTY': None
        }.get(var, default)
        mock_path_exists.return_value = False
        
        assert SystemCapabilityDetector._detect_headless() is False
        
        # Test headless (no display)
        mock_env_get.side_effect = lambda var, default=None: {
            'DISPLAY': None,
            'WAYLAND_DISPLAY': None,
            'XDG_SESSION_TYPE': None,
            'container': None,
            'SSH_CLIENT': None,
            'SSH_TTY': None
        }.get(var, default)
        
        assert SystemCapabilityDetector._detect_headless() is True
        
        # Test container environment
        mock_env_get.side_effect = lambda var, default=None: {
            'DISPLAY': ':0',
            'container': 'docker',
            'SSH_CLIENT': None,
            'SSH_TTY': None
        }.get(var, default)
        mock_path_exists.return_value = True  # /.dockerenv exists
        
        assert SystemCapabilityDetector._detect_headless() is True
    
    @patch('spotify_splitter.config_profiles.subprocess.run')
    @patch('spotify_splitter.config_profiles.os.path.exists')
    def test_detect_audio_backend(self, mock_path_exists, mock_subprocess):
        """Test audio backend detection."""
        # Test PipeWire detection
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # pipewire found
        ]
        
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "pipewire"
        
        # Reset mock for next test
        mock_subprocess.reset_mock()
        mock_subprocess.side_effect = [
            MagicMock(returncode=1),  # pipewire not found
            MagicMock(returncode=0),  # pulseaudio found
        ]
        
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "pulseaudio"
        
        # Reset mock for ALSA test
        mock_subprocess.reset_mock()
        mock_subprocess.side_effect = [
            MagicMock(returncode=1),  # pipewire not found
            MagicMock(returncode=1),  # pulseaudio not found
        ]
        mock_path_exists.return_value = True  # /proc/asound/cards exists
        
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "alsa"
        
        # Reset mock for unknown test
        mock_subprocess.reset_mock()
        mock_subprocess.side_effect = [
            MagicMock(returncode=1),  # pipewire not found
            MagicMock(returncode=1),  # pulseaudio not found
        ]
        mock_path_exists.return_value = False  # no /proc/asound/cards
        
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "unknown"
        
        # Test exception handling (should return pulseaudio default)
        mock_subprocess.reset_mock()
        mock_subprocess.side_effect = Exception("Command failed")
        
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "pulseaudio"
    
    @patch('spotify_splitter.config_profiles.os.environ.get')
    def test_detect_gui(self, mock_env_get):
        """Test GUI detection logic."""
        # Test with desktop environment
        mock_env_get.side_effect = lambda var, default=None: {
            'XDG_CURRENT_DESKTOP': 'GNOME',
            'DISPLAY': None,
            'WAYLAND_DISPLAY': None
        }.get(var, default)
        
        assert SystemCapabilityDetector._detect_gui() is True
        
        # Test with X11 display
        mock_env_get.side_effect = lambda var, default=None: {
            'XDG_CURRENT_DESKTOP': None,
            'DISPLAY': ':0',
            'WAYLAND_DISPLAY': None
        }.get(var, default)
        
        assert SystemCapabilityDetector._detect_gui() is True
        
        # Test with Wayland display
        mock_env_get.side_effect = lambda var, default=None: {
            'XDG_CURRENT_DESKTOP': None,
            'DISPLAY': None,
            'WAYLAND_DISPLAY': 'wayland-0'
        }.get(var, default)
        
        assert SystemCapabilityDetector._detect_gui() is True
        
        # Test no GUI
        mock_env_get.side_effect = lambda var, default=None: {
            'XDG_CURRENT_DESKTOP': None,
            'DISPLAY': None,
            'WAYLAND_DISPLAY': None
        }.get(var, default)
        
        assert SystemCapabilityDetector._detect_gui() is False
    
    @patch('spotify_splitter.config_profiles.psutil.cpu_count')
    @patch('spotify_splitter.config_profiles.psutil.virtual_memory')
    @patch('spotify_splitter.config_profiles.psutil.cpu_percent')
    def test_detect_capabilities_error_handling(
        self, mock_cpu_percent, mock_virtual_memory, mock_cpu_count
    ):
        """Test capability detection error handling."""
        # Simulate error in detection
        mock_cpu_count.side_effect = Exception("CPU detection failed")
        
        capabilities = SystemCapabilityDetector.detect_capabilities()
        
        # Should return conservative defaults
        assert capabilities.cpu_cores == 2
        assert capabilities.memory_gb == 4.0
        assert capabilities.is_headless is True
        assert capabilities.audio_backend == "pulseaudio"
        assert capabilities.has_gui is False
        assert capabilities.system_load == 0.5


class TestProfileManager:
    """Test ProfileManager functionality."""
    
    def test_get_profile_specific_types(self):
        """Test getting specific profile types."""
        # Test headless profile
        headless = ProfileManager.get_profile(ProfileType.HEADLESS)
        assert headless.name == "headless"
        assert headless.buffer_strategy == BufferStrategy.CONSERVATIVE
        assert headless.queue_size >= 300
        assert headless.latency >= 0.15
        assert headless.enable_adaptive_management is True
        
        # Test desktop profile
        desktop = ProfileManager.get_profile(ProfileType.DESKTOP)
        assert desktop.name == "desktop"
        assert desktop.buffer_strategy == BufferStrategy.BALANCED
        assert 200 <= desktop.queue_size <= 300
        assert 0.08 <= desktop.latency <= 0.12
        
        # Test high-performance profile
        high_perf = ProfileManager.get_profile(ProfileType.HIGH_PERFORMANCE)
        assert high_perf.name == "high_performance"
        assert high_perf.buffer_strategy == BufferStrategy.LOW_LATENCY
        assert high_perf.queue_size <= 200
        assert high_perf.latency <= 0.06
        assert high_perf.enable_debug_mode is True
    
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector.detect_capabilities')
    def test_get_profile_auto_selection(self, mock_detect):
        """Test automatic profile selection."""
        # Setup mock capabilities
        mock_capabilities = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.4
        )
        mock_detect.return_value = mock_capabilities
        
        profile = ProfileManager.get_profile(ProfileType.AUTO)
        
        # Should return desktop profile for this configuration
        assert "desktop" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.BALANCED
    
    def test_select_optimal_profile_headless(self):
        """Test optimal profile selection for headless systems."""
        headless_caps = SystemCapabilities(
            cpu_cores=2, memory_gb=4.0, is_headless=True,
            audio_backend="pulseaudio", has_gui=False, system_load=0.3
        )
        
        profile = ProfileManager.select_optimal_profile(headless_caps)
        
        assert "headless" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.CONSERVATIVE
    
    def test_select_optimal_profile_high_performance(self):
        """Test optimal profile selection for high-performance systems."""
        high_perf_caps = SystemCapabilities(
            cpu_cores=16, memory_gb=32.0, is_headless=False,
            audio_backend="pipewire", has_gui=True, system_load=0.1
        )
        
        profile = ProfileManager.select_optimal_profile(high_perf_caps)
        
        assert "high_performance" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.LOW_LATENCY
    
    def test_select_optimal_profile_desktop(self):
        """Test optimal profile selection for desktop systems."""
        desktop_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.5
        )
        
        profile = ProfileManager.select_optimal_profile(desktop_caps)
        
        assert "desktop" in profile.name.lower()
        assert profile.buffer_strategy == BufferStrategy.BALANCED
    
    def test_adjust_profile_for_low_memory(self):
        """Test profile adjustment for low memory systems."""
        low_memory_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=1.5, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, low_memory_caps)
        
        # Should reduce buffer size and disable debug mode
        assert adjusted.queue_size < base_profile.queue_size
        assert adjusted.enable_debug_mode is False
        assert adjusted.collection_interval >= base_profile.collection_interval
    
    def test_adjust_profile_for_high_memory(self):
        """Test profile adjustment for high memory systems."""
        high_memory_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=16.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, high_memory_caps)
        
        # Should increase buffer size
        assert adjusted.queue_size > base_profile.queue_size
    
    def test_adjust_profile_for_low_cpu(self):
        """Test profile adjustment for low CPU systems."""
        low_cpu_caps = SystemCapabilities(
            cpu_cores=2, memory_gb=4.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, low_cpu_caps)
        
        # Should reduce processing overhead
        assert adjusted.collection_interval >= base_profile.collection_interval
        assert adjusted.enable_debug_mode is False
    
    def test_adjust_profile_for_high_cpu(self):
        """Test profile adjustment for high CPU systems."""
        high_cpu_caps = SystemCapabilities(
            cpu_cores=16, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.2
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, high_cpu_caps)
        
        # Should enable more frequent monitoring
        assert adjusted.collection_interval <= base_profile.collection_interval
    
    def test_adjust_profile_for_high_load(self):
        """Test profile adjustment for high system load."""
        high_load_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.9
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, high_load_caps)
        
        # Should use conservative settings
        assert adjusted.buffer_strategy == BufferStrategy.CONSERVATIVE
        assert adjusted.queue_size >= base_profile.queue_size
        assert adjusted.latency >= base_profile.latency
        assert adjusted.collection_interval >= base_profile.collection_interval
    
    def test_adjust_profile_for_pipewire(self):
        """Test profile adjustment for PipeWire backend."""
        pipewire_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pipewire", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, pipewire_caps)
        
        # Should reduce latency for PipeWire
        assert adjusted.latency <= base_profile.latency
    
    def test_adjust_profile_for_alsa(self):
        """Test profile adjustment for ALSA backend."""
        alsa_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="alsa", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted = ProfileManager._adjust_profile_for_system(base_profile, alsa_caps)
        
        # Should use more conservative settings for ALSA
        assert adjusted.latency >= base_profile.latency
        assert adjusted.buffer_strategy == BufferStrategy.CONSERVATIVE
    
    def test_list_available_profiles(self):
        """Test listing available profiles."""
        profiles = ProfileManager.list_available_profiles()
        
        expected_profiles = {"headless", "desktop", "high_performance", "auto"}
        assert set(profiles.keys()) == expected_profiles
        
        # Check that descriptions are provided
        for profile_name, description in profiles.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_create_custom_profile(self):
        """Test creating custom profiles."""
        custom = ProfileManager.create_custom_profile(
            name="custom_test",
            base_profile=ProfileType.DESKTOP,
            queue_size=300,
            enable_debug_mode=True,
            description="Custom test profile"
        )
        
        assert custom.name == "custom_test"
        assert custom.description == "Custom test profile"
        assert custom.queue_size == 300
        assert custom.enable_debug_mode is True
        
        # Should inherit other settings from desktop profile
        desktop_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        assert custom.buffer_strategy == desktop_profile.buffer_strategy
        assert custom.blocksize == desktop_profile.blocksize


class TestProfileIntegration:
    """Test integration between profiles and other components."""
    
    def test_profile_to_audio_settings_conversion(self):
        """Test converting profiles to audio settings format."""
        for profile_type in [ProfileType.HEADLESS, ProfileType.DESKTOP, ProfileType.HIGH_PERFORMANCE]:
            profile = ProfileManager.get_profile(profile_type)
            
            # Verify profile has all required settings
            assert hasattr(profile, 'queue_size')
            assert hasattr(profile, 'blocksize')
            assert hasattr(profile, 'latency')
            assert hasattr(profile, 'buffer_strategy')
            
            # Verify settings are reasonable
            assert 50 <= profile.queue_size <= 1000
            assert 512 <= profile.blocksize <= 8192
            assert 0.01 <= profile.latency <= 0.5
            assert profile.buffer_strategy in [
                BufferStrategy.CONSERVATIVE, 
                BufferStrategy.BALANCED, 
                BufferStrategy.LOW_LATENCY
            ]
    
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector.detect_capabilities')
    def test_end_to_end_profile_selection(self, mock_detect):
        """Test complete end-to-end profile selection workflow."""
        # Simulate different system configurations
        test_configs = [
            # Headless server
            SystemCapabilities(
                cpu_cores=2, memory_gb=4.0, is_headless=True,
                audio_backend="pulseaudio", has_gui=False, system_load=0.2
            ),
            # Desktop workstation
            SystemCapabilities(
                cpu_cores=8, memory_gb=16.0, is_headless=False,
                audio_backend="pipewire", has_gui=True, system_load=0.4
            ),
            # High-performance system
            SystemCapabilities(
                cpu_cores=16, memory_gb=32.0, is_headless=False,
                audio_backend="pipewire", has_gui=True, system_load=0.1
            ),
        ]
        
        expected_profiles = ["headless", "desktop", "high_performance"]
        
        for i, config in enumerate(test_configs):
            mock_detect.return_value = config
            
            # Test automatic selection
            auto_profile = ProfileManager.get_profile(ProfileType.AUTO)
            
            # Verify appropriate profile was selected
            assert expected_profiles[i] in auto_profile.name.lower()
            
            # Verify profile is properly configured
            assert auto_profile.queue_size > 0
            assert auto_profile.latency > 0
            assert auto_profile.blocksize > 0