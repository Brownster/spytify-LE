"""
Configuration profiles for different usage scenarios.

This module provides predefined configuration profiles optimized for
different environments and use cases, with automatic system capability
detection and profile selection.
"""

import logging
import os
import platform
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import subprocess

from .buffer_management import BufferStrategy, AudioSettings

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Available configuration profiles."""
    HEADLESS = "headless"
    DESKTOP = "desktop"
    HIGH_PERFORMANCE = "high_performance"
    AUTO = "auto"


@dataclass
class SystemCapabilities:
    """System capability assessment for profile selection."""
    cpu_cores: int
    memory_gb: float
    is_headless: bool
    audio_backend: str
    has_gui: bool
    system_load: float
    
    def __post_init__(self):
        """Validate system capabilities."""
        if self.cpu_cores < 1:
            raise ValueError("CPU cores must be at least 1")
        if self.memory_gb < 0:
            raise ValueError("Memory GB cannot be negative")


@dataclass
class ConfigProfile:
    """Configuration profile with optimized settings."""
    name: str
    description: str
    buffer_strategy: BufferStrategy
    queue_size: int
    blocksize: int
    latency: float
    collection_interval: float
    enable_debug_mode: bool
    enable_adaptive_management: bool
    enable_health_monitoring: bool
    enable_metrics_collection: bool
    max_reconnection_attempts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'buffer_strategy': self.buffer_strategy.value,
            'queue_size': self.queue_size,
            'blocksize': self.blocksize,
            'latency': self.latency,
            'collection_interval': self.collection_interval,
            'enable_debug_mode': self.enable_debug_mode,
            'enable_adaptive_management': self.enable_adaptive_management,
            'enable_health_monitoring': self.enable_health_monitoring,
            'enable_metrics_collection': self.enable_metrics_collection,
            'max_reconnection_attempts': self.max_reconnection_attempts
        }


class SystemCapabilityDetector:
    """Detects system capabilities for automatic profile selection."""
    
    @staticmethod
    def detect_capabilities() -> SystemCapabilities:
        """
        Detect current system capabilities.
        
        Returns:
            SystemCapabilities object with detected values
        """
        try:
            # CPU information
            cpu_cores = psutil.cpu_count(logical=True) or 1
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024 ** 3)
            
            # System load
            system_load = psutil.cpu_percent(interval=1.0) / 100.0
            
            # Detect if running headless
            is_headless = SystemCapabilityDetector._detect_headless()
            
            # Detect audio backend
            audio_backend = SystemCapabilityDetector._detect_audio_backend()
            
            # Detect GUI availability
            has_gui = SystemCapabilityDetector._detect_gui()
            
            return SystemCapabilities(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                is_headless=is_headless,
                audio_backend=audio_backend,
                has_gui=has_gui,
                system_load=system_load
            )
            
        except Exception as e:
            logger.error("Error detecting system capabilities: %s", e)
            # Return conservative defaults
            return SystemCapabilities(
                cpu_cores=2,
                memory_gb=4.0,
                is_headless=True,
                audio_backend="pulseaudio",
                has_gui=False,
                system_load=0.5
            )
    
    @staticmethod
    def _detect_headless() -> bool:
        """Detect if system is running headless."""
        # Check for display environment variables
        display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE']
        has_display = any(os.environ.get(var) for var in display_vars)
        
        # Check if we're in a container or SSH session
        in_container = os.path.exists('/.dockerenv') or os.environ.get('container') is not None
        in_ssh = os.environ.get('SSH_CLIENT') is not None or os.environ.get('SSH_TTY') is not None
        
        # Consider headless if no display or in container/SSH
        return not has_display or in_container or in_ssh
    
    @staticmethod
    def _detect_audio_backend() -> str:
        """Detect the audio backend in use."""
        try:
            # Check for PipeWire
            result = subprocess.run(['pgrep', 'pipewire'], capture_output=True, text=True)
            if result.returncode == 0:
                return "pipewire"
            
            # Check for PulseAudio
            result = subprocess.run(['pgrep', 'pulseaudio'], capture_output=True, text=True)
            if result.returncode == 0:
                return "pulseaudio"
            
            # Check for ALSA
            if os.path.exists('/proc/asound/cards'):
                return "alsa"
            
            return "unknown"
            
        except Exception:
            return "pulseaudio"  # Default assumption
    
    @staticmethod
    def _detect_gui() -> bool:
        """Detect if GUI is available."""
        # Check for common desktop environments
        desktop_env = os.environ.get('XDG_CURRENT_DESKTOP')
        if desktop_env:
            return True
        
        # Check for X11 or Wayland
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            return True
        
        return False


class ProfileManager:
    """Manages configuration profiles and automatic selection."""
    
    # Predefined profiles
    PROFILES = {
        ProfileType.HEADLESS: ConfigProfile(
            name="headless",
            description="Optimized for spotifyd and headless operation with stability focus",
            buffer_strategy=BufferStrategy.CONSERVATIVE,
            queue_size=400,
            blocksize=4096,
            latency=0.2,  # 200ms for stability
            collection_interval=2.0,
            enable_debug_mode=False,
            enable_adaptive_management=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            max_reconnection_attempts=10
        ),
        
        ProfileType.DESKTOP: ConfigProfile(
            name="desktop",
            description="Balanced settings for desktop use with GUI feedback",
            buffer_strategy=BufferStrategy.BALANCED,
            queue_size=250,
            blocksize=2048,
            latency=0.1,  # 100ms balanced
            collection_interval=1.0,
            enable_debug_mode=False,
            enable_adaptive_management=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            max_reconnection_attempts=5
        ),
        
        ProfileType.HIGH_PERFORMANCE: ConfigProfile(
            name="high_performance",
            description="Minimal latency for high-performance systems",
            buffer_strategy=BufferStrategy.LOW_LATENCY,
            queue_size=150,
            blocksize=1024,
            latency=0.05,  # 50ms minimal
            collection_interval=0.5,
            enable_debug_mode=True,
            enable_adaptive_management=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            max_reconnection_attempts=3
        )
    }
    
    @classmethod
    def get_profile(cls, profile_type: ProfileType) -> ConfigProfile:
        """
        Get a configuration profile by type.
        
        Args:
            profile_type: The profile type to retrieve
            
        Returns:
            ConfigProfile for the specified type
        """
        if profile_type == ProfileType.AUTO:
            return cls.select_optimal_profile()
        
        return cls.PROFILES[profile_type]
    
    @classmethod
    def select_optimal_profile(cls, capabilities: Optional[SystemCapabilities] = None) -> ConfigProfile:
        """
        Automatically select the optimal profile based on system capabilities.
        
        Args:
            capabilities: Optional pre-detected capabilities
            
        Returns:
            ConfigProfile best suited for the current system
        """
        if capabilities is None:
            capabilities = SystemCapabilityDetector.detect_capabilities()
        
        logger.info(
            "System capabilities: cores=%d, memory=%.1fGB, headless=%s, load=%.2f",
            capabilities.cpu_cores, capabilities.memory_gb, 
            capabilities.is_headless, capabilities.system_load
        )
        
        # Decision logic for profile selection
        if capabilities.is_headless:
            # Headless systems prioritize stability
            selected_profile = cls.PROFILES[ProfileType.HEADLESS]
            logger.info("Selected headless profile for headless system")
            
        elif capabilities.cpu_cores >= 8 and capabilities.memory_gb >= 16 and capabilities.system_load < 0.3:
            # High-performance systems with low load
            selected_profile = cls.PROFILES[ProfileType.HIGH_PERFORMANCE]
            logger.info("Selected high-performance profile for powerful system")
            
        else:
            # Default to balanced desktop profile
            selected_profile = cls.PROFILES[ProfileType.DESKTOP]
            logger.info("Selected desktop profile for balanced operation")
        
        # Apply system-specific adjustments
        adjusted_profile = cls._adjust_profile_for_system(selected_profile, capabilities)
        
        return adjusted_profile
    
    @classmethod
    def _adjust_profile_for_system(
        cls, 
        base_profile: ConfigProfile, 
        capabilities: SystemCapabilities
    ) -> ConfigProfile:
        """
        Adjust profile settings based on specific system characteristics.
        
        Args:
            base_profile: Base profile to adjust
            capabilities: System capabilities
            
        Returns:
            Adjusted ConfigProfile
        """
        # Create a copy of the base profile
        adjusted = ConfigProfile(
            name=f"{base_profile.name}_adjusted",
            description=f"{base_profile.description} (system-adjusted)",
            buffer_strategy=base_profile.buffer_strategy,
            queue_size=base_profile.queue_size,
            blocksize=base_profile.blocksize,
            latency=base_profile.latency,
            collection_interval=base_profile.collection_interval,
            enable_debug_mode=base_profile.enable_debug_mode,
            enable_adaptive_management=base_profile.enable_adaptive_management,
            enable_health_monitoring=base_profile.enable_health_monitoring,
            enable_metrics_collection=base_profile.enable_metrics_collection,
            max_reconnection_attempts=base_profile.max_reconnection_attempts
        )
        
        # Adjust based on memory constraints
        if capabilities.memory_gb < 2.0:
            # Low memory - reduce buffer sizes and disable some features
            adjusted.queue_size = max(100, int(adjusted.queue_size * 0.7))
            adjusted.enable_debug_mode = False
            adjusted.collection_interval = min(2.0, adjusted.collection_interval * 1.5)
            logger.info("Adjusted profile for low memory system")
            
        elif capabilities.memory_gb > 8.0:
            # High memory - can afford larger buffers
            adjusted.queue_size = min(600, int(adjusted.queue_size * 1.3))
            logger.info("Adjusted profile for high memory system")
        
        # Adjust based on CPU constraints
        if capabilities.cpu_cores <= 2:
            # Low CPU - reduce processing overhead
            adjusted.collection_interval = max(1.0, adjusted.collection_interval * 1.5)
            adjusted.enable_debug_mode = False
            logger.info("Adjusted profile for low CPU system")
            
        elif capabilities.cpu_cores >= 8:
            # High CPU - can handle more frequent monitoring
            adjusted.collection_interval = max(0.5, adjusted.collection_interval * 0.8)
            logger.info("Adjusted profile for high CPU system")
        
        # Adjust based on current system load
        if capabilities.system_load > 0.8:
            # High load - conservative settings
            adjusted.buffer_strategy = BufferStrategy.CONSERVATIVE
            adjusted.queue_size = min(500, int(adjusted.queue_size * 1.2))
            adjusted.latency = max(0.1, adjusted.latency * 1.2)
            adjusted.collection_interval = max(1.0, adjusted.collection_interval * 1.5)
            logger.info("Adjusted profile for high system load")
        
        # Adjust based on audio backend
        if capabilities.audio_backend == "pipewire":
            # PipeWire can handle lower latencies better
            adjusted.latency = max(0.05, adjusted.latency * 0.8)
            logger.info("Adjusted profile for PipeWire backend")
            
        elif capabilities.audio_backend == "alsa":
            # ALSA might need more conservative settings
            adjusted.latency = max(0.1, adjusted.latency * 1.2)
            adjusted.buffer_strategy = BufferStrategy.CONSERVATIVE
            logger.info("Adjusted profile for ALSA backend")
        
        return adjusted
    
    @classmethod
    def list_available_profiles(cls) -> Dict[str, str]:
        """
        List all available profiles with descriptions.
        
        Returns:
            Dictionary mapping profile names to descriptions
        """
        profiles = {}
        for profile_type, profile in cls.PROFILES.items():
            profiles[profile_type.value] = profile.description
        profiles["auto"] = "Automatically select optimal profile based on system capabilities"
        
        return profiles
    
    @classmethod
    def create_custom_profile(
        cls,
        name: str,
        base_profile: ProfileType = ProfileType.DESKTOP,
        **overrides
    ) -> ConfigProfile:
        """
        Create a custom profile based on an existing profile with overrides.
        
        Args:
            name: Name for the custom profile
            base_profile: Base profile to start from
            **overrides: Settings to override
            
        Returns:
            Custom ConfigProfile
        """
        base = cls.PROFILES[base_profile]
        
        # Create new profile with overrides
        custom_profile = ConfigProfile(
            name=name,
            description=overrides.get('description', f"Custom profile based on {base.name}"),
            buffer_strategy=overrides.get('buffer_strategy', base.buffer_strategy),
            queue_size=overrides.get('queue_size', base.queue_size),
            blocksize=overrides.get('blocksize', base.blocksize),
            latency=overrides.get('latency', base.latency),
            collection_interval=overrides.get('collection_interval', base.collection_interval),
            enable_debug_mode=overrides.get('enable_debug_mode', base.enable_debug_mode),
            enable_adaptive_management=overrides.get('enable_adaptive_management', base.enable_adaptive_management),
            enable_health_monitoring=overrides.get('enable_health_monitoring', base.enable_health_monitoring),
            enable_metrics_collection=overrides.get('enable_metrics_collection', base.enable_metrics_collection),
            max_reconnection_attempts=overrides.get('max_reconnection_attempts', base.max_reconnection_attempts)
        )
        
        logger.info("Created custom profile: %s", name)
        return custom_profile