"""
Wave pattern and signal generation components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from .binary_wave import WaveConfig

class SignalSynthesizer:
    """Synthesizer for generating various waveforms."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize signal synthesizer.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def sine_wave(self,
                 frequency: float,
                 duration: float,
                 amplitude: float = 1.0,
                 phase: float = 0.0) -> torch.Tensor:
        """
        Generate sine wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude
            phase: Initial phase in radians

        Returns:
            Tensor containing sine wave
        """
        t = torch.linspace(0, duration, int(duration * self.sampling_rate))
        return amplitude * torch.sin(2 * math.pi * frequency * t + phase)
        
    def square_wave(self,
                   frequency: float,
                   duration: float,
                   amplitude: float = 1.0,
                   duty_cycle: float = 0.5) -> torch.Tensor:
        """
        Generate square wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude
            duty_cycle: Duty cycle (0 to 1)

        Returns:
            Tensor containing square wave
        """
        t = torch.linspace(0, duration, int(duration * self.sampling_rate))
        wave = torch.sin(2 * math.pi * frequency * t)
        return amplitude * torch.sign(wave - math.cos(math.pi * duty_cycle))
        
    def sawtooth_wave(self,
                     frequency: float,
                     duration: float,
                     amplitude: float = 1.0) -> torch.Tensor:
        """
        Generate sawtooth wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude

        Returns:
            Tensor containing sawtooth wave
        """
        t = torch.linspace(0, duration, int(duration * self.sampling_rate))
        wave = t * frequency - torch.floor(t * frequency)
        return 2 * amplitude * (wave - 0.5)
        
    def triangle_wave(self,
                     frequency: float,
                     duration: float,
                     amplitude: float = 1.0) -> torch.Tensor:
        """
        Generate triangle wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude

        Returns:
            Tensor containing triangle wave
        """
        t = torch.linspace(0, duration, int(duration * self.sampling_rate))
        wave = t * frequency - torch.floor(t * frequency)
        return 2 * amplitude * torch.abs(2 * wave - 1) - amplitude
        
    def noise(self,
             duration: float,
             amplitude: float = 1.0,
             distribution: str = 'uniform') -> torch.Tensor:
        """
        Generate noise signal.

        Args:
            duration: Signal duration in seconds
            amplitude: Noise amplitude
            distribution: Noise distribution ('uniform' or 'gaussian')

        Returns:
            Tensor containing noise signal
        """
        n_samples = int(duration * self.sampling_rate)
        if distribution == 'uniform':
            return 2 * amplitude * (torch.rand(n_samples) - 0.5)
        elif distribution == 'gaussian':
            return amplitude * torch.randn(n_samples)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

class PatternGenerator:
    """Generator for 2D wave patterns."""
    
    def __init__(self, config: WaveConfig):
        """
        Initialize pattern generator.

        Args:
            config: Wave configuration
        """
        self.config = config
        
    def binary_pattern(self, density: float) -> torch.Tensor:
        """
        Generate binary pattern.

        Args:
            density: Target pattern density (0 to 1)

        Returns:
            Binary pattern tensor
        """
        pattern = torch.rand(self.config.grid_size, self.config.grid_size)
        return (pattern < density).float()
        
    def wave_pattern(self,
                    frequency: float,
                    duration: float) -> torch.Tensor:
        """
        Generate wave-based pattern.

        Args:
            frequency: Pattern frequency
            duration: Time duration for pattern

        Returns:
            Wave pattern tensor
        """
        x = torch.linspace(0, duration, self.config.grid_size)
        y = torch.linspace(0, duration, self.config.grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        pattern = torch.sin(2 * math.pi * frequency * X) * \
                 torch.sin(2 * math.pi * frequency * Y)
        return 0.5 * (pattern + 1)  # Normalize to [0, 1]
        
    def interference_pattern(self,
                           frequencies: List[float],
                           amplitudes: List[float],
                           duration: float) -> torch.Tensor:
        """
        Generate interference pattern from multiple waves.

        Args:
            frequencies: List of wave frequencies
            amplitudes: List of wave amplitudes
            duration: Time duration for pattern

        Returns:
            Interference pattern tensor
        """
        pattern = torch.zeros(self.config.grid_size, self.config.grid_size)
        for freq, amp in zip(frequencies, amplitudes):
            pattern += amp * self.wave_pattern(freq, duration)
        return torch.clamp(pattern, 0, 1)

class WaveGenerator(nn.Module):
    """Neural network-based wave pattern generator."""
    
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 config: WaveConfig):
        """
        Initialize wave generator.

        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            config: Wave configuration
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Generator network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.grid_size * config.grid_size),
            nn.Sigmoid()
        )
        
        # Phase network
        self.phase_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.num_phases),
            nn.Sigmoid()
        )
        
    def forward(self,
                z: torch.Tensor,
                return_phases: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate wave pattern.

        Args:
            z: Latent vector [batch_size, latent_dim]
            return_phases: Whether to return phase information

        Returns:
            Generated pattern tensor [batch_size, grid_size, grid_size]
            and optionally phases [batch_size, num_phases]
        """
        # Generate pattern
        pattern = self.net(z)
        pattern = pattern.view(z.size(0), self.config.grid_size, self.config.grid_size)
        
        if return_phases:
            phases = self.phase_net(z)
            return pattern, phases
        return pattern
        
    def interpolate(self,
                   z1: torch.Tensor,
                   z2: torch.Tensor,
                   steps: int) -> torch.Tensor:
        """
        Interpolate between two latent vectors.

        Args:
            z1: First latent vector [latent_dim]
            z2: Second latent vector [latent_dim]
            steps: Number of interpolation steps

        Returns:
            Tensor of interpolated patterns [steps, grid_size, grid_size]
        """
        alphas = torch.linspace(0, 1, steps)
        patterns = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z = z.unsqueeze(0)  # Add batch dimension
            pattern = self(z)
            patterns.append(pattern.squeeze(0))  # Remove batch dimension
            
        return torch.stack(patterns)
        
    def random_sample(self,
                     num_samples: int,
                     seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random patterns.

        Args:
            num_samples: Number of patterns to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of generated patterns [num_samples, grid_size, grid_size]
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        z = torch.randn(num_samples, self.latent_dim)
        return self(z)