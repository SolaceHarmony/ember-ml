"""
Binary wave neural processing components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Tuple
from dataclasses import dataclass
from ember_ml.nn import tensor

@dataclass
class WaveConfig:
    """Configuration for binary wave processing."""
    grid_size: int = 4
    num_phases: int = 8
    fade_rate: float = 0.1
    threshold: float = 0.5

class BinaryWave(nn.Module):
    """Base class for binary wave processing."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize binary wave processor.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
        # Learnable parameters
        self.phase_shift = nn.Parameter(torch.zeros(config.num_phases))
        self.amplitude_scale = nn.Parameter(torch.ones(config.num_phases))
        
    def encode(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Encode input into wave pattern.

        Args:
            x: Input tensor

        Returns:
            Wave pattern
        """
        # Project to phase space
        phases = torch.arange(self.config.num_phases).float()
        phases = phases + self.phase_shift
        
        # Generate wave pattern
        t = torch.linspace(0, 1, self.config.grid_size * self.config.grid_size)
        wave = torch.sin(2 * torch.pi * phases.unsqueeze(-1) * t.unsqueeze(0))
        wave = wave * self.amplitude_scale.unsqueeze(-1)
        
        # Apply input modulation
        x_flat = x.view(-1)
        wave = wave * x_flat.unsqueeze(0)
        
        # Reshape to grid
        wave = wave.view(
            self.config.num_phases,
            self.config.grid_size,
            self.config.grid_size
        )
        
        return wave
    
    def decode(self, wave: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Decode wave pattern to output.

        Args:
            wave: Wave pattern

        Returns:
            Decoded output
        """
        # Apply inverse wave transform
        wave_flat = wave.view(self.config.num_phases, -1)
        phases = torch.arange(self.config.num_phases).float()
        phases = phases + self.phase_shift
        
        t = torch.linspace(0, 1, wave_flat.size(1))
        basis = torch.sin(2 * torch.pi * phases.unsqueeze(-1) * t.unsqueeze(0))
        basis = basis * self.amplitude_scale.unsqueeze(-1)
        
        # Solve for output
        output = ops.matmul(
            torch.pinverse(basis),
            wave_flat
        )
        
        # Reshape to grid
        output = output.view(
            self.config.grid_size,
            self.config.grid_size
        )
        
        return output
    
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Process input through wave transform.

        Args:
            x: Input tensor

        Returns:
            Processed output
        """
        wave = self.encode(x)
        return self.decode(wave)

class BinaryWaveProcessor(nn.Module):
    """Processes binary wave patterns."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize processor.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
    def wave_interference(self,
                         wave1: tensor.convert_to_tensor,
                         wave2: tensor.convert_to_tensor,
                         mode: str = 'XOR') -> tensor.convert_to_tensor:
        """
        Apply wave interference between two patterns.
        
        Args:
            wave1, wave2: Binary wave patterns
            mode: Interference type ('XOR', 'AND', or 'OR')
            
        Returns:
            Interference pattern
        """
        # Threshold to binary
        binary1 = wave1 > self.config.threshold
        binary2 = wave2 > self.config.threshold
        
        if mode == 'XOR':
            result = torch.logical_xor(binary1, binary2)
        elif mode == 'AND':
            result = torch.logical_and(binary1, binary2)
        else:  # OR
            result = torch.logical_or(binary1, binary2)
            
        return result.float()
    
    def phase_similarity(self,
                        wave1: tensor.convert_to_tensor,
                        wave2: tensor.convert_to_tensor,
                        max_shift: Optional[int] = None) -> Dict[str, tensor.convert_to_tensor]:
        """
        Calculate similarity allowing for phase shifts.
        
        Args:
            wave1, wave2: Binary wave patterns
            max_shift: Maximum phase shift to try
            
        Returns:
            Dict containing similarity metrics
        """
        if max_shift is None:
            max_shift = self.config.num_phases // 4
            
        best_similarity = tensor.convert_to_tensor(0.0, device=wave1.device)
        best_shift = tensor.convert_to_tensor(0, device=wave1.device)
        
        for shift in range(max_shift):
            shifted = torch.roll(wave2, shifts=shift, dims=0)
            similarity = 1.0 - torch.abs(wave1 - shifted).mean()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_shift = tensor.convert_to_tensor(shift, device=wave1.device)
                
        return {
            'similarity': best_similarity,
            'shift': best_shift
        }
    
    def extract_features(self,
                        wave: tensor.convert_to_tensor) -> Dict[str, tensor.convert_to_tensor]:
        """
        Extract characteristic features from wave pattern.
        
        Args:
            wave: Binary wave pattern
            
        Returns:
            Dict of features
        """
        binary = wave > self.config.threshold
        binary_float = binary.float()
        
        # Basic features
        density = binary_float.mean()
        
        # Transitions (changes between 0 and 1)
        transitions = torch.abs(
            binary_float[..., 1:] - binary_float[..., :-1]
        ).sum()
        
        # Symmetry measure
        flipped = torch.flip(binary_float, dims=[-2, -1])
        symmetry = 1.0 - torch.abs(binary_float - flipped).mean()
        
        return {
            'density': density,
            'transitions': transitions,
            'symmetry': symmetry
        }

class BinaryWaveEncoder(nn.Module):
    """Encodes data into binary wave patterns."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize encoder.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
    def encode_char(self, char: str) -> tensor.convert_to_tensor:
        """
        Encode a character into a binary wave pattern.
        
        Args:
            char: Single character to encode
            
        Returns:
            4D tensor of shape (num_phases, grid_size, grid_size, 1)
            representing the binary wave pattern
        """
        # Convert to binary
        code_point = ord(char)
        bin_repr = f"{code_point:016b}"
        
        # Create 2D grid
        bit_matrix = tensor.convert_to_tensor([int(b) for b in bin_repr], dtype=torch.float32)
        bit_matrix = bit_matrix.reshape(self.config.grid_size, self.config.grid_size)
        
        # Generate phase shifts
        time_slices = []
        for t in range(self.config.num_phases):
            # Roll the matrix for phase shift
            shifted = torch.roll(bit_matrix, shifts=t, dims=1)
            
            # Apply fade factor
            fade_factor = max(0.0, 1.0 - t * self.config.fade_rate)
            time_slices.append(shifted * fade_factor)
            
        # Stack into 4D tensor
        wave_pattern = torch.stack(time_slices)
        return wave_pattern.unsqueeze(-1)
    
    def encode_sequence(self, sequence: str) -> tensor.convert_to_tensor:
        """
        Encode a sequence of characters into wave patterns.
        
        Args:
            sequence: String to encode
            
        Returns:
            5D tensor of shape (seq_len, num_phases, grid_size, grid_size, 1)
        """
        patterns = [self.encode_char(c) for c in sequence]
        return torch.stack(patterns)

class BinaryWaveNetwork(nn.Module):
    """Neural network using binary wave processing."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 config: WaveConfig = WaveConfig()):
        """
        Initialize network.

        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = BinaryWaveEncoder(config)
        self.processor = BinaryWaveProcessor(config)
        
        # Learnable parameters
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.wave_proj = nn.Linear(
            hidden_size,
            config.grid_size * config.grid_size
        )
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        # Wave memory
        self.register_buffer(
            'memory_gate',
            torch.randn(config.grid_size, config.grid_size)
        )
        self.register_buffer(
            'update_gate',
            torch.randn(config.grid_size, config.grid_size)
        )
        
    def forward(self,
                x: tensor.convert_to_tensor,
                memory: Optional[tensor.convert_to_tensor] = None) -> Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Process input through binary wave network.
        
        Args:
            x: Input tensor
            memory: Optional previous memory state
            
        Returns:
            (output, new_memory) tuple
        """
        # Project to hidden
        hidden = self.input_proj(x)
        
        # Generate wave pattern
        wave = self.wave_proj(hidden)
        wave = wave.reshape(-1, self.config.grid_size, self.config.grid_size)
        
        # Apply memory gate
        if memory is not None:
            gate = self.processor.wave_interference(
                wave,
                self.memory_gate,
                mode='AND'
            )
            wave = self.processor.wave_interference(
                wave,
                gate,
                mode='AND'
            )
            
            # Update memory
            update = self.processor.wave_interference(
                wave,
                self.update_gate,
                mode='AND'
            )
            memory = self.processor.wave_interference(
                memory,
                update,
                mode='OR'
            )
        else:
            memory = wave
            
        # Project to output
        output = self.output_proj(hidden)
        
        return output, memory