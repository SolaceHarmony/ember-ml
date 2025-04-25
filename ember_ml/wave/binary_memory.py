"""
Wave-based memory storage and pattern retrieval mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from .binary_wave import WaveConfig, BinaryWave

@dataclass
class MemoryPattern:
    """Container for stored wave patterns."""
    
    pattern: tensor.convert_to_tensor
    timestamp: float
    metadata: Dict[str, Any]
    
    def similarity(self, other: 'MemoryPattern') -> float:
        """
        Compute similarity with another pattern.

        Args:
            other: Pattern to compare with

        Returns:
            Similarity score
        """
        return 1.0 - torch.mean(torch.abs(self.pattern - other.pattern)).item()

class WaveStorage:
    """Storage mechanism for wave patterns."""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize wave storage.

        Args:
            capacity: Maximum number of patterns to store
        """
        self.capacity = capacity
        self.patterns: List[MemoryPattern] = []
        
    def store(self,
              pattern: tensor.convert_to_tensor,
              timestamp: float,
              metadata: Optional[Dict[str, Any]] = None):
        """
        Store a wave pattern.

        Args:
            pattern: Wave pattern tensor
            timestamp: Storage timestamp
            metadata: Optional pattern metadata
        """
        if metadata is None:
            metadata = {}
            
        memory_pattern = MemoryPattern(
            pattern=pattern.clone(),
            timestamp=timestamp,
            metadata=metadata
        )
        
        self.patterns.append(memory_pattern)
        
        # Remove oldest if capacity exceeded
        if len(self.patterns) > self.capacity:
            self.patterns.pop(0)
            
    def retrieve(self,
                query_pattern: tensor.convert_to_tensor,
                threshold: float = 0.8) -> List[MemoryPattern]:
        """
        Retrieve similar patterns.

        Args:
            query_pattern: Pattern to search for
            threshold: Similarity threshold

        Returns:
            List of matching patterns
        """
        query = MemoryPattern(
            pattern=query_pattern,
            timestamp=0.0,
            metadata={}
        )
        
        matches = []
        for pattern in self.patterns:
            if pattern.similarity(query) >= threshold:
                matches.append(pattern)
                
        return matches
    
    def clear(self):
        """Clear all stored patterns."""
        self.patterns.clear()

class BinaryMemory(nn.Module):
    """Wave-based memory system with interference-based storage and retrieval."""
    
    def __init__(self,
                 config: WaveConfig = WaveConfig(),
                 capacity: int = 1000):
        """
        Initialize binary memory.

        Args:
            config: Wave configuration
            capacity: Memory capacity
        """
        super().__init__()
        self.config = config
        self.wave_processor = BinaryWave(config)
        self.storage = WaveStorage(capacity)
        
        # Learnable components
        self.store_gate = nn.Sequential(
            nn.Linear(config.grid_size * config.grid_size, config.grid_size * config.grid_size),
            nn.Sigmoid()
        )
        
        self.retrieve_gate = nn.Sequential(
            nn.Linear(config.grid_size * config.grid_size, config.grid_size * config.grid_size),
            nn.Sigmoid()
        )
        
    def store_pattern(self,
                     pattern: tensor.convert_to_tensor,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Store a pattern in memory.

        Args:
            pattern: Input pattern
            metadata: Optional pattern metadata
        """
        # Convert to wave pattern
        wave = self.wave_processor.encode(pattern)
        
        # Apply storage gate
        flat_wave = wave.view(-1, self.config.grid_size * self.config.grid_size)
        gated = self.store_gate(flat_wave)
        gated = gated.view_as(wave)
        
        # Store with current timestamp
        self.storage.store(
            gated,
            timestamp=torch.cuda.current_stream().elapsed_time(None)
            if torch.cuda.is_available()
            else tensor.convert_to_tensor(0.0),
            metadata=metadata
        )
        
    def retrieve_pattern(self,
                        query: tensor.convert_to_tensor,
                        threshold: float = 0.8) -> Tuple[List[tensor.convert_to_tensor], List[Dict[str, Any]]]:
        """
        Retrieve patterns from memory.

        Args:
            query: Query pattern
            threshold: Similarity threshold

        Returns:
            Tuple of (retrieved patterns, metadata)
        """
        # Convert query to wave pattern
        wave_query = self.wave_processor.encode(query)
        
        # Apply retrieval gate
        flat_query = wave_query.view(-1, self.config.grid_size * self.config.grid_size)
        gated = self.retrieve_gate(flat_query)
        gated = gated.view_as(wave_query)
        
        # Retrieve similar patterns
        matches = self.storage.retrieve(gated, threshold)
        
        # Decode and return patterns with metadata
        patterns = []
        metadata = []
        for match in matches:
            decoded = self.wave_processor.decode(match.pattern)
            patterns.append(decoded)
            metadata.append(match.metadata)
            
        return patterns, metadata
    
    def clear_memory(self):
        """Clear all stored patterns."""
        self.storage.clear()
        
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state.

        Returns:
            Dictionary containing memory state
        """
        return {
            'num_patterns': len(self.storage.patterns),
            'capacity': self.storage.capacity,
            'patterns': [
                {
                    'timestamp': p.timestamp,
                    'metadata': p.metadata
                }
                for p in self.storage.patterns
            ]
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save memory system state."""
        return {
            'config': {
                'grid_size': self.config.grid_size,
                'num_phases': self.config.num_phases,
                'fade_rate': self.config.fade_rate,
                'threshold': self.config.threshold
            },
            'storage_capacity': self.storage.capacity,
            'store_gate': self.store_gate.state_dict(),
            'retrieve_gate': self.retrieve_gate.state_dict(),
            'patterns': [
                {
                    'pattern': p.pattern.tolist(),
                    'timestamp': p.timestamp,
                    'metadata': p.metadata
                }
                for p in self.storage.patterns
            ]
        }
    
    def load_state(self, state_dict: Dict[str, Any]):
        """
        Load memory system state.

        Args:
            state_dict: Saved state dictionary
        """
        self.config = WaveConfig(
            grid_size=state_dict['config']['grid_size'],
            num_phases=state_dict['config']['num_phases'],
            fade_rate=state_dict['config']['fade_rate'],
            threshold=state_dict['config']['threshold']
        )
        
        self.storage.capacity = state_dict['storage_capacity']
        self.store_gate.load_state_dict(state_dict['store_gate'])
        self.retrieve_gate.load_state_dict(state_dict['retrieve_gate'])
        
        self.storage.patterns = [
            MemoryPattern(
                pattern=tensor.convert_to_tensor(p['pattern']),
                timestamp=p['timestamp'],
                metadata=p['metadata']
            )
            for p in state_dict['patterns']
        ]