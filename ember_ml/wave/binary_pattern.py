"""
Pattern recognition using wave interference and binary pattern matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from .binary_wave import WaveConfig, BinaryWave

@dataclass
class PatternMatch:
    """Container for pattern matching results."""
    
    similarity: float
    position: Tuple[int, int]
    phase_shift: int
    confidence: float
    
    def __lt__(self, other: 'PatternMatch') -> bool:
        """Compare matches by similarity."""
        return self.similarity < other.similarity

class InterferenceDetector:
    """Detects and analyzes wave interference patterns."""
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize interference detector.

        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold
        
    def detect_interference(self,
                          wave1: torch.Tensor,
                          wave2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect interference patterns between waves.

        Args:
            wave1: First wave pattern
            wave2: Second wave pattern

        Returns:
            Dictionary of interference metrics
        """
        # Compute different interference types
        constructive = wave1 + wave2
        destructive = wave1 - wave2
        multiplicative = wave1 * wave2
        
        # Analyze interference strengths
        total_energy = torch.sum(wave1 ** 2) + torch.sum(wave2 ** 2)
        constructive_strength = torch.sum(constructive ** 2) / total_energy
        destructive_strength = torch.sum(destructive ** 2) / total_energy
        multiplicative_strength = torch.sum(multiplicative ** 2) / total_energy
        
        return {
            'constructive': constructive,
            'destructive': destructive,
            'multiplicative': multiplicative,
            'constructive_strength': constructive_strength,
            'destructive_strength': destructive_strength,
            'multiplicative_strength': multiplicative_strength
        }
    
    def find_resonance(self,
                      wave: torch.Tensor,
                      num_shifts: int = 8) -> Dict[str, Any]:
        """
        Find resonance patterns in wave.

        Args:
            wave: Wave pattern
            num_shifts: Number of phase shifts to try

        Returns:
            Dictionary of resonance metrics
        """
        best_resonance = 0.0
        best_shift = 0
        
        for shift in range(num_shifts):
            shifted = torch.roll(wave, shifts=shift, dims=0)
            interference = wave * shifted
            resonance = torch.mean(interference).item()
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_shift = shift
                
        return {
            'resonance': best_resonance,
            'phase_shift': best_shift,
            'is_resonant': best_resonance > self.threshold
        }

class PatternMatcher:
    """Matches and aligns binary wave patterns."""
    
    def __init__(self,
                 template_size: Tuple[int, int],
                 max_shifts: int = 8):
        """
        Initialize pattern matcher.

        Args:
            template_size: Size of pattern template
            max_shifts: Maximum phase shifts to try
        """
        self.template_size = template_size
        self.max_shifts = max_shifts
        
    def match_pattern(self,
                     template: torch.Tensor,
                     target: torch.Tensor,
                     threshold: float = 0.8) -> List[PatternMatch]:
        """
        Find pattern matches in target.

        Args:
            template: Pattern template
            target: Target to search in
            threshold: Matching threshold

        Returns:
            List of pattern matches
        """
        matches = []
        h, w = self.template_size
        
        # Normalize template
        template = template / torch.norm(template)
        
        # Slide template over target
        for i in range(target.size(0) - h + 1):
            for j in range(target.size(1) - w + 1):
                # Extract window
                window = target[i:i+h, j:j+w]
                window = window / torch.norm(window)
                
                # Try different phase shifts
                best_similarity = 0.0
                best_shift = 0
                
                for shift in range(self.max_shifts):
                    shifted_window = torch.roll(window, shifts=shift, dims=0)
                    similarity = torch.sum(template * shifted_window).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_shift = shift
                
                if best_similarity > threshold:
                    matches.append(PatternMatch(
                        similarity=best_similarity,
                        position=(i, j),
                        phase_shift=best_shift,
                        confidence=best_similarity
                    ))
                    
        # Sort matches by similarity
        matches.sort(reverse=True)
        return matches

class BinaryPattern(nn.Module):
    """Pattern recognition using binary wave interference."""
    
    def __init__(self,
                 config: WaveConfig = WaveConfig(),
                 template_size: Optional[Tuple[int, int]] = None):
        """
        Initialize binary pattern recognizer.

        Args:
            config: Wave configuration
            template_size: Optional template size
        """
        super().__init__()
        self.config = config
        self.wave_processor = BinaryWave(config)
        
        if template_size is None:
            template_size = (config.grid_size, config.grid_size)
            
        self.matcher = PatternMatcher(template_size)
        self.detector = InterferenceDetector()
        
        # Learnable components
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def extract_pattern(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Extract pattern features from wave.

        Args:
            wave: Input wave pattern

        Returns:
            Extracted pattern features
        """
        # Add channel dimension
        x = wave.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Remove channel dimension
        return features.squeeze(1)
    
    def match_pattern(self,
                     template: torch.Tensor,
                     target: torch.Tensor,
                     threshold: float = 0.8) -> List[PatternMatch]:
        """
        Find pattern matches.

        Args:
            template: Pattern template
            target: Target to search in
            threshold: Matching threshold

        Returns:
            List of pattern matches
        """
        # Convert to wave patterns
        template_wave = self.wave_processor.encode(template)
        target_wave = self.wave_processor.encode(target)
        
        # Extract pattern features
        template_features = self.extract_pattern(template_wave)
        target_features = self.extract_pattern(target_wave)
        
        # Find matches
        return self.matcher.match_pattern(
            template_features,
            target_features,
            threshold
        )
    
    def analyze_interference(self,
                           wave1: torch.Tensor,
                           wave2: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze interference between patterns.

        Args:
            wave1: First wave pattern
            wave2: Second wave pattern

        Returns:
            Dictionary of interference metrics
        """
        # Extract features
        features1 = self.extract_pattern(wave1)
        features2 = self.extract_pattern(wave2)
        
        # Detect interference
        interference = self.detector.detect_interference(features1, features2)
        
        # Find resonance
        resonance1 = self.detector.find_resonance(features1)
        resonance2 = self.detector.find_resonance(features2)
        
        return {
            'interference': interference,
            'resonance1': resonance1,
            'resonance2': resonance2
        }
    
    def forward(self,
                input_wave: torch.Tensor,
                template: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process input through pattern recognizer.

        Args:
            input_wave: Input wave pattern
            template: Optional template for matching

        Returns:
            Dictionary of results
        """
        # Extract features
        features = self.extract_pattern(input_wave)
        
        results = {
            'features': features,
            'resonance': self.detector.find_resonance(features)
        }
        
        # Match against template if provided
        if template is not None:
            template_features = self.extract_pattern(template)
            matches = self.matcher.match_pattern(
                template_features,
                features
            )
            results['matches'] = matches
            
        return results