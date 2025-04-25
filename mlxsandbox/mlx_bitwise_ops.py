"""
MLX implementation of basic bitwise operations.

This module provides MLX-based implementations of bitwise operations
that will form the foundation for binary wave neural networks.
"""

import mlx.core as mx
from typing import List, Union, Tuple, Optional

def bitwise_and(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the bitwise AND of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise AND of x and y
    """
    # MLX has direct bitwise operations
    return mx.bitwise_and(x, y)

def bitwise_or(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the bitwise OR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise OR of x and y
    """
    return mx.bitwise_or(x, y)

def bitwise_xor(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the bitwise XOR of x and y element-wise.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Tensor with the bitwise XOR of x and y
    """
    return mx.bitwise_xor(x, y)

def bitwise_not(x: mx.array) -> mx.array:
    """
    Compute the bitwise NOT of x element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with the bitwise NOT of x
    """
    return mx.bitwise_not(x)

def left_shift(x: mx.array, shifts: Union[mx.array, int]) -> mx.array:
    """
    Shift the bits of x to the left by shifts positions.
    
    Args:
        x: Input tensor
        shifts: Number of bits to shift
        
    Returns:
        Tensor with x shifted left by shifts bits
    """
    return mx.left_shift(x, shifts)

def right_shift(x: mx.array, shifts: Union[mx.array, int]) -> mx.array:
    """
    Shift the bits of x to the right by shifts positions.
    
    Args:
        x: Input tensor
        shifts: Number of bits to shift
        
    Returns:
        Tensor with x shifted right by shifts bits
    """
    return mx.right_shift(x, shifts)

def count_ones(x: mx.array) -> mx.array:
    """
    Count the number of 1 bits in each element of x.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with the count of 1 bits in each element of x
    """
    # MLX doesn't have a direct popcount function, so we implement it manually
    # by counting bits using a divide-and-conquer approach
    
    # Ensure x is integer type
    x = mx.astype(x, mx.int32)
    
    # Count bits using divide-and-conquer approach (Hamming weight)
    # This is the classic SWAR algorithm (SIMD Within A Register)
    
    # 32-bit version
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F  # Extract the lower 6 bits (enough for 32-bit integers)

def get_bit(x: mx.array, position: Union[mx.array, int]) -> mx.array:
    """
    Get the bit at the specified position in x.
    
    Args:
        x: Input tensor
        position: Bit position (0-based, from least significant bit)
        
    Returns:
        Tensor with the bit at the specified position in x
    """
    # Create a mask with 1 at the specified position
    mask = mx.left_shift(mx.ones_like(x), position)
    
    # Extract the bit using bitwise AND and right shift
    return mx.right_shift(mx.bitwise_and(x, mask), position)

def set_bit(x: mx.array, position: Union[mx.array, int], value: Union[mx.array, int]) -> mx.array:
    """
    Set the bit at the specified position in x to value.
    
    Args:
        x: Input tensor
        position: Bit position (0-based, from least significant bit)
        value: Bit value (0 or 1)
        
    Returns:
        Tensor with the bit at the specified position in x set to value
    """
    # Create a mask with 1 at the specified position
    mask = mx.left_shift(mx.ones_like(x), position)
    
    # Clear the bit
    cleared = mx.bitwise_and(x, mx.bitwise_not(mask))
    
    # Set the bit if value is 1
    if isinstance(value, int):
        value = mx.array([value])
    value_shifted = mx.left_shift(mx.bitwise_and(value, 1), position)
    
    return mx.bitwise_or(cleared, value_shifted)

def toggle_bit(x: mx.array, position: Union[mx.array, int]) -> mx.array:
    """
    Toggle the bit at the specified position in x.
    
    Args:
        x: Input tensor
        position: Bit position (0-based, from least significant bit)
        
    Returns:
        Tensor with the bit at the specified position in x toggled
    """
    # Create a mask with 1 at the specified position
    mask = mx.left_shift(mx.ones_like(x), position)
    
    # Toggle the bit using XOR
    return mx.bitwise_xor(x, mask)

def binary_wave_interference(waves: List[mx.array], mode: str = 'xor') -> mx.array:
    """
    Apply wave interference between multiple binary patterns.
    
    Args:
        waves: List of binary wave patterns
        mode: Interference type ('xor', 'and', or 'or')
        
    Returns:
        Interference pattern
    """
    if not waves:
        raise ValueError("At least one wave is required")
    
    result = waves[0]
    
    for wave in waves[1:]:
        if mode == 'xor':
            result = mx.bitwise_xor(result, wave)
        elif mode == 'and':
            result = mx.bitwise_and(result, wave)
        elif mode == 'or':
            result = mx.bitwise_or(result, wave)
        else:
            raise ValueError(f"Unsupported interference mode: {mode}")
    
    return result

def binary_wave_propagate(wave: mx.array, shift: Union[mx.array, int]) -> mx.array:
    """
    Propagate a binary wave by shifting it.
    
    Args:
        wave: Binary wave pattern
        shift: Number of positions to shift
        
    Returns:
        Propagated wave pattern
    """
    # Positive shift means left shift, negative shift means right shift
    if isinstance(shift, int):
        if shift >= 0:
            return mx.left_shift(wave, shift)
        else:
            return mx.right_shift(wave, -shift)
    else:
        # For array shifts, we need to handle positive and negative shifts separately
        positive_mask = shift >= 0
        negative_mask = shift < 0
        
        result = mx.zeros_like(wave)
        
        # Apply left shift for positive shifts
        if mx.any(positive_mask):
            pos_shifts = mx.where(positive_mask, shift, 0)
            pos_result = mx.left_shift(wave, pos_shifts)
            result = mx.where(positive_mask, pos_result, result)
        
        # Apply right shift for negative shifts
        if mx.any(negative_mask):
            neg_shifts = mx.where(negative_mask, -shift, 0)
            neg_result = mx.right_shift(wave, neg_shifts)
            result = mx.where(negative_mask, neg_result, result)
        
        return result

def create_duty_cycle(length: int, duty_cycle: float) -> mx.array:
    """
    Create a binary pattern with the specified duty cycle.
    
    Args:
        length: Length of the pattern
        duty_cycle: Fraction of bits that should be 1 (between 0 and 1)
        
    Returns:
        Binary pattern with the specified duty cycle
    """
    if duty_cycle < 0 or duty_cycle > 1:
        raise ValueError("Duty cycle must be between 0 and 1")
    
    # Calculate number of 1 bits
    num_ones = int(length * duty_cycle)
    
    # Create pattern with 1s at the beginning
    pattern = mx.zeros((length,), dtype=mx.int32)
    if num_ones > 0:
        pattern = mx.array([1] * num_ones + [0] * (length - num_ones), dtype=mx.int32)
    
    return pattern

def generate_blocky_sin(length: int, half_period: int) -> mx.array:
    """
    Generate a blocky sine wave pattern.
    
    Args:
        length: Length of the pattern
        half_period: Half the period of the wave
        
    Returns:
        Blocky sine wave pattern
    """
    if half_period <= 0:
        raise ValueError("Half period must be positive")
    
    # Create indices
    indices = mx.arange(length)
    
    # Calculate cycle position
    cycle_position = mx.remainder(indices, 2 * half_period)
    
    # Create pattern: 1 for first half of period, 0 for second half
    pattern = mx.where(cycle_position < half_period, 1, 0)
    
    return pattern

# Test the functions
if __name__ == "__main__":
    # Create test arrays
    a = mx.array([0b1010, 0b1100, 0b1111], dtype=mx.int32)
    b = mx.array([0b0101, 0b1010, 0b0000], dtype=mx.int32)
    
    print("a:", a)
    print("b:", b)
    
    # Test bitwise operations
    print("a & b:", bitwise_and(a, b))
    print("a | b:", bitwise_or(a, b))
    print("a ^ b:", bitwise_xor(a, b))
    print("~a:", bitwise_not(a))
    
    # Test shift operations
    print("a << 1:", left_shift(a, 1))
    print("a >> 1:", right_shift(a, 1))
    
    # Test bit operations
    print("Count ones in a:", count_ones(a))
    print("Get bit 1 from a:", get_bit(a, 1))
    print("Set bit 2 in a to 1:", set_bit(a, 2, 1))
    print("Toggle bit 0 in a:", toggle_bit(a, 0))
    
    # Test wave operations
    print("Wave interference (XOR):", binary_wave_interference([a, b], mode='xor'))
    print("Wave propagation (shift 1):", binary_wave_propagate(a, 1))
    print("Duty cycle (0.5, length 8):", create_duty_cycle(8, 0.5))
    print("Blocky sin (length 8, half_period 2):", generate_blocky_sin(8, 2))