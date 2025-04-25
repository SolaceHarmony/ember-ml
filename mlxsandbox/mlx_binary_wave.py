"""
MLX implementation of binary wave operations.

This module provides MLX-based implementations of binary wave operations
using MLXMegaBinary for all operations, ensuring proper integration with
the MegaNumber infrastructure.
"""

import mlx.core as mx
from typing import List, Tuple, Optional, Union, Dict, Any

from mlx_mega_binary import MLXMegaBinary, InterferenceMode

class MLXBinaryWave:
    """
    Binary wave operations using MLXMegaBinary.
    
    This class provides methods for binary wave processing using MLXMegaBinary,
    ensuring proper integration with the MegaNumber infrastructure.
    """
    
    @staticmethod
    def _ensure_mega_binary(x: Union[mx.array, MLXMegaBinary]) -> MLXMegaBinary:
        """
        Ensure the input is an MLXMegaBinary object.
        
        Args:
            x: Input value (mx.array or MLXMegaBinary)
            
        Returns:
            MLXMegaBinary object
        """
        if isinstance(x, MLXMegaBinary):
            return x
        
        # Convert mx.array to MLXMegaBinary
        if isinstance(x, mx.array):
            # Convert to integer
            if x.size == 1:
                val = int(x.item())
                
                # Convert to binary string
                bin_str = bin(val)[2:]
                
                # Create MLXMegaBinary
                return MLXMegaBinary(bin_str)
            else:
                # For arrays, we'll handle this at the operation level
                return x
        
        # Convert integer to MLXMegaBinary
        if isinstance(x, int):
            bin_str = bin(x)[2:]
            return MLXMegaBinary(bin_str)
        
        # If it's already a string, assume it's a binary string
        if isinstance(x, str):
            return MLXMegaBinary(x)
        
        raise TypeError(f"Cannot convert {type(x)} to MLXMegaBinary")
    
    @staticmethod
    def _convert_to_array(x: MLXMegaBinary) -> mx.array:
        """
        Convert MLXMegaBinary to mx.array.
        
        Args:
            x: MLXMegaBinary object
            
        Returns:
            mx.array representation
        """
        # Convert to integer
        val = int(x.to_decimal_string())
        
        # Convert to mx.array
        return mx.array(val, dtype=mx.uint16)
    
    @staticmethod
    def bitwise_and(x: Union[mx.array, MLXMegaBinary], 
                   y: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Compute the bitwise AND of x and y element-wise.
        
        Args:
            x: First input tensor or MLXMegaBinary
            y: Second input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the bitwise AND of x and y
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            if isinstance(y, mx.array) and y.size > 1:
                # Both are arrays
                return mx.bitwise_and(x, y)
            else:
                # x is array, y is scalar
                y_mb = MLXBinaryWave._ensure_mega_binary(y)
                y_arr = MLXBinaryWave._convert_to_array(y_mb)
                return mx.bitwise_and(x, y_arr)
        elif isinstance(y, mx.array) and y.size > 1:
            # y is array, x is scalar
            x_mb = MLXBinaryWave._ensure_mega_binary(x)
            x_arr = MLXBinaryWave._convert_to_array(x_mb)
            return mx.bitwise_and(x_arr, y)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        y_mb = MLXBinaryWave._ensure_mega_binary(y)
        
        # Perform operation
        result_mb = x_mb.bitwise_and(y_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def bitwise_or(x: Union[mx.array, MLXMegaBinary], 
                  y: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Compute the bitwise OR of x and y element-wise.
        
        Args:
            x: First input tensor or MLXMegaBinary
            y: Second input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the bitwise OR of x and y
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            if isinstance(y, mx.array) and y.size > 1:
                # Both are arrays
                return mx.bitwise_or(x, y)
            else:
                # x is array, y is scalar
                y_mb = MLXBinaryWave._ensure_mega_binary(y)
                y_arr = MLXBinaryWave._convert_to_array(y_mb)
                return mx.bitwise_or(x, y_arr)
        elif isinstance(y, mx.array) and y.size > 1:
            # y is array, x is scalar
            x_mb = MLXBinaryWave._ensure_mega_binary(x)
            x_arr = MLXBinaryWave._convert_to_array(x_mb)
            return mx.bitwise_or(x_arr, y)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        y_mb = MLXBinaryWave._ensure_mega_binary(y)
        
        # Perform operation
        result_mb = x_mb.bitwise_or(y_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def bitwise_xor(x: Union[mx.array, MLXMegaBinary], 
                   y: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Compute the bitwise XOR of x and y element-wise.
        
        Args:
            x: First input tensor or MLXMegaBinary
            y: Second input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the bitwise XOR of x and y
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            if isinstance(y, mx.array) and y.size > 1:
                # Both are arrays
                return mx.bitwise_xor(x, y)
            else:
                # x is array, y is scalar
                y_mb = MLXBinaryWave._ensure_mega_binary(y)
                y_arr = MLXBinaryWave._convert_to_array(y_mb)
                return mx.bitwise_xor(x, y_arr)
        elif isinstance(y, mx.array) and y.size > 1:
            # y is array, x is scalar
            x_mb = MLXBinaryWave._ensure_mega_binary(x)
            x_arr = MLXBinaryWave._convert_to_array(x_mb)
            return mx.bitwise_xor(x_arr, y)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        y_mb = MLXBinaryWave._ensure_mega_binary(y)
        
        # Perform operation
        result_mb = x_mb.bitwise_xor(y_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def bitwise_not(x: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Compute the bitwise NOT of x element-wise.
        
        Args:
            x: Input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the bitwise NOT of x
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            return mx.bitwise_invert(x)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        
        # Perform operation
        result_mb = x_mb.bitwise_not()
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def left_shift(x: Union[mx.array, MLXMegaBinary], 
                  shifts: Union[mx.array, MLXMegaBinary, int]) -> mx.array:
        """
        Shift the bits of x to the left by shifts positions.
        
        Args:
            x: Input tensor or MLXMegaBinary
            shifts: Number of bits to shift
            
        Returns:
            Tensor with x shifted left by shifts bits
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            if isinstance(shifts, int):
                return mx.left_shift(x, shifts)
            elif isinstance(shifts, mx.array):
                return mx.left_shift(x, shifts)
            else:
                shifts_mb = MLXBinaryWave._ensure_mega_binary(shifts)
                shifts_val = int(shifts_mb.to_decimal_string())
                return mx.left_shift(x, shifts_val)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        shifts_mb = MLXBinaryWave._ensure_mega_binary(shifts)
        
        # Perform operation
        result_mb = x_mb.shift_left(shifts_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def right_shift(x: Union[mx.array, MLXMegaBinary], 
                   shifts: Union[mx.array, MLXMegaBinary, int]) -> mx.array:
        """
        Shift the bits of x to the right by shifts positions.
        
        Args:
            x: Input tensor or MLXMegaBinary
            shifts: Number of bits to shift
            
        Returns:
            Tensor with x shifted right by shifts bits
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            if isinstance(shifts, int):
                return mx.right_shift(x, shifts)
            elif isinstance(shifts, mx.array):
                return mx.right_shift(x, shifts)
            else:
                shifts_mb = MLXBinaryWave._ensure_mega_binary(shifts)
                shifts_val = int(shifts_mb.to_decimal_string())
                return mx.right_shift(x, shifts_val)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        shifts_mb = MLXBinaryWave._ensure_mega_binary(shifts)
        
        # Perform operation
        result_mb = x_mb.shift_right(shifts_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def count_ones(x: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Count the number of 1 bits in each element of x.
        
        Args:
            x: Input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the count of 1 bits in each element of x
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            # For each element in the array
            result = mx.zeros_like(x)
            for i in range(x.size):
                x_i = x.reshape(-1)[i]
                x_mb = MLXBinaryWave._ensure_mega_binary(x_i)
                bits = x_mb.to_bits()
                count = sum(bits)
                result = result.at[i].add(count)
            return result.reshape(x.shape)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        
        # Count bits
        bits = x_mb.to_bits()
        count = sum(bits)
        
        # Convert to mx.array
        return mx.array(count, dtype=mx.uint16)
    
    @staticmethod
    def count_zeros(x: Union[mx.array, MLXMegaBinary]) -> mx.array:
        """
        Count the number of 0 bits in each element of x.
        
        Args:
            x: Input tensor or MLXMegaBinary
            
        Returns:
            Tensor with the count of 0 bits in each element of x
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            # For each element in the array
            result = mx.zeros_like(x)
            for i in range(x.size):
                x_i = x.reshape(-1)[i]
                x_mb = MLXBinaryWave._ensure_mega_binary(x_i)
                bits = x_mb.to_bits()
                count = len(bits) - sum(bits)
                result = result.at[i].add(count)
            return result.reshape(x.shape)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        
        # Count bits
        bits = x_mb.to_bits()
        count = len(bits) - sum(bits)
        
        # Convert to mx.array
        return mx.array(count, dtype=mx.uint16)
    
    @staticmethod
    def get_bit(x: Union[mx.array, MLXMegaBinary], 
               position: Union[mx.array, MLXMegaBinary, int]) -> mx.array:
        """
        Get the bit at the specified position in x.
        
        Args:
            x: Input tensor or MLXMegaBinary
            position: Bit position (0-based, from least significant bit)
            
        Returns:
            Tensor with the bit at the specified position in x
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            # For each element in the array
            result = mx.zeros_like(x)
            for i in range(x.size):
                x_i = x.reshape(-1)[i]
                x_mb = MLXBinaryWave._ensure_mega_binary(x_i)
                pos_mb = MLXBinaryWave._ensure_mega_binary(position)
                bit = 1 if x_mb.get_bit(pos_mb) else 0
                result = result.at[i].add(bit)
            return result.reshape(x.shape)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        pos_mb = MLXBinaryWave._ensure_mega_binary(position)
        
        # Get bit
        bit = 1 if x_mb.get_bit(pos_mb) else 0
        
        # Convert to mx.array
        return mx.array(bit, dtype=mx.uint16)
    
    @staticmethod
    def set_bit(x: Union[mx.array, MLXMegaBinary], 
               position: Union[mx.array, MLXMegaBinary, int], 
               value: Union[mx.array, MLXMegaBinary, int, bool]) -> mx.array:
        """
        Set the bit at the specified position in x to value.
        
        Args:
            x: Input tensor or MLXMegaBinary
            position: Bit position (0-based, from least significant bit)
            value: Bit value (0 or 1)
            
        Returns:
            Tensor with the bit at the specified position in x set to value
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            # For each element in the array
            result = mx.zeros_like(x)
            for i in range(x.size):
                x_i = x.reshape(-1)[i]
                x_mb = MLXBinaryWave._ensure_mega_binary(x_i)
                pos_mb = MLXBinaryWave._ensure_mega_binary(position)
                
                # Convert value to boolean
                if isinstance(value, mx.array):
                    val_bool = bool(value.item())
                elif isinstance(value, MLXMegaBinary):
                    val_bool = not value.is_zero()
                else:
                    val_bool = bool(value)
                
                # Set bit
                x_mb.set_bit(pos_mb, val_bool)
                
                # Convert back to mx.array
                result = result.at[i].add(MLXBinaryWave._convert_to_array(x_mb))
            return result.reshape(x.shape)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        pos_mb = MLXBinaryWave._ensure_mega_binary(position)
        
        # Convert value to boolean
        if isinstance(value, mx.array):
            val_bool = bool(value.item())
        elif isinstance(value, MLXMegaBinary):
            val_bool = not value.is_zero()
        else:
            val_bool = bool(value)
        
        # Set bit
        x_mb_copy = MLXMegaBinary(x_mb.to_string())
        x_mb_copy.set_bit(pos_mb, val_bool)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(x_mb_copy)
    
    @staticmethod
    def toggle_bit(x: Union[mx.array, MLXMegaBinary], 
                  position: Union[mx.array, MLXMegaBinary, int]) -> mx.array:
        """
        Toggle the bit at the specified position in x.
        
        Args:
            x: Input tensor or MLXMegaBinary
            position: Bit position (0-based, from least significant bit)
            
        Returns:
            Tensor with the bit at the specified position in x toggled
        """
        # Handle array inputs
        if isinstance(x, mx.array) and x.size > 1:
            # For each element in the array
            result = mx.zeros_like(x)
            for i in range(x.size):
                x_i = x.reshape(-1)[i]
                x_mb = MLXBinaryWave._ensure_mega_binary(x_i)
                pos_mb = MLXBinaryWave._ensure_mega_binary(position)
                
                # Get current bit
                current_bit = x_mb.get_bit(pos_mb)
                
                # Toggle bit
                x_mb_copy = MLXMegaBinary(x_mb.to_string())
                x_mb_copy.set_bit(pos_mb, not current_bit)
                
                # Convert back to mx.array
                result = result.at[i].add(MLXBinaryWave._convert_to_array(x_mb_copy))
            return result.reshape(x.shape)
        
        # Convert to MLXMegaBinary
        x_mb = MLXBinaryWave._ensure_mega_binary(x)
        pos_mb = MLXBinaryWave._ensure_mega_binary(position)
        
        # Get current bit
        current_bit = x_mb.get_bit(pos_mb)
        
        # Toggle bit
        x_mb_copy = MLXMegaBinary(x_mb.to_string())
        x_mb_copy.set_bit(pos_mb, not current_bit)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(x_mb_copy)
    
    @staticmethod
    def binary_wave_interference(waves: List[Union[mx.array, MLXMegaBinary]],
                                mode: str = 'xor') -> mx.array:
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
        
        # Check if all waves are mx.array with size > 1
        if all(isinstance(wave, mx.array) and wave.size > 1 for wave in waves):
            # Use mx operations directly
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
        
        # Convert to MLXMegaBinary
        mb_waves = []
        for wave in waves:
            if isinstance(wave, mx.array) and wave.size > 1:
                # For arrays, we need to handle each element
                for i in range(wave.size):
                    mb_waves.append(MLXBinaryWave._ensure_mega_binary(wave[i]))
            else:
                mb_waves.append(MLXBinaryWave._ensure_mega_binary(wave))
        
        # Map mode string to InterferenceMode
        if mode == 'xor':
            interference_mode = InterferenceMode.XOR
        elif mode == 'and':
            interference_mode = InterferenceMode.AND
        elif mode == 'or':
            interference_mode = InterferenceMode.OR
        else:
            raise ValueError(f"Unsupported interference mode: {mode}")
        
        # Perform interference
        result_mb = mb_waves[0]
        for wave in mb_waves[1:]:
            if interference_mode == InterferenceMode.XOR:
                result_mb = result_mb.bitwise_xor(wave)
            elif interference_mode == InterferenceMode.AND:
                result_mb = result_mb.bitwise_and(wave)
            elif interference_mode == InterferenceMode.OR:
                result_mb = result_mb.bitwise_or(wave)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def binary_wave_propagate(wave: Union[mx.array, MLXMegaBinary], 
                             shift: Union[mx.array, MLXMegaBinary, int]) -> mx.array:
        """
        Propagate a binary wave by shifting it.
        
        Args:
            wave: Binary wave pattern
            shift: Number of positions to shift
            
        Returns:
            Propagated wave pattern
        """
        # Handle array inputs
        if isinstance(wave, mx.array) and wave.size > 1:
            # For each element in the array
            result = mx.zeros_like(wave)
            for i in range(wave.size):
                wave_i = wave.reshape(-1)[i]
                wave_mb = MLXBinaryWave._ensure_mega_binary(wave_i)
                shift_mb = MLXBinaryWave._ensure_mega_binary(shift)
                
                # Propagate wave
                result_mb = wave_mb.propagate(shift_mb)
                
                # Convert back to mx.array
                result = result.at[i].add(MLXBinaryWave._convert_to_array(result_mb))
            return result.reshape(wave.shape)
        
        # Convert to MLXMegaBinary
        wave_mb = MLXBinaryWave._ensure_mega_binary(wave)
        shift_mb = MLXBinaryWave._ensure_mega_binary(shift)
        
        # Propagate wave
        result_mb = wave_mb.propagate(shift_mb)
        
        # Convert back to mx.array
        return MLXBinaryWave._convert_to_array(result_mb)
    
    @staticmethod
    def create_duty_cycle(length: int, duty_cycle: float) -> mx.array:
        """
        Create a binary pattern with the specified duty cycle.
        
        Args:
            length: Length of the pattern
            duty_cycle: Fraction of bits that should be 1
            
        Returns:
            Binary pattern with the specified duty cycle
        """
        # Calculate number of 1 bits
        num_ones = int(length * duty_cycle)
        
        # Create pattern
        pattern = mx.zeros(length, dtype=mx.uint16)
        
        # Set first num_ones bits to 1
        if num_ones > 0:
            pattern = mx.array([1] * num_ones + [0] * (length - num_ones), dtype=mx.uint16)
        
        return pattern
    
    @staticmethod
    def generate_blocky_sin(length: int, half_period: int) -> mx.array:
        """
        Generate a blocky sine wave pattern.
        
        Args:
            length: Length of the pattern
            half_period: Half the period of the wave
            
        Returns:
            Blocky sine wave pattern
        """
        # Create indices
        indices = mx.arange(length)
        
        # Calculate cycle position
        cycle_position = mx.remainder(indices, mx.array(2 * half_period))
        
        # Create pattern: 1 for first half of period, 0 for second half
        pattern = mx.where(mx.less(cycle_position, mx.array(half_period)),
                          mx.array(1, dtype=mx.uint16),
                          mx.array(0, dtype=mx.uint16))
        
        return pattern


# Test the class
if __name__ == "__main__":
    # Create test arrays
    a = mx.array([0b1010, 0b1100, 0b1111], dtype=mx.uint16)
    b = mx.array([0b0101, 0b1010, 0b0000], dtype=mx.uint16)
    
    print("a:", a)
    print("b:", b)
    
    # Test bitwise operations
    print("a & b:", MLXBinaryWave.bitwise_and(a, b))
    print("a | b:", MLXBinaryWave.bitwise_or(a, b))
    print("a ^ b:", MLXBinaryWave.bitwise_xor(a, b))
    print("~a:", MLXBinaryWave.bitwise_not(a))
    
    # Test shift operations
    print("a << 1:", MLXBinaryWave.left_shift(a, 1))
    print("a >> 1:", MLXBinaryWave.right_shift(a, 1))
    
    # Test bit operations
    print("Count ones in a:", MLXBinaryWave.count_ones(a))
    print("Get bit 1 from a:", MLXBinaryWave.get_bit(a, 1))
    print("Set bit 2 in a to 1:", MLXBinaryWave.set_bit(a, 2, 1))
    print("Toggle bit 0 in a:", MLXBinaryWave.toggle_bit(a, 0))
    
    # Test wave operations
    print("Wave interference (XOR):", MLXBinaryWave.binary_wave_interference([a[0], b[0]], mode='xor'))
    print("Wave propagation (shift 1):", MLXBinaryWave.binary_wave_propagate(a[0], 1))
    print("Duty cycle (0.5, length 8):", MLXBinaryWave.create_duty_cycle(8, 0.5))
    print("Blocky sin (length 8, half_period 2):", MLXBinaryWave.generate_blocky_sin(8, 2))