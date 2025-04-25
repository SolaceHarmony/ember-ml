"""
MLX implementation of MegaBinary from BizarroMath.

This module provides a direct transliteration of the MegaBinary class
from BizarroMath to MLX, using mlx.array with dtype=uint16 as the
underlying representation.
"""

import mlx.core as mx
from typing import Tuple, Union, List, Optional, Any
from enum import Enum

from mlx_mega_number import MLXMegaNumber

class InterferenceMode(Enum):
    """Interference modes for binary wave operations."""
    XOR = "xor"
    AND = "and"
    OR  = "or"

class MLXMegaBinary(MLXMegaNumber):
    """
    MLX-based binary data class, storing bits in MLX arrays with uint16 dtype.
    Includes wave generation, duty-cycle patterns, interference, and
    optional leading-zero preservation.
    """
    
    def __init__(self, value: Union[str, bytes, bytearray] = "0",
                 keep_leading_zeros: bool = True,
                 **kwargs):
        """
        Initialize a MLXMegaBinary object.
        
        Args:
            value: Initial value, can be:
                - String of binary digits (e.g., "1010" or "0b1010")
                - bytes/bytearray (will parse each byte => 8 bits => MLX array)
                - Default "0" => MLX array of just [0]
            keep_leading_zeros: Whether to keep leading zeros (default: True)
            **kwargs: Additional arguments for MLXMegaNumber
        """
        super().__init__(
            mantissa=None,
            exponent=None,
            negative=False,
            is_float=False,
            exponent_negative=False,
            keep_leading_zeros=keep_leading_zeros,
            **kwargs
        )
        
        # Step 1) Auto-detect and convert input
        if isinstance(value, (bytes, bytearray)):
            # Store original bytes if needed
            self.byte_data = bytearray(value)
            # Convert them to a binary string => MLX array
            bin_str = "".join(format(b, "08b") for b in self.byte_data)
        else:
            # Assume it's a string of bits (e.g., "1010" or "0b1010")
            # or possibly an empty string => "0"
            bin_str = value
            if bin_str.startswith("0b"):
                bin_str = bin_str[2:]
            if not bin_str:
                bin_str = "0"
                
            # Also build self.byte_data from this binary string
            # so we have a consistent stored representation if needed.
            # We'll chunk every 8 bits => int => byte
            self.byte_data = bytearray()
            for i in range(0, len(bin_str), 8):
                chunk = bin_str[i:i+8]
                self.byte_data.append(int(chunk.zfill(8), 2))  # ensure it's 8 bits
        
        # Step 2) Parse bin_str into MLX array
        self._parse_binary_string(bin_str)
        
        # Step 3) Normalize with respect to keep_leading_zeros
        self._normalize()
        
        # Store bit length
        self._bit_length = len(bin_str)
    
    def _parse_binary_string(self, bin_str: str) -> None:
        """
        Convert binary string => MLX array in little-endian chunk form.
        
        Args:
            bin_str: Binary string (e.g., "1010" or "0b1010")
        """
        if bin_str.startswith("0b"):
            bin_str = bin_str[2:]
        if not bin_str:
            bin_str = "0"
        
        # Store bit length
        self._bit_length = len(bin_str)
        
        # Convert to integer
        val = int(bin_str, 2)
        
        # Get mask as integer
        mask = int(self._mask)
        
        # Build MLX array in little-endian
        csize = int(self._global_chunk_size)
        limbs = []
        while val > 0:
            limbs.append(val & mask)
            val >>= csize
        if not limbs:
            limbs = [0]
        # Ensure all values in limbs are within uint16 range
        limbs = [int(x) & 0xFFFF for x in limbs]
        self.mantissa = mx.array(limbs, dtype=mx.uint16)
    
    def bitwise_and(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Perform bitwise AND operation.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Result of bitwise AND operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))
        
        # Pad arrays to the same length
        self_arr = mx.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = mx.pad(other.mantissa, [(0, max_len - len(other.mantissa))])
        
        # Perform bitwise AND
        result_arr = mx.bitwise_and(self_arr, other_arr)
        
        # Create result
        result = MLXMegaBinary("0")
        result.mantissa = result_arr
        result._normalize()
        
        return result
    
    def bitwise_or(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Perform bitwise OR operation.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Result of bitwise OR operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))
        
        # Pad arrays to the same length
        self_arr = mx.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = mx.pad(other.mantissa, [(0, max_len - len(other.mantissa))])
        
        # Perform bitwise OR
        result_arr = mx.bitwise_or(self_arr, other_arr)
        
        # Create result
        result = MLXMegaBinary("0")
        result.mantissa = result_arr
        result._normalize()
        
        return result
    
    def bitwise_xor(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Perform bitwise XOR operation.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Result of bitwise XOR operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))
        
        # Pad arrays to the same length
        self_arr = mx.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = mx.pad(other.mantissa, [(0, max_len - len(other.mantissa))])
        
        # Perform bitwise XOR
        result_arr = mx.bitwise_xor(self_arr, other_arr)
        
        # Create result
        result = MLXMegaBinary("0")
        result.mantissa = result_arr
        result._normalize()
        
        return result
    
    def bitwise_not(self) -> "MLXMegaBinary":
        """
        Perform bitwise NOT operation.
        
        Returns:
            Result of bitwise NOT operation
        """
        # Perform bitwise NOT
        result_arr = mx.bitwise_invert(self.mantissa)
        
        # Create result
        result = MLXMegaBinary("0")
        result.mantissa = result_arr
        result._normalize()
        
        return result
    
    def add(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Add two MLXMegaBinary objects.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Sum as MLXMegaBinary
        """
        result = super().add(other)
        
        # Build new MLXMegaBinary from result.mantissa
        result_int = self._chunklist_to_int(result.mantissa)
        
        # Convert to binary string
        out_str = bin(int(result_int))[2:]
        
        # Create new MLXMegaBinary
        out_bin = self.__class__(out_str)
        
        # Copy bit_length
        out_bin._bit_length = len(out_str)
        
        return out_bin
    
    def sub(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Subtract other from self.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Difference as MLXMegaBinary
        """
        result = super().sub(other)
        
        # Build new MLXMegaBinary from result.mantissa
        result_int = self._chunklist_to_int(result.mantissa)
        
        # Convert to binary string
        out_str = bin(int(result_int))[2:]
        
        # Create new MLXMegaBinary
        out_bin = self.__class__(out_str)
        
        # Copy bit_length
        out_bin._bit_length = len(out_str)
        
        return out_bin
    
    def mul(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Multiply two MLXMegaBinary objects.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Product as MLXMegaBinary
        """
        result = super().mul(other)
        
        # Build new MLXMegaBinary from result.mantissa
        result_int = self._chunklist_to_int(result.mantissa)
        
        # Convert to binary string
        out_str = bin(int(result_int))[2:]
        
        # Create new MLXMegaBinary
        out_bin = self.__class__(out_str)
        
        # Copy bit_length
        out_bin._bit_length = len(out_str)
        
        return out_bin
    
    def div(self, other: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Divide self by other.
        
        Args:
            other: Another MLXMegaBinary object
            
        Returns:
            Quotient as MLXMegaBinary
        """
        # Check for division by zero
        if len(other.mantissa) == 1 and other.mantissa[0] == 0:
            raise ZeroDivisionError("Divide by zero")
        
        result = super().div(other)
        
        # Build new MLXMegaBinary from result.mantissa
        result_int = self._chunklist_to_int(result.mantissa)
        
        # Convert to binary string
        out_str = bin(int(result_int))[2:]
        
        # Create new MLXMegaBinary
        out_bin = self.__class__(out_str)
        
        # Copy bit_length
        out_bin._bit_length = len(out_str)
        
        return out_bin
    
    def shift_left(self, bits: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Shift left by bits.
        
        Args:
            bits: Number of bits to shift
            
        Returns:
            Shifted MLXMegaBinary
        """
        # Convert bits to integer
        shift_count = int(bits.to_decimal_string())
        
        # Convert to binary string
        bin_str = self.to_string()
        
        # Shift left by adding zeros to the right
        shifted_str = bin_str + "0" * shift_count
        
        # Create result
        result = MLXMegaBinary(shifted_str)
        
        return result
    
    def shift_right(self, bits: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Shift right by bits.
        
        Args:
            bits: Number of bits to shift
            
        Returns:
            Shifted MLXMegaBinary
        """
        # Convert bits to integer
        shift_count = int(bits.to_decimal_string())
        
        # Convert to binary string
        bin_str = self.to_string()
        
        # Shift right by removing bits from the right
        if shift_count >= len(bin_str):
            shifted_str = "0"
        else:
            shifted_str = bin_str[:-shift_count] if shift_count > 0 else bin_str
        
        # Create result
        result = MLXMegaBinary(shifted_str)
        
        return result
    
    def get_bit(self, position: "MLXMegaBinary") -> bool:
        """
        Get the bit at the specified position.
        
        Args:
            position: Bit position (0-based, from least significant bit)
            
        Returns:
            Bit value (True or False)
        """
        # Convert position to integer
        pos = int(position.to_decimal_string())
        
        # Convert to binary string
        bin_str = self.to_string()
        
        # Reverse the string to get LSB first
        bin_str = bin_str[::-1]
        
        # Get the bit
        if pos >= len(bin_str):
            return False
        
        return bin_str[pos] == "1"
    
    def set_bit(self, position: "MLXMegaBinary", value: bool) -> None:
        """
        Set the bit at the specified position.
        
        Args:
            position: Bit position (0-based, from least significant bit)
            value: Bit value (True or False)
        """
        # Convert position to integer
        pos = int(position.to_decimal_string())
        
        # Convert to binary string
        bin_str = self.to_string()
        
        # Reverse the string to get LSB first
        bin_str = bin_str[::-1]
        
        # Extend the string if needed
        if pos >= len(bin_str):
            bin_str = bin_str + "0" * (pos - len(bin_str) + 1)
        
        # Set the bit
        bin_list = list(bin_str)
        bin_list[pos] = "1" if value else "0"
        bin_str = "".join(bin_list)
        
        # Reverse back
        bin_str = bin_str[::-1]
        
        # Remove leading zeros
        bin_str = bin_str.lstrip("0")
        if not bin_str:
            bin_str = "0"
        
        # Parse the new binary string
        self._parse_binary_string(bin_str)
        self._normalize()
    
    @classmethod
    def interfere(cls, waves: List["MLXMegaBinary"], mode: InterferenceMode) -> "MLXMegaBinary":
        """
        Combine multiple waves bitwise (XOR, AND, OR).
        
        Args:
            waves: List of MLXMegaBinary objects
            mode: Interference mode (XOR, AND, OR)
            
        Returns:
            Interference pattern
        """
        if not waves:
            raise ValueError("Need at least one wave for interference")
        
        result = waves[0]
        
        for wave in waves[1:]:
            if mode == InterferenceMode.XOR:
                result = result.bitwise_xor(wave)
            elif mode == InterferenceMode.AND:
                result = result.bitwise_and(wave)
            elif mode == InterferenceMode.OR:
                result = result.bitwise_or(wave)
        
        return result
    
    @classmethod
    def generate_blocky_sin(cls, length: "MLXMegaBinary", half_period: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Create a blocky sine wave pattern.
        
        Args:
            length: Length of the pattern in bits
            half_period: Half the period of the wave in bits
            
        Returns:
            Blocky sine wave pattern
        """
        # Create wave
        wave = cls("0")
        wave._keep_leading_zeros = True
        
        # Calculate two_half_period = half_period * 2
        two = cls("10")  # binary "10" => decimal 2
        two_half_period = half_period.mul(two)
        
        # Initialize accumulators
        i = cls("0")
        acc = cls("0")
        
        while True:
            # Compare i vs length: if i >= length => break
            cmp_i_len = i._compare_abs(i.mantissa, length.mantissa)
            if cmp_i_len >= 0:
                break
            
            # Compare acc vs half_period => if acc < half_period => bit=1, else 0
            cmp_acc_half = acc._compare_abs(acc.mantissa, half_period.mantissa)
            wave_bit = (cmp_acc_half < 0)  # True=>1, False=>0
            
            # Set wave bit at index i
            wave.set_bit(i, wave_bit)
            
            # acc++ => increment
            acc = acc.add(cls("1"))
            
            # if acc >= 2*half_period => acc -= 2*half_period
            cmp_acc_twoperiod = acc._compare_abs(acc.mantissa, two_half_period.mantissa)
            if cmp_acc_twoperiod >= 0:
                acc = acc.sub(two_half_period)
            
            # i++ => increment
            i = i.add(cls("1"))
        
        wave._normalize()
        return wave
    
    @classmethod
    def create_duty_cycle(cls, length: "MLXMegaBinary", duty_cycle: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Create a binary pattern with the specified duty cycle.
        
        Args:
            length: Length of the pattern in bits
            duty_cycle: Fraction of bits that should be 1
            
        Returns:
            Binary pattern with the specified duty cycle
        """
        # Calculate high_samples = length * duty_cycle
        high_samples = length.mul(duty_cycle)
        
        # Create pattern = (1 << high_samples) - 1
        one = cls("1")
        pattern = one.shift_left(high_samples).sub(one)
        
        # Calculate remaining = length - high_samples
        remaining = length.sub(high_samples)
        
        # If remaining is not zero, shift pattern left by remaining
        if not remaining.is_zero():
            pattern = pattern.shift_left(remaining)
        
        return pattern
    
    def propagate(self, shift: "MLXMegaBinary") -> "MLXMegaBinary":
        """
        Propagate the wave by shifting it.
        
        Args:
            shift: Number of bits to shift
            
        Returns:
            Propagated wave
        """
        return self.shift_left(shift)
    
    def to_bits(self) -> List[int]:
        """
        Convert to list of bits (LSB first).
        
        Returns:
            List of bits (0 or 1)
        """
        # Convert to binary string
        bin_str = self.to_string()
        
        # Reverse the string to get LSB first
        bin_str = bin_str[::-1]
        
        # Convert to list of integers
        return [int(bit) for bit in bin_str]
    
    def to_bits_bigendian(self) -> List[int]:
        """
        Convert to list of bits (MSB first).
        
        Returns:
            List of bits (0 or 1)
        """
        # Convert to binary string
        bin_str = self.to_string()
        
        # Convert to list of integers
        return [int(bit) for bit in bin_str]
    
    def to_string(self) -> str:
        """
        Convert to binary string.
        
        Returns:
            Binary string representation
        """
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"
        
        # Convert to integer
        val = self._chunklist_to_int(self.mantissa)
        
        # Convert to binary string
        return bin(int(val))[2:]
    
    def to_string_bigendian(self) -> str:
        """
        Convert to binary string (MSB first).
        
        Returns:
            Binary string representation (MSB first)
        """
        return self.to_string()
    
    def is_zero(self) -> bool:
        """
        Check if the value is zero.
        
        Returns:
            True if the value is zero, False otherwise
        """
        return len(self.mantissa) == 1 and self.mantissa[0] == 0
    
    def to_bytes(self) -> bytearray:
        """
        Convert to bytes.
        
        Returns:
            Byte representation
        """
        return self.byte_data
    
    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation
        """
        return f"<MLXMegaBinary {self.to_string()}>"


# Test the class
if __name__ == "__main__":
    # Create MLXMegaBinary objects
    a = MLXMegaBinary("1010")
    b = MLXMegaBinary("0101")
    
    print("a:", a)
    print("b:", b)
    
    # Test bitwise operations
    print("a & b:", a.bitwise_and(b))
    print("a | b:", a.bitwise_or(b))
    print("a ^ b:", a.bitwise_xor(b))
    print("~a:", a.bitwise_not())
    
    # Test shift operations
    print("a << 2:", a.shift_left(MLXMegaBinary("10")))
    print("a >> 1:", a.shift_right(MLXMegaBinary("1")))
    
    # Test bit operations
    print("a.get_bit(1):", a.get_bit(MLXMegaBinary("1")))
    a.set_bit(MLXMegaBinary("10"), True)
    print("a after setting bit 2:", a)
    
    # Test wave operations
    print("Blocky sin (length=8, half_period=2):", MLXMegaBinary.generate_blocky_sin(
        MLXMegaBinary("1000"), MLXMegaBinary("10")))
    print("Duty cycle (length=8, duty_cycle=0.5):", MLXMegaBinary.create_duty_cycle(
        MLXMegaBinary("1000"), MLXMegaBinary("100")))
    
    # Test interference
    print("Interference (XOR):", MLXMegaBinary.interfere([a, b], InterferenceMode.XOR))
    print("Interference (AND):", MLXMegaBinary.interfere([a, b], InterferenceMode.AND))
    print("Interference (OR):", MLXMegaBinary.interfere([a, b], InterferenceMode.OR))