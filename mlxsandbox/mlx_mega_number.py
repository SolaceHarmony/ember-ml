"""
MLX implementation of MegaNumber from BizarroMath.

This module provides a direct transliteration of the MegaNumber class
from BizarroMath to MLX, using mlx.array with dtype=int16 as the
underlying representation.
"""

import mlx.core as mx
from typing import Tuple, Union, List, Optional, Any

class MLXMegaNumber:
    """
    A chunk-based big integer (or float) with HPC-limb arithmetic,
    using MLX arrays with int16 dtype to mimic BigBase65536 logic.
    """

    # Constants as MLX arrays
    _global_chunk_size = mx.array(16, dtype=mx.int16)  # bits per limb
    _base = mx.array(65536, dtype=mx.int32)  # 2^16
    _mask = mx.array(65535, dtype=mx.int16)  # 2^16 - 1
    
    # Optional thresholds for advanced multiplication
    _MUL_THRESHOLD_KARATSUBA = mx.array(32, dtype=mx.int16)
    _MUL_THRESHOLD_TOOM = mx.array(128, dtype=mx.int16)
    
    _max_precision_bits = None
    _log2_of_10_cache = None  # class-level for caching log2(10)
    
    def __init__(
        self,
        value: Union[str, 'MLXMegaNumber', mx.array] = None,
        mantissa: mx.array = None,
        exponent: mx.array = None,
        negative: bool = False,
        is_float: bool = False,
        exponent_negative: bool = False,
        keep_leading_zeros: bool = False
    ):
        """
        Initialize a HPC-limb object using MLX arrays.
        
        Args:
            value: Initial value, can be:
                - String (decimal or binary)
                - MLXMegaNumber
                - MLX array of limbs
            mantissa: MLX array of limbs
            exponent: MLX array of limbs
            negative: Sign flag
            is_float: Float flag
            exponent_negative: Exponent sign flag
            keep_leading_zeros: Whether to keep leading zeros
        """
        if mantissa is None:
            mantissa = mx.array([0], dtype=mx.int16)
        if exponent is None:
            exponent = mx.array([0], dtype=mx.int16)
            
        self.mantissa = mantissa
        self.exponent = exponent
        self.negative = negative
        self.is_float = is_float
        self.exponent_negative = exponent_negative
        self._keep_leading_zeros = keep_leading_zeros
        
        if isinstance(value, str):
            # Parse decimal string
            tmp = MLXMegaNumber.from_decimal_string(value)
            self.mantissa = tmp.mantissa
            self.exponent = tmp.exponent
            self.negative = tmp.negative
            self.is_float = tmp.is_float
            self.exponent_negative = tmp.exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        elif isinstance(value, MLXMegaNumber):
            # Copy
            self.mantissa = mx.array(value.mantissa, dtype=mx.int16)
            self.exponent = mx.array(value.exponent, dtype=mx.int16)
            self.negative = value.negative
            self.is_float = value.is_float
            self.exponent_negative = value.exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        elif isinstance(value, mx.array):
            # Interpret as mantissa
            self.mantissa = value
            self.exponent = mx.array([0], dtype=mx.int16)
            self.negative = negative
            self.is_float = is_float
            self.exponent_negative = exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        else:
            # If nothing => user-supplied mantissa/exponent or default [0]
            pass
            
        # Normalize
        self._normalize()
    
    def _normalize(self):
        """
        If keep_leading_zeros=False => remove trailing zero-limbs from mantissa.
        If float => also remove trailing zeros from exponent. Keep at least 1 limb.
        If everything is zero => unify sign bits to false/positive.
        """
        if not self._keep_leading_zeros:
            # Trim mantissa
            while len(self.mantissa) > 1 and self.mantissa[-1] == 0:
                self.mantissa = self.mantissa[:-1]
                
            # Trim exponent if float
            if self.is_float:
                while len(self.exponent) > 1 and self.exponent[-1] == 0:
                    self.exponent = self.exponent[:-1]
            
            # If mantissa is entirely zero => unify sign
            if len(self.mantissa) == 1 and self.mantissa[0] == 0:
                self.negative = False
                self.exponent = mx.array([0], dtype=mx.int16)
                self.exponent_negative = False
        else:
            # If keep_leading_zeros => only unify if mantissa is all zero
            if mx.all(self.mantissa == 0):
                self.negative = False
                self.exponent_negative = False
    
    @classmethod
    def from_decimal_string(cls, dec_str: str) -> "MLXMegaNumber":
        """
        Convert decimal => HPC big-int or HPC float.
        We detect fractional by '.' => if present => treat as float, shifting exponent.
        
        Args:
            dec_str: Decimal string
            
        Returns:
            MLXMegaNumber
        """
        s = dec_str.strip()
        if not s:
            return cls(mantissa=mx.array([0], dtype=mx.int16),
                      exponent=mx.array([0], dtype=mx.int16),
                      negative=False, is_float=False)
        
        negative = False
        if s.startswith('-'):
            negative = True
            s = s[1:].strip()
        
        # Detect fractional
        point_pos = s.find('.')
        frac_len = 0
        if point_pos >= 0:
            frac_len = len(s) - (point_pos + 1)
            s = s.replace('.', '')
        
        # Repeatedly multiply by 10 and add digit
        mant = mx.array([0], dtype=mx.int16)
        for ch in s:
            if ch < '0' or ch > '9':
                raise ValueError(f"Invalid digit '{ch}' in decimal string.")
            
            # Convert digit to MLX array
            digit_val = mx.array(int(ch), dtype=mx.int16)
            
            # Multiply mant by 10
            ten = mx.array([10], dtype=mx.int16)
            mant = cls._mul_chunklists(
                mant,
                ten,
                cls._global_chunk_size,
                cls._base
            )
            
            # Add digit
            carry = digit_val
            idx = mx.array(0, dtype=mx.int16)
            
            # Create a new array to store the result
            new_mant = mx.array(mant)
            
            while mx.any(mx.not_equal(carry, mx.array(0))) or idx < len(mant):
                if idx == len(mant):
                    # Append carry to new_mant
                    new_mant = mx.concatenate([new_mant, mx.array([0], dtype=mx.int16)])
                
                # Get the value at idx
                val = new_mant[idx] if idx < len(new_mant) else mx.array(0, dtype=mx.int16)
                
                # Add carry
                ssum = mx.add(val, carry)
                
                # Update new_mant[idx]
                new_mant = mx.array([*new_mant[:idx], mx.bitwise_and(ssum, cls._mask), *new_mant[idx+1:]])
                
                # Update carry
                carry = mx.right_shift(ssum, cls._global_chunk_size)
                
                # Increment idx
                idx = mx.add(idx, mx.array(1, dtype=mx.int16))
            
            mant = new_mant
        
        exp_limb = mx.array([0], dtype=mx.int16)
        exponent_negative = False
        is_float = False
        
        # If we had fraction => shift exponent
        if frac_len > 0:
            is_float = True
            exponent_negative = True
            
            # Approximate: frac_len * log2(10) => bit shift exponent
            # Convert frac_len to MLX array
            frac_len_mx = mx.array(frac_len, dtype=mx.int16)
            
            # Multiply by log2(10) ≈ 3.32
            log2_10 = mx.array(3.32, dtype=mx.float32)
            bits_needed_float = mx.multiply(mx.array(frac_len, dtype=mx.float32), log2_10)
            bits_needed = mx.array(mx.ceil(bits_needed_float), dtype=mx.int16)
            
            exp_limb = cls._int_to_chunklist(bits_needed, cls._global_chunk_size)
        
        obj = cls(
            mantissa=mant,
            exponent=exp_limb,
            negative=negative,
            is_float=is_float,
            exponent_negative=exponent_negative
        )
        obj._normalize()
        return obj
    
    @classmethod
    def from_binary_string(cls, bin_str: str) -> "MLXMegaNumber":
        """
        Convert binary string => HPC big-int.
        
        Args:
            bin_str: Binary string (e.g., "1010" or "0b1010")
            
        Returns:
            MLXMegaNumber
        """
        s = bin_str.strip()
        if s.startswith('0b'):
            s = s[2:]
        if not s:
            s = "0"
        
        # Convert to integer
        val = int(s, 2)
        
        # Convert to MLX array
        val_mx = mx.array(val, dtype=mx.int32)
        
        # Convert to limbs
        limbs = cls._int_to_chunklist(val_mx, cls._global_chunk_size)
        
        return cls(
            mantissa=limbs,
            exponent=mx.array([0], dtype=mx.int16),
            negative=False,
            is_float=False
        )
    
    def to_decimal_string(self, max_digits=None) -> str:
        """
        Convert to decimal string.
        
        Args:
            max_digits: Maximum number of digits to include
            
        Returns:
            Decimal string representation
        """
        # Handle zero
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"
        
        sign_str = "-" if self.negative else ""
        
        if not self.is_float:
            # Integer => repeated divmod 10
            tmp = mx.array(self.mantissa)
            digits_rev = []
            
            zero = mx.array([0], dtype=mx.int16)
            ten = mx.array(10, dtype=mx.int16)
            
            while not (len(tmp) == 1 and tmp[0] == 0):
                tmp, r = self._divmod_small(tmp, ten)
                digits_rev.append(str(int(r)))
            
            digits_rev.reverse()
            dec_str = "".join(digits_rev)
            
            if max_digits and len(dec_str) > max_digits:
                dec_str = f"...{dec_str[-max_digits:]}"
            
            return sign_str + dec_str
        else:
            # Float => exponent shift
            # If exponent_negative => we do mantissa // 2^(exponent), capturing remainder => fractional digits.
            # else => mantissa << exponent => integer.
            exp_int = self._chunklist_to_int(self.exponent)
            
            if self.exponent_negative:
                # Do integer part
                int_part, remainder = self._div_by_2exp(self.mantissa, exp_int)
                int_str = self._chunk_to_dec_str(int_part, max_digits)
                
                # If remainder=0 => done
                zero = mx.array([0], dtype=mx.int16)
                if self._compare_abs(remainder, zero) == 0:
                    return sign_str + int_str
                
                # Else => build fractional by repeatedly *10 // 2^exp_int
                frac_digits = []
                steps = max_digits or 50
                cur_rem = remainder
                
                ten = mx.array([10], dtype=mx.int16)
                
                for _ in range(steps):
                    cur_rem = self._mul_chunklists(
                        cur_rem,
                        ten,
                        self._global_chunk_size,
                        self._base
                    )
                    
                    # Compute 2^exp_int
                    two_exp = mx.array(1 << 16, dtype=mx.int32)  # 2^16
                    q, cur_rem = self._div_chunk(cur_rem, two_exp)
                    
                    digit_val = self._chunklist_to_int(q)
                    frac_digits.append(str(int(digit_val)))
                    
                    if self._compare_abs(cur_rem, zero) == 0:
                        break
                
                return sign_str + int_str + "." + "".join(frac_digits)
            else:
                # Exponent positive => mantissa << exp_int
                shifted = self._mul_by_2exp(self.mantissa, exp_int)
                return sign_str + self._chunk_to_dec_str(shifted, max_digits)
    
    def _chunk_to_dec_str(self, chunks, max_digits=None) -> str:
        """
        Convert chunks to decimal string.
        
        Args:
            chunks: MLX array of chunks
            max_digits: Maximum number of digits
            
        Returns:
            Decimal string
        """
        # Use to_decimal_string with a temporary MLXMegaNumber
        tmp = MLXMegaNumber(mantissa=chunks, is_float=False)
        return tmp.to_decimal_string(max_digits)
    
    def _div_by_2exp(self, limbs: mx.array, bits: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Integer division: limbs // 2^bits, remainder = limbs % 2^bits.
        
        Args:
            limbs: MLX array of limbs
            bits: Number of bits to divide by
            
        Returns:
            Tuple of (quotient, remainder)
        """
        zero = mx.array(0, dtype=mx.int16)
        
        if mx.all(mx.less_equal(bits, zero)):
            return (mx.array(limbs), mx.array([0], dtype=mx.int16))
        
        # Convert limbs to integer
        val_A = self._chunklist_to_int(limbs)
        
        # Calculate total bits in limbs
        total_bits = mx.multiply(mx.array(len(limbs), dtype=mx.int16), self._global_chunk_size)
        
        if mx.all(mx.greater_equal(bits, total_bits)):
            # Everything is remainder
            return (mx.array([0], dtype=mx.int16), limbs)
        
        # Calculate remainder mask
        one = mx.array(1, dtype=mx.int32)
        remainder_mask = mx.subtract(mx.left_shift(one, bits), one)
        
        # Calculate remainder
        remainder_val = mx.bitwise_and(val_A, remainder_mask)
        
        # Calculate quotient
        int_val = mx.right_shift(val_A, bits)
        
        # Convert back to chunks
        int_part = self._int_to_chunklist(int_val, self._global_chunk_size)
        rem_part = self._int_to_chunklist(remainder_val, self._global_chunk_size)
        
        return (int_part, rem_part)
    
    def _mul_by_2exp(self, limbs: mx.array, bits: mx.array) -> mx.array:
        """
        Multiply by 2^bits.
        
        Args:
            limbs: MLX array of limbs
            bits: Number of bits to multiply by
            
        Returns:
            MLX array of limbs
        """
        zero = mx.array(0, dtype=mx.int16)
        
        if mx.all(mx.less_equal(bits, zero)):
            return mx.array(limbs)
        
        # Convert limbs to integer
        val_A = self._chunklist_to_int(limbs)
        
        # Shift left
        val_shifted = mx.left_shift(val_A, bits)
        
        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)
    
    def add(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Add two MLXMegaNumbers.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            Sum as MLXMegaNumber
        """
        if self.is_float or other.is_float:
            return self._add_float(other)
        
        # Integer addition
        if self.negative == other.negative:
            # Same sign => add
            sum_limb = self._add_chunklists(self.mantissa, other.mantissa)
            sign = self.negative
            out = MLXMegaNumber(
                mantissa=sum_limb,
                exponent=mx.array([0], dtype=mx.int16),
                negative=sign
            )
            return out
        else:
            # Opposite sign => subtract smaller from bigger
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)
            
            zero = mx.array(0, dtype=mx.int16)
            
            if cmp_val == 0:
                # Zero
                return MLXMegaNumber()
            elif cmp_val > 0:
                diff = self._sub_chunklists(self.mantissa, other.mantissa)
                return MLXMegaNumber(
                    mantissa=diff,
                    exponent=mx.array([0], dtype=mx.int16),
                    negative=self.negative
                )
            else:
                diff = self._sub_chunklists(other.mantissa, self.mantissa)
                return MLXMegaNumber(
                    mantissa=diff,
                    exponent=mx.array([0], dtype=mx.int16),
                    negative=other.negative
                )
    
    def sub(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Subtract other from self.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            Difference as MLXMegaNumber
        """
        # a - b => a + (-b)
        negB = MLXMegaNumber(
            mantissa=mx.array(other.mantissa),
            exponent=mx.array(other.exponent),
            negative=not other.negative,
            is_float=other.is_float,
            exponent_negative=other.exponent_negative
        )
        return self.add(negB)
    
    def mul(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Multiply two MLXMegaNumbers.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            Product as MLXMegaNumber
        """
        # If float => combine exponents
        if not (self.is_float or other.is_float):
            # Integer multiply
            sign = (self.negative != other.negative)
            out_limb = self._mul_chunklists(
                self.mantissa,
                other.mantissa,
                self._global_chunk_size,
                self._base
            )
            out = MLXMegaNumber(
                mantissa=out_limb,
                exponent=mx.array([0], dtype=mx.int16),
                negative=sign
            )
            out._normalize()
            return out
        else:
            # Float multiply
            sign = (self.negative != other.negative)
            eA = self._exp_as_int(self)
            eB = self._exp_as_int(other)
            
            # Add exponents
            sum_exp = mx.add(eA, eB)
            
            # Multiply mantissas
            out_limb = self._mul_chunklists(
                self.mantissa,
                other.mantissa,
                self._global_chunk_size,
                self._base
            )
            
            # Determine exponent sign
            zero = mx.array(0, dtype=mx.int16)
            exp_neg = mx.all(mx.less(sum_exp, zero))
            
            # Get absolute value of exponent
            sum_exp_abs = mx.abs(sum_exp)
            
            # Convert to chunks
            new_exp = self._int_to_chunklist(sum_exp_abs, self._global_chunk_size) if mx.any(mx.not_equal(sum_exp_abs, zero)) else mx.array([0], dtype=mx.int16)
            
            out = MLXMegaNumber(
                mantissa=out_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )
            out._normalize()
            return out
    
    def div(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Divide self by other.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            Quotient as MLXMegaNumber
        """
        if not (self.is_float or other.is_float):
            # Integer division
            if len(other.mantissa) == 1 and other.mantissa[0] == 0:
                raise ZeroDivisionError("division by zero")
            
            sign = (self.negative != other.negative)
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)
            
            if cmp_val < 0:
                # Result = 0
                return MLXMegaNumber.from_decimal_string("0")
            elif cmp_val == 0:
                # Result = 1 or -1
                return MLXMegaNumber.from_decimal_string("1" if not sign else "-1")
            else:
                q, _ = self._div_chunk(self.mantissa, other.mantissa)
                out = MLXMegaNumber(
                    mantissa=q,
                    exponent=mx.array([0], dtype=mx.int16),
                    negative=sign
                )
                out._normalize()
                return out
        else:
            # Float division
            sign = (self.negative != other.negative)
            eA = self._exp_as_int(self)
            eB = self._exp_as_int(other)
            
            # Subtract exponents
            newExpVal = mx.subtract(eA, eB)
            
            # Check for division by zero
            if len(other.mantissa) == 1 and other.mantissa[0] == 0:
                raise ZeroDivisionError("division by zero")
            
            # Compare mantissas
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)
            
            if cmp_val < 0:
                q_limb = mx.array([0], dtype=mx.int16)
            elif cmp_val == 0:
                q_limb = mx.array([1], dtype=mx.int16)
            else:
                q_limb, _ = self._div_chunk(self.mantissa, other.mantissa)
            
            # Determine exponent sign
            zero = mx.array(0, dtype=mx.int16)
            exp_neg = mx.all(mx.less(newExpVal, zero))
            
            # Get absolute value of exponent
            newExpVal = mx.abs(newExpVal)
            
            # Convert to chunks
            new_exp = self._int_to_chunklist(newExpVal, self._global_chunk_size) if mx.any(mx.not_equal(newExpVal, zero)) else mx.array([0], dtype=mx.int16)
            
            out = MLXMegaNumber(
                mantissa=q_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )
            out._normalize()
            return out
    
    def _add_float(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Add two MLXMegaNumbers in float mode.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            Sum as MLXMegaNumber
        """
        eA = self._exp_as_int(self)
        eB = self._exp_as_int(other)
        
        zero = mx.array(0, dtype=mx.int16)
        
        if mx.all(mx.equal(eA, eB)):
            # Same exponent
            mantA, mantB = self.mantissa, other.mantissa
            final_exp = eA
        elif mx.all(mx.greater(eA, eB)):
            # Self has bigger exponent
            shift = mx.subtract(eA, eB)
            mantA = self.mantissa
            mantB = self._shift_right(other.mantissa, shift)
            final_exp = eA
        else:
            # Other has bigger exponent
            shift = mx.subtract(eB, eA)
            mantA = self._shift_right(self.mantissa, shift)
            mantB = other.mantissa
            final_exp = eB
        
        # Combine signs
        if self.negative == other.negative:
            sum_limb = self._add_chunklists(mantA, mantB)
            sign = self.negative
        else:
            c = self._compare_abs(mantA, mantB)
            if c == 0:
                return MLXMegaNumber(is_float=True)  # Zero
            elif c > 0:
                sum_limb = self._sub_chunklists(mantA, mantB)
                sign = self.negative
            else:
                sum_limb = self._sub_chunklists(mantB, mantA)
                sign = other.negative
        
        # Determine exponent sign
        exp_neg = mx.all(mx.less(final_exp, zero))
        
        # Get absolute value of exponent
        final_exp_abs = mx.abs(final_exp)
        
        # Convert to chunks
        exp_chunk = self._int_to_chunklist(final_exp_abs, self._global_chunk_size) if mx.any(mx.not_equal(final_exp_abs, zero)) else mx.array([0], dtype=mx.int16)
        
        out = MLXMegaNumber(
            mantissa=sum_limb,
            exponent=exp_chunk,
            negative=sign,
            is_float=True,
            exponent_negative=exp_neg
        )
        out._normalize()
        return out
    
    def _exp_as_int(self, mn: "MLXMegaNumber") -> mx.array:
        """
        Get exponent as integer.
        
        Args:
            mn: MLXMegaNumber
            
        Returns:
            Exponent as MLX array
        """
        val = self._chunklist_to_int(mn.exponent)
        return mx.negative(val) if mn.exponent_negative else val
    
    def _shift_right(self, limbs: mx.array, shift: mx.array) -> mx.array:
        """
        Shift limbs right by shift bits.
        
        Args:
            limbs: MLX array of limbs
            shift: Number of bits to shift
            
        Returns:
            Shifted limbs
        """
        # Convert to integer
        val = self._chunklist_to_int(limbs)
        
        # Shift right
        val_shifted = mx.right_shift(val, shift)
        
        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)
    
    def compare_abs(self, other: "MLXMegaNumber") -> int:
        """
        Compare absolute values.
        
        Args:
            other: Another MLXMegaNumber
            
        Returns:
            1 if self > other, -1 if self < other, 0 if equal
        """
        return self._compare_abs(self.mantissa, other.mantissa)
    
    @classmethod
    def _compare_abs(cls, A: mx.array, B: mx.array) -> int:
        """
        Compare absolute values of two MLX arrays.
        
        Args:
            A: First MLX array
            B: Second MLX array
            
        Returns:
            1 if A > B, -1 if A < B, 0 if equal
        """
        if len(A) > len(B):
            return 1
        elif len(A) < len(B):
            return -1
        else:
            for i in reversed(range(len(A))):
                if A[i] > B[i]:
                    return 1
                elif A[i] < B[i]:
                    return -1
            return 0
    
    @classmethod
    def _int_to_chunklist(cls, val: mx.array, csize: mx.array) -> mx.array:
        """
        Convert integer to chunk list.
        
        Args:
            val: Integer as MLX array
            csize: Chunk size
            
        Returns:
            MLX array of chunks
        """
        # Create mask
        one = mx.array(1, dtype=mx.int32)
        mask = mx.subtract(mx.left_shift(one, csize), one)
        
        # Initialize output
        out = []
        
        # Check if val is zero
        zero = mx.array(0, dtype=mx.int32)
        if mx.all(mx.equal(val, zero)):
            return mx.array([0], dtype=mx.int16)
        
        # Convert to chunks
        while mx.any(mx.greater(val, zero)):
            # Extract lowest chunk
            chunk = mx.bitwise_and(val, mask)
            out.append(chunk)
            
            # Shift right
            val = mx.right_shift(val, csize)
        
        return mx.array(out, dtype=mx.int16)
    
    @classmethod
    def _chunklist_to_int(cls, limbs: mx.array) -> mx.array:
        """
        Combine limbs => integer, little-endian.
        
        Args:
            limbs: MLX array of limbs
            
        Returns:
            Integer as MLX array
        """
        # Initialize result
        val = mx.array(0, dtype=mx.int32)
        
        # Initialize shift
        shift = mx.array(0, dtype=mx.int16)
        
        # Combine limbs
        for i in range(len(limbs)):
            # Shift limb and add to result
            limb_shifted = mx.left_shift(mx.array(limbs[i], dtype=mx.int32), shift)
            val = mx.add(val, limb_shifted)
            
            # Increment shift
            shift = mx.add(shift, cls._global_chunk_size)
        
        return val
    
    @classmethod
    def _mul_chunklists(cls, A: mx.array, B: mx.array, csize: mx.array, base: mx.array) -> mx.array:
        """
        Multiplication dispatcher: naive / Karatsuba / Toom.
        
        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs
            csize: Chunk size
            base: Base (2^csize)
            
        Returns:
            Product as MLX array of limbs
        """
        # Get lengths
        la, lb = len(A), len(B)
        n = max(la, lb)
        
        # Choose multiplication algorithm based on size
        threshold = cls._MUL_THRESHOLD_KARATSUBA
        if isinstance(threshold, mx.array):
            threshold_value = threshold.item() if threshold.size == 1 else threshold[0]
        else:
            threshold_value = threshold
        
        if n < threshold_value:
            return cls._mul_naive_chunklists(A, B, csize, base)
        elif n < cls._MUL_THRESHOLD_TOOM[0]:
            return cls._mul_karatsuba_chunklists(A, B, csize, base)
        else:
            return cls._mul_toom_chunklists(A, B, csize, base)
    
    @classmethod
    def _mul_naive_chunklists(cls, A: mx.array, B: mx.array, csize: mx.array, base: mx.array) -> mx.array:
        """
        Naive multiplication of chunk lists.
        
        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs
            csize: Chunk size
            base: Base (2^csize)
            
        Returns:
            Product as MLX array of limbs
        """
        # Get lengths
        la, lb = len(A), len(B)
        
        # Initialize output
        out = mx.zeros(la + lb, dtype=mx.int16)
        
        # Multiply limbs
        for i in range(la):
            carry = mx.array(0, dtype=mx.int32)
            for j in range(lb):
                # Multiply limbs
                mul_val = mx.add(
                    mx.add(
                        mx.multiply(mx.array(A[i], dtype=mx.int32), mx.array(B[j], dtype=mx.int32)),
                        mx.array(out[i + j], dtype=mx.int32)
                    ),
                    carry
                )
                
                # Update output
                out_ij = mx.bitwise_and(mul_val, mx.subtract(base, mx.array(1, dtype=mx.int32)))
                out_ij_array = mx.array([out_ij], dtype=mx.int16)
                out = mx.array([*out[:i+j], out_ij_array[0], *out[i+j+1:]])
                
                # Update carry
                carry = mx.right_shift(mul_val, csize)
            
            # Add carry to output
            if mx.any(mx.not_equal(carry, mx.array(0, dtype=mx.int32))):
                out_i_lb = mx.add(mx.array(out[i + lb], dtype=mx.int32), carry)
                out = mx.array([*out[:i+lb], mx.array(out_i_lb, dtype=mx.int16)[0], *out[i+lb+1:]])
        
        # Trim trailing zeros
        while len(out) > 1 and out[-1] == 0:
            out = out[:-1]
        
        return out
    
    @classmethod
    def _add_chunklists(cls, A: mx.array, B: mx.array) -> mx.array:
        """
        Add two chunk lists.
        
        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs
            
        Returns:
            Sum as MLX array of limbs
        """
        # Get lengths
        la, lb = len(A), len(B)
        max_len = max(la, lb)
        
        # Initialize output
        out = mx.array([], dtype=mx.int16)
        
        # Initialize carry
        carry = mx.array(0, dtype=mx.int16)
        
        # Add limbs
        for i in range(max_len):
            # Get limbs
            av = A[i] if i < la else mx.array(0, dtype=mx.int16)
            bv = B[i] if i < lb else mx.array(0, dtype=mx.int16)
            
            # Add limbs and carry
            s = mx.add(mx.add(av, bv), carry)
            
            # Update carry
            carry = mx.right_shift(s, cls._global_chunk_size)
            
            # Update output
            out = mx.concatenate([out, mx.array([mx.bitwise_and(s, cls._mask)], dtype=mx.int16)])
        
        # Add carry if needed
        if mx.any(mx.not_equal(carry, mx.array(0, dtype=mx.int16))):
            out = mx.concatenate([out, mx.array([carry], dtype=mx.int16)])
        
        # Trim trailing zeros
        while len(out) > 1 and out[-1] == 0:
            out = out[:-1]
        
        return out
    
    @classmethod
    def _sub_chunklists(cls, A: mx.array, B: mx.array) -> mx.array:
        """
        Subtract B from A (A >= B).
        
        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs
            
        Returns:
            Difference as MLX array of limbs
        """
        # Get lengths
        la, lb = len(A), len(B)
        max_len = max(la, lb)
        
        # Initialize output
        out = mx.array([], dtype=mx.int16)
        
        # Initialize carry
        carry = mx.array(0, dtype=mx.int16)
        
        # Subtract limbs
        for i in range(max_len):
            # Get limbs
            av = A[i] if i < la else mx.array(0, dtype=mx.int16)
            bv = B[i] if i < lb else mx.array(0, dtype=mx.int16)
            
            # Subtract limbs and carry
            diff = mx.subtract(mx.subtract(av, bv), carry)
            
            # Check if diff is negative
            if mx.any(mx.less(diff, mx.array(0, dtype=mx.int16))):
                diff = mx.add(diff, cls._base)
                carry = mx.array(1, dtype=mx.int16)
            else:
                carry = mx.array(0, dtype=mx.int16)
            
            # Update output
            out = mx.concatenate([out, mx.array([mx.bitwise_and(diff, cls._mask)], dtype=mx.int16)])
        
        # Trim trailing zeros
        while len(out) > 1 and out[-1] == 0:
            out = out[:-1]
        
        return out
    
    @classmethod
    def _div_chunk(cls, A: mx.array, B: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Divide A by B.
        
        Args:
            A: Dividend as MLX array of limbs
            B: Divisor as MLX array of limbs
            
        Returns:
            Tuple of (quotient, remainder) as MLX arrays of limbs
        """
        # Compare A and B
        c = cls._compare_abs(A, B)
        
        # If A < B, quotient is 0, remainder is A
        if c < 0:
            return (mx.array([0], dtype=mx.int16), A)
        
        # If A == B, quotient is 1, remainder is 0
        if c == 0:
            return (mx.array([1], dtype=mx.int16), mx.array([0], dtype=mx.int16))
        
        # Initialize quotient and remainder
        Q = mx.zeros(len(A), dtype=mx.int16)
        R = mx.array([0], dtype=mx.int16)
        
        # Get base
        base = mx.left_shift(mx.array(1, dtype=mx.int32), cls._global_chunk_size)
        
        # Long division
        for i in reversed(range(len(A))):
            # R <<= chunk_size
            R = cls._shiftleft_one_chunk(R)
            
            # R += A[i]
            R = cls._add_chunklists(R, mx.array([A[i]], dtype=mx.int16))
            
            # Binary search for quotient digit
            low = mx.array(0, dtype=mx.int32)
            high = mx.subtract(base, mx.array(1, dtype=mx.int32))
            guess = mx.array(0, dtype=mx.int32)
            
            while mx.all(mx.less_equal(low, high)):
                mid = mx.right_shift(mx.add(low, high), mx.array(1, dtype=mx.int32))
                mm = cls._mul_naive_chunklists(B, mx.array([mid], dtype=mx.int16), cls._global_chunk_size, base)
                cmpv = cls._compare_abs(mm, R)
                
                if cmpv <= 0:
                    guess = mid
                    low = mx.add(mid, mx.array(1, dtype=mx.int32))
                else:
                    high = mx.subtract(mid, mx.array(1, dtype=mx.int32))
            
            # Subtract guess * B from R
            if mx.any(mx.greater(guess, mx.array(0, dtype=mx.int32))):
                mm = cls._mul_naive_chunklists(B, mx.array([guess], dtype=mx.int16), cls._global_chunk_size, base)
                R = cls._sub_chunklists(R, mm)
            
            # Update quotient
            Q = mx.array([*Q[:i], mx.array(guess, dtype=mx.int16)[0], *Q[i+1:]])
        
        # Trim trailing zeros
        while len(Q) > 1 and Q[-1] == 0:
            Q = Q[:-1]
        
        while len(R) > 1 and R[-1] == 0:
            R = R[:-1]
        
        return (Q, R)
    
    @classmethod
    def _shiftleft_one_chunk(cls, limbs: mx.array) -> mx.array:
        """
        Shift limbs left by one chunk.
        
        Args:
            limbs: MLX array of limbs
            
        Returns:
            Shifted limbs
        """
        # Initialize output
        out = mx.zeros(len(limbs) + 1, dtype=mx.int16)
        
        # Shift limbs
        for i in range(len(limbs)):
            out = mx.array([*out[:i+1], limbs[i], *out[i+2:]])
        
        # Trim trailing zeros
        while len(out) > 1 and out[-1] == 0:
            out = out[:-1]
        
        return out
    
    @classmethod
    def _divmod_small(cls, A: mx.array, small_val: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Divide A by small_val.
        
        Args:
            A: Dividend as MLX array of limbs
            small_val: Divisor as MLX array
            
        Returns:
            Tuple of (quotient, remainder) as MLX arrays
        """
        # Initialize remainder
        remainder = mx.array(0, dtype=mx.int32)
        
        # Initialize output
        out = mx.array(A)
        
        # Divide limbs
        for i in reversed(range(len(out))):
            # Get current value
            cur = mx.add(
                mx.left_shift(remainder, cls._global_chunk_size),
                mx.array(out[i], dtype=mx.int32)
            )
            
            # Divide by small_val
            q = mx.floor_divide(cur, small_val)
            remainder = mx.remainder(cur, small_val)
            
            # Update output
            q_array = mx.array([q], dtype=mx.int16)
            out = mx.array([*out[:i], q_array[0], *out[i+1:]])
        
        # Trim trailing zeros
        while len(out) > 1 and out[-1] == 0:
            out = out[:-1]
        
        return (out, remainder)
    
    def copy(self) -> "MLXMegaNumber":
        """
        Create a copy of this MLXMegaNumber.
        
        Returns:
            Copy of this MLXMegaNumber
        """
        return MLXMegaNumber(
            mantissa=mx.array(self.mantissa),
            exponent=mx.array(self.exponent),
            negative=self.negative,
            is_float=self.is_float,
            exponent_negative=self.exponent_negative,
            keep_leading_zeros=self._keep_leading_zeros
        )
    
    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation
        """
        return f"<MLXMegaNumber {self.to_decimal_string(50)}>"


# Test the class
if __name__ == "__main__":
    # Create MLXMegaNumber objects
    a = MLXMegaNumber("123")
    b = MLXMegaNumber("456")
    
    print("a:", a)
    print("b:", b)
    
    # Test arithmetic operations
    print("a + b:", a.add(b))
    print("a - b:", a.sub(b))
    print("a * b:", a.mul(b))
    print("a / b:", a.div(b))
    
    # Test with floating-point numbers
    c = MLXMegaNumber("123.456")
    d = MLXMegaNumber("7.89")
    
    print("\nFloating-point numbers:")
    print("c:", c)
    print("d:", d)
    print("c + d:", c.add(d))
    print("c - d:", c.sub(d))
    print("c * d:", c.mul(d))
    print("c / d:", c.div(d))
