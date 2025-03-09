import time, random, math, os, pickle

class MegaNumber:
    """
    Represents a big number internally as:
      - self.mantissa: list of limb values (little-endian, base=2^chunk_size)
      - self.exponent: list of limb values (little-endian) for 'binary exponent'
      - self.negative: bool sign for the mantissa
    
    We parse decimal strings => chunk-based, do all operations chunk-based,
    and return decimal strings. The user never deals with limb arrays directly.
    """

    _global_chunk_size = None
    _base = None
    _mask = None
    _auto_detect_done = False
    _max_precision_bits = None

    def __init__(self, mantissa=None, exponent=None, negative=False, 
                 is_float=False, exponent_negative=False):
        # 1) Possibly auto-pick chunk size
        if not self._auto_detect_done:
            self._auto_pick_chunk_size()
            type(self)._auto_detect_done = True

        # 2) Default arrays
        if mantissa is None:
            mantissa = [0]
        if exponent is None:
            exponent = [0]

        # 3) Store them
        self.mantissa = mantissa
        self.exponent = exponent
        self.negative = negative
        self.is_float = is_float
        self.exponent_negative = exponent_negative

        # 4) Normalize
        self._normalize()

    # ----------------------------------------------------------------
    #       0) AUTO-DETECT CHUNK SIZE
    # ----------------------------------------------------------------
    @classmethod
    def _auto_pick_chunk_size(cls, candidates=None, test_bit_len=1024, trials=10):
        """
        Determine chunk_size from [8,16,32,64],
        measuring performance with a quick multiplication test.
        Store the best in cls._global_chunk_size, cls._base, cls._mask.
        """
        if candidates is None:
            candidates = [8, 16, 32, 64]
        best_csize = None
        best_time = float('inf')
        for csize in candidates:
            t = cls._benchmark_mul(csize, test_bit_len, trials)
            if t < best_time:
                best_time = t
                best_csize = csize
        cls._global_chunk_size = best_csize
        cls._base = 1 << best_csize
        cls._mask = cls._base - 1

    @classmethod
    def _benchmark_mul(cls, csize, bit_len, trials):
        start = time.time()
        base = 1 << csize
        for _ in range(trials):
            # Build random chunk-lists for ~bit_len bits
            A_val = random.getrandbits(bit_len)
            B_val = random.getrandbits(bit_len)
            A_limb = cls._int_to_chunklist(A_val, csize)
            B_limb = cls._int_to_chunklist(B_val, csize)
            # do multiple multiplications
            for __ in range(3):
                _ = cls._mul_chunklists(A_limb, B_limb, csize, base)
        return time.time() - start
    def _check_precision_limit(self, num: "MegaNumber"):
        """
        Checks if the given MegaNumber exceeds the maximum precision limit.
        Raises ValueError if the limit is exceeded.
        """
        if self._max_precision_bits is not None:
            total_bits = len(num.mantissa) * self._global_chunk_size
            if total_bits > self._max_precision_bits:
                raise ValueError("Precision exceeded!")
    @classmethod
    def dynamic_precision_test(cls, operation='mul', threshold_seconds=2.0, hard_limit=6.0):
        """
        Dynamically discover a feasible max bit size for 'operation' by ramping up
        until threshold_seconds is exceeded or we hit hard_limit.

        1) Start small, e.g. N=1024 bits.
        2) Double => measure time => if < threshold => keep going.
        3) If time >= hard_limit => revert to partial/binary search between last success and now.
        4) Final is stored in cls._max_precision_bits.
        """
        # If we already have a cached limit, respect it unless user calls a reset.
        if cls._max_precision_bits is not None:
            return cls._max_precision_bits

        def test_fn(bit_size):
            # Very rough measure: do a chunk-based exponent by squaring
            # or multiplication with random data.
            # For simplicity, do a quick multiply test again, or define a custom measure:
            start_t = time.time()
            # Example operation: (bit_size-bit random) ^ 50
            base_val = random.getrandbits(bit_size)
            base_mn = cls.from_int(base_val)
            exp_mn = cls.from_int(50)
            _ = base_mn.pow(exp_mn)
            return time.time() - start_t

        N = 1024
        last_good = N
        # Step 1: Exponential ramp
        while True:
            elapsed = test_fn(N)
            if elapsed < threshold_seconds:
                last_good = N
                N *= 2
            elif elapsed >= hard_limit:
                break
            else:
                # borderline => refine
                last_good = N
                break

        # Step 2: partial/binary search between last_good and N
        lower = last_good
        upper = max(N, last_good)  # in case we never actually ramped
        while (upper - lower) > 512:
            mid = (lower + upper) // 2
            elapsed_mid = test_fn(mid)
            if elapsed_mid < threshold_seconds:
                lower = mid
            elif elapsed_mid >= hard_limit:
                upper = mid
            else:
                lower = mid

        final_choice = lower
        cls._max_precision_bits = final_choice
        return final_choice

    @classmethod
    def load_cached_precision(cls, cache_file="precision.pkl"):
        """
        Load a previously pickled _max_precision_bits if it exists,
        else set from dynamic test or default.
        """
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cls._max_precision_bits = pickle.load(f)
            print(f"Loaded cached precision: {cls._max_precision_bits}")
        else:
            print("No cached precision found; will do dynamic test or use default.")

    @classmethod
    def save_cached_precision(cls, cache_file="precision.pkl"):
        """
        Write cls._max_precision_bits to a pickle, so we can skip retesting next time.
        """
        if cls._max_precision_bits is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(cls._max_precision_bits, f)
            print(f"Saved precision to {cache_file}: {cls._max_precision_bits}")
        else:
            print("No max_precision_bits to save yet.")

    def _normalize(self):
        """
        Remove leading zeros in mantissa & exponent arrays.
        If mantissa=0 => exponent=0, negative=False.
        Possibly handle is_float logic.
        """
        while len(self.mantissa) > 1 and self.mantissa[-1] == 0:
            self.mantissa.pop()
        if self.is_float:
            while len(self.exponent) > 1 and self.exponent[-1] == 0:
                self.exponent.pop()

        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            self.negative = False
            self.exponent = [0]
            self.exponent_negative = False

    @property
    def max_precision_bits(self):
        """
        Expose the internally stored or discovered max precision,
        so the user can introspect the limit.
        """
        return self._max_precision_bits

    # ----------------------------------------------------------------
    #       2) STRING <-> CHUNK
    # ----------------------------------------------------------------
    @classmethod
    def from_decimal_string(cls, dec_str: str) -> "MegaNumber":
        # Make sure chunk size is set:
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True

        s = dec_str.strip()
        if not s:
            return cls([0], [0], False)

        negative = False
        if s.startswith('-'):
            negative = True
            s = s[1:].strip()
        point_pos = s.find('.')
        frac_len = 0
        if point_pos >= 0:
            frac_len = len(s) - (point_pos + 1)
            s = s.replace('.', '')

        mant_limb = [0]
        for ch in s:
            if not ('0' <= ch <= '9'):
                raise ValueError(f"Invalid decimal digit {ch} in {dec_str}")
            digit_val = (ord(ch) - ord('0'))
            # multiply mant_limb by 10
            mant_limb = cls._mul_chunklists(
                mant_limb,
                cls._int_to_chunklist(10, cls._global_chunk_size),
                cls._global_chunk_size,
                cls._base
            )
            # add digit_val
            carry = digit_val
            idx = 0
            while carry != 0 or idx < len(mant_limb):
                if idx == len(mant_limb):
                    mant_limb.append(0)
                ssum = mant_limb[idx] + carry
                mant_limb[idx] = ssum & cls._mask
                carry = ssum >> cls._global_chunk_size
                idx += 1

        exp_limb = [0]
        if frac_len > 0:
            shift_bits = int(math.ceil(frac_len * math.log2(10)))
            if shift_bits > 0:
                exp_limb = cls._int_to_chunklist(shift_bits, cls._global_chunk_size)

        obj = cls(mantissa=mant_limb, exponent=exp_limb, negative=negative)
        obj._exp_neg = (frac_len > 0)  # store negative exponent marker
        obj._normalize()
        return obj

    def to_decimal_string(self, max_digits=None) -> str:
        # if mantissa=0 => "0"
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"

        sign_str = "-" if self.negative else ""
        is_exp_nonzero = (len(self.exponent) > 1 or self.exponent[0] != 0)
        exp_is_neg = getattr(self, "_exp_neg", False)

        if not is_exp_nonzero and not exp_is_neg:
            # pure integer
            dec_str = self._chunk_to_dec_str(self.mantissa, max_digits)
            return sign_str + dec_str
        else:
            mant_str = self._chunk_to_dec_str(self.mantissa, max_digits)
            e_val = self._chunklist_to_small_int(self.exponent, self._global_chunk_size)
            if exp_is_neg:
                e_val = -e_val
            return f"{sign_str}{mant_str} * 2^{e_val}"

    @classmethod
    def _chunk_to_dec_str(cls, limbs, max_digits=None):
        if len(limbs) == 1 and limbs[0] == 0:
            return "0"
        temp = limbs[:]
        digits = []
        while not (len(temp) == 1 and temp[0] == 0):
            temp, r = cls._divmod_small(temp, 10)
            digits.append(str(r))
        digits.reverse()
        full_str = "".join(digits)
        if max_digits is None or max_digits >= len(full_str):
            return full_str
        else:
            return f"...{full_str[-max_digits:]}"

    # ----------------------------------------------------------------
    #       3) ARITHMETIC: add, sub, mul, div, pow
    # ----------------------------------------------------------------
    def add(self, other: "MegaNumber") -> "MegaNumber":
        # 1. float addition?
        if self.is_float or other.is_float:
            return self._add_float(other)

        # 2. integer addition
        if self.negative == other.negative:
            sum_limb = self._add_chunklists(self.mantissa, other.mantissa)
            sign = self.negative
            result = MegaNumber(mantissa=sum_limb, negative=sign)
        else:
            c = self._compare_abs(self.mantissa, other.mantissa)
            if c == 0:
                return MegaNumber()  # zero
            elif c > 0:
                diff = self._sub_chunklists(self.mantissa, other.mantissa)
                result = MegaNumber(mantissa=diff, negative=self.negative)
            else:
                diff = self._sub_chunklists(other.mantissa, self.mantissa)
                result = MegaNumber(mantissa=diff, negative=other.negative)

        self._check_precision_limit(result)
        return result

    def _add_float(self, other: "MegaNumber") -> "MegaNumber":
        self_exponent = self._chunklist_to_int(self.exponent) if self.is_float else 0
        other_exponent = other._chunklist_to_int(other.exponent) if other.is_float else 0
        if self.exponent_negative:
            self_exponent = -self_exponent
        if other.exponent_negative:
            other_exponent = -other_exponent

        if self_exponent == other_exponent:
            mantissa_a, mantissa_b = self.mantissa, other.mantissa
            final_exponent = self_exponent
            final_exp_neg = self.exponent_negative
        elif self_exponent > other_exponent:
            shift_amount = self_exponent - other_exponent
            mantissa_a = self.mantissa
            mantissa_b = self._shift_right(other.mantissa, shift_amount)
            final_exponent = self_exponent
            final_exp_neg = self.exponent_negative
        else:
            shift_amount = other_exponent - self_exponent
            mantissa_a = self._shift_right(self.mantissa, shift_amount)
            mantissa_b = other.mantissa
            final_exponent = other_exponent
            final_exp_neg = other.exponent_negative

        if self.negative == other.negative:
            diff = self._add_chunklists(mantissa_a, mantissa_b)
            sign = self.negative
        else:
            c = self._compare_abs(mantissa_a, mantissa_b)
            if c == 0:
                return MegaNumber(is_float=True)
            elif c > 0:
                diff = self._sub_chunklists(mantissa_a, mantissa_b)
                sign = self.negative
            else:
                diff = self._sub_chunklists(mantissa_b, mantissa_a)
                sign = other.negative

        result = MegaNumber(
            mantissa=diff,
            exponent=self._int_to_chunklist(abs(final_exponent)),
            negative=sign,
            is_float=True,
            exponent_negative=final_exp_neg
        )
        self._check_precision_limit(result)
        return result

    def sub(self, other: "MegaNumber") -> "MegaNumber":
        neg_other = MegaNumber(other.mantissa[:], other.exponent[:], not other.negative)
        return self.add(neg_other)

    def mul(self, other: "MegaNumber") -> "MegaNumber":
        # minimal integer-only logic
        if (len(self.exponent) > 1 or self.exponent[0] != 0) or hasattr(self, '_exp_neg'):
            raise NotImplementedError("Floating mul not in minimal example.")
        if (len(other.exponent) > 1 or other.exponent[0] != 0) or hasattr(other, '_exp_neg'):
            raise NotImplementedError("Floating mul not in minimal example.")

        sign = (self.negative != other.negative)
        out_limb = self._mul_chunklists(self.mantissa, other.mantissa, self._global_chunk_size, self._base)
        out = MegaNumber(out_limb, [0], sign)
        out._normalize()
        return out

    def div(self, other: "MegaNumber") -> "MegaNumber":
        # minimal integer-only
        if (len(self.exponent) > 1 or self.exponent[0] != 0) or hasattr(self, '_exp_neg'):
            raise NotImplementedError("Floating div not in minimal example.")
        if (len(other.exponent) > 1 or other.exponent[0] != 0) or hasattr(other, '_exp_neg'):
            raise NotImplementedError("Floating div not in minimal example.")

        if len(other.mantissa) == 1 and other.mantissa[0] == 0:
            raise ZeroDivisionError("division by zero")

        sign = (self.negative != other.negative)
        c = self._compare_abs(self.mantissa, other.mantissa)
        if c < 0:
            return MegaNumber([0], [0], False)
        elif c == 0:
            return MegaNumber([1], [0], sign)
        else:
            q, r = self._div_chunk(self.mantissa, other.mantissa)
            out = MegaNumber(q, [0], sign)
            out._normalize()
            return out

    def pow(self, exponent: "MegaNumber") -> "MegaNumber":
        # integer exponent-by-squaring
        if self._global_chunk_size is None:
            self._auto_pick_chunk_size()
            self._auto_detect_done = True
        if exponent.negative:
            raise NotImplementedError("Negative exponent not supported in minimal example.")
        e_val = self._chunklist_to_small_int(exponent.mantissa, self._global_chunk_size)
        if e_val < 0:
            raise ValueError("Negative exponent not supported here.")

        base_copy = MegaNumber(self.mantissa[:], [0], self.negative)
        result = MegaNumber([1], [0], False)
        while e_val > 0:
            if (e_val & 1) == 1:
                result = result.mul(base_copy)
            base_copy = base_copy.mul(base_copy)
            e_val >>= 1
        return result

    def sqrt(self) -> "MegaNumber":
        # integer sqrt only
        if (len(self.exponent) > 1 or self.exponent[0] != 0) or hasattr(self, '_exp_neg'):
            raise NotImplementedError("Floating sqrt not in minimal example.")

        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return MegaNumber([0], [0], False)
        A = self.mantissa
        low = [0]
        high = A[:]
        if self._global_chunk_size is None:
            self._auto_pick_chunk_size()
            self._auto_detect_done = True
        csize = self._global_chunk_size
        base = self._base

        while True:
            sum_lh = self._add_chunklists(low, high)
            mid = self._div2(sum_lh)
            c_lo = self._compare_abs(mid, low)
            c_hi = self._compare_abs(mid, high)
            if c_lo == 0 or c_hi == 0:
                return MegaNumber(mid, [0], False)
            mid_sqr = self._mul_chunklists(mid, mid, csize, base)
            c_cmp = self._compare_abs(mid_sqr, A)
            if c_cmp == 0:
                return MegaNumber(mid, [0], False)
            elif c_cmp < 0:
                low = mid
            else:
                high = mid

    # ----------------------------------------------------------------
    #       5) INTERNAL CHUNK UTILS
    # ----------------------------------------------------------------
    @classmethod
    def _int_to_chunklist(cls, val, csize):
        if val == 0:
            return [0]
        out = []
        while val > 0:
            out.append(val & ((1 << csize) - 1))
            val >>= csize
        return out

    @classmethod
    def _chunklist_to_small_int(cls, limbs, csize):
        val = 0
        shift = 0
        for limb in limbs:
            val += (limb << shift)
            shift += csize
        return val

    @classmethod
    def from_int(cls, val: int) -> "MegaNumber":
        """
        Build a MegaNumber from a Python integer.
        """
        if val == 0:
            return cls(mantissa=[0], exponent=[0], negative=False)
        negative = (val < 0)
        val_abs = abs(val)
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        limbs = cls._int_to_chunklist(val_abs, cls._global_chunk_size)
        return cls(mantissa=limbs, exponent=[0], negative=negative)

    @classmethod
    def _compare_abs(cls, A, B):
        if len(A) > len(B):
            return 1
        if len(B) > len(A):
            return -1
        for i in range(len(A)-1, -1, -1):
            if A[i] > B[i]:
                return 1
            elif A[i] < B[i]:
                return -1
        return 0

    @classmethod
    def _mul_chunklists(cls, A, B, csize, base):
        la = len(A)
        lb = len(B)
        out = [0]*(la+lb)
        for i in range(la):
            carry = 0
            av = A[i]
            for j in range(lb):
                mul_val = av * B[j] + out[i+j] + carry
                out[i+j] = mul_val & (base-1)
                carry = mul_val >> csize
            if carry:
                out[i+lb] += carry
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _div_chunk(cls, A, B):
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        if len(B) == 1 and B[0] == 0:
            raise ZeroDivisionError("divide by zero")
        c = cls._compare_abs(A, B)
        if c < 0:
            return ([0], A)
        if c == 0:
            return ([1], [0])

        Q = [0]*len(A)
        R = [0]
        for i in range(len(A)-1, -1, -1):
            R = cls._mul_chunklists(R, [cls._base], cls._global_chunk_size, cls._base)
            R = cls._add_chunklists(R, [A[i]])
            low, high = 0, cls._base-1
            guess = 0
            while low <= high:
                mid = (low+high) >> 1
                mm = cls._mul_chunklists(B, [mid], cls._global_chunk_size, cls._base)
                cmpv = cls._compare_abs(mm, R)
                if cmpv <= 0:
                    guess = mid
                    low = mid+1
                else:
                    high = mid-1
            if guess != 0:
                mm = cls._mul_chunklists(B, [guess], cls._global_chunk_size, cls._base)
                R = cls._sub_chunklists(R, mm)
            Q[i] = guess
        while len(Q) > 1 and Q[-1] == 0:
            Q.pop()
        while len(R) > 1 and R[-1] == 0:
            R.pop()
        return (Q,R)

    @classmethod
    def _divmod_small(cls, A, small_val):
        remainder = 0
        out = [0]*len(A)
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        for i in reversed(range(len(A))):
            cur = (remainder << cls._global_chunk_size) + A[i]
            qd = cur // small_val
            remainder = cur % small_val
            out[i] = qd & cls._mask
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return (out, remainder)

    @classmethod
    def _add_chunklists(cls, A, B):
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        out = []
        carry = 0
        max_len = max(len(A), len(B))
        for i in range(max_len):
            av = A[i] if i < len(A) else 0
            bv = B[i] if i < len(B) else 0
            s = av + bv + carry
            carry = s >> cls._global_chunk_size
            out.append(s & cls._mask)
        if carry:
            out.append(carry)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _sub_chunklists(cls, A, B):
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        out = []
        carry = 0
        max_len = max(len(A), len(B))
        for i in range(max_len):
            av = A[i] if i < len(A) else 0
            bv = B[i] if i < len(B) else 0
            diff = av - bv - carry
            if diff < 0:
                diff += cls._base
                carry = 1
            else:
                carry = 0
            out.append(diff & cls._mask)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _div2(cls, limbs):
        out = []
        carry = 0
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        csize = cls._global_chunk_size
        for i in reversed(range(len(limbs))):
            val = (carry << csize) + limbs[i]
            q = val >> 1
            r = val & 1
            carry = r
            out.append(q)
        out.reverse()
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    def copy(self) -> "MegaNumber":
        return MegaNumber(self.mantissa[:], self.exponent[:], self.negative)

    def __repr__(self):
        return f"<MegaNumber {self.to_decimal_string(50)}>"

# ----------------------------------------------------------------
#   USER-FACING: STRING IN, STRING OUT
# ----------------------------------------------------------------
def add_str(a_str: str, b_str: str) -> str:
    A = MegaNumber.from_decimal_string(a_str)
    B = MegaNumber.from_decimal_string(b_str)
    C = A.add(B)
    return C.to_decimal_string()

def sub_str(a_str: str, b_str: str) -> str:
    A = MegaNumber.from_decimal_string(a_str)
    B = MegaNumber.from_decimal_string(b_str)
    C = A.sub(B)
    return C.to_decimal_string()

def mul_str(a_str: str, b_str: str) -> str:
    A = MegaNumber.from_decimal_string(a_str)
    B = MegaNumber.from_decimal_string(b_str)
    C = A.mul(B)
    return C.to_decimal_string()

def div_str(a_str: str, b_str: str) -> str:
    A = MegaNumber.from_decimal_string(a_str)
    B = MegaNumber.from_decimal_string(b_str)
    C = A.div(B)
    return C.to_decimal_string()

def pow_str(base_str: str, exp_str: str) -> str:
    base_obj = MegaNumber.from_decimal_string(base_str)
    exp_obj  = MegaNumber.from_decimal_string(exp_str)
    out_obj = base_obj.pow(exp_obj)  # must be integer exponent
    return out_obj.to_decimal_string()

def sqrt_str(x_str: str) -> str:
    X = MegaNumber.from_decimal_string(x_str)
    Y = X.sqrt()
    return Y.to_decimal_string()


# ----------------------------------------------------------------
#   SKELETON FOR BLOCKYNUMBER
# ----------------------------------------------------------------
class BlockyNumber(MegaNumber):
    """
    Advanced HPC/arbitrary-precision class, extending MegaNumber:
      - HPC concurrency for trig and wave operations.
      - Shared memory expansions, kill switch, “vMotion” logic, etc.
      - Binary-focused math for sin/cos/tan, sigmoid, wave transforms.
      - No rounding, all chunk-based expansions.

    This skeleton stands ready for advanced methods. For now, it inherits
    from MegaNumber and could override or extend as needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional HPC or concurrency fields can be set up here.

    def sin(self):
        """
        Placeholder: compute sin(self) in chunk-based form,
        possibly parallel HPC approach. 
        """
        raise NotImplementedError("sin not yet implemented in BlockyNumber")

    def cos(self):
        raise NotImplementedError("cos not yet implemented in BlockyNumber")

    def tan(self):
        raise NotImplementedError("tan not yet implemented in BlockyNumber")

    def sigmoid(self):
        raise NotImplementedError("sigmoid not yet implemented in BlockyNumber")

    def kill_switch(self):
        """
        HPC 'kill switch' or 'vMotion' logic for safely suspending or migrating
        large computations. Not yet implemented.
        """
        raise NotImplementedError("kill_switch not yet implemented in BlockyNumber")
    
    
import random
import time


def test_exponentiation(bit_count: int, exponent: int) -> float:
    """
    Perform a 'difficult' chunk-based exponentiation at 'bit_count' bits,
    measuring total time in seconds. If it surpasses some large threshold,
    you can forcibly stop or just let it run. Here, we'll let it run
    for simplicity. Return the elapsed time.
    """
    # 1) Build random ~bit_count-bit base
    base_val = random.getrandbits(bit_count)

    # 2) Convert to chunk-based MegaNumber
    base_mn = MegaNumber.from_int(base_val)
    exp_mn  = MegaNumber.from_int(exponent)

    # 3) Time exponentiation (base_mn^exponent)
    start_t = time.time()
    _result = base_mn.pow(exp_mn)  # chunk-based exponent by squaring
    elapsed = time.time() - start_t

    return elapsed

def dynamic_precision_test_stepwise(
    initial_bits: int = 1024,
    real_threshold: float = 2.0,
    hard_limit: float = 6.0,
    exponent: int = 1000
) -> int:
    """
    1) Start at 'initial_bits'.
    2) Exponential ramp:
       - double bit_count if we run under 'real_threshold' seconds.
       - if we exceed 'hard_limit', revert to last success, do narrower search.
    3) Return the final feasible bit size.

    'real_threshold' = the target max seconds for an operation (like 2s).
    'hard_limit'     = a bigger bounding time (like 6s) to avoid huge overshoots.
    """
    N = initial_bits
    last_good = N

    # ========== Step 1: Exponential ramp ==========
    while True:
        elapsed = test_exponentiation(N, exponent)
        print(f"[STEP] {N} bits => {elapsed:.3f}s")

        if elapsed < real_threshold:
            # Good => try doubling
            last_good = N
            N *= 2
        elif elapsed >= hard_limit:
            # We overshot big-time => revert to last_good, do partial search
            print(f"[WARN] Exceeded hard_limit at {N} bits => {elapsed:.3f}s. Reverting to partial search.")
            break
        else:
            # We are between real_threshold and hard_limit => borderline
            print(f"[INFO] {N} bits => {elapsed:.3f}s is between {real_threshold}..{hard_limit}, let's refine.")
            last_good = N
            break

    # ========== Step 2: Partial / Binary search in [last_good..N] ==========
    lower = last_good
    upper = N
    while (upper - lower) > 512:  # or a smaller step if you prefer
        mid = (lower + upper) // 2
        elapsed_mid = test_exponentiation(mid, exponent)
        print(f"[PARTIAL] mid={mid} => {elapsed_mid:.3f}s")

        if elapsed_mid < real_threshold:
            lower = mid  # we can do bigger
        elif elapsed_mid >= hard_limit:
            upper = mid  # too big
        else:
            # It's in the middle => we can accept mid but see if we can push a bit more
            lower = mid

    final_choice = lower
    print(f"[RESULT] Final feasible bit size ~ {final_choice}")
    return final_choice

import array
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Dict
import mpmath
mpmath.mp.prec = 20000
#
# 1) BlockMetrics + CPUMemoryPool
#
@dataclass
class BlockMetrics:
    """Track performance of block operations, memory usage, and timing."""
    block_hits: int = 0           # times we reused a buffer
    cache_misses: int = 0         # times we allocated a new buffer
    total_ops: int = 0            # generic counter if needed
    peak_memory: int = 0          # track largest memory usage
    # timing for key algorithms
    time_spent: Dict[str, float] = None

    def __post_init__(self):
        self.time_spent = {
            'schoolbook': 0.0,
            'karatsuba': 0.0,
            'toom3': 0.0,
            'evaluation': 0.0,
            'interpolation': 0.0
        }

class CPUMemoryPool:
    """Thread-safe memory pool for array reuse"""
    def __init__(self):
        self._lock = threading.Lock()
        self.pools: Dict[int, List[array.array]] = {}
        self.stats = BlockMetrics()

    def get_buffer(self, size: int) -> array.array:
        """Get buffer, aligned to multiples of 8 for 64-byte alignment."""
        aligned_size = (size + 7) & ~7
        with self._lock:
            if aligned_size in self.pools and self.pools[aligned_size]:
                self.stats.block_hits += 1
                return self.pools[aligned_size].pop()
            self.stats.cache_misses += 1
            buf = array.array('Q', [0]*aligned_size)
            # optional memory tracking
            cur_mem = sum(len(lst)*aligned_size for lst in self.pools.values())
            self.stats.peak_memory = max(self.stats.peak_memory, cur_mem)
            return buf

    def return_buffer(self, buf: array.array) -> None:
        """Return buffer to pool for reuse."""
        size = (len(buf) + 7) & ~7
        with self._lock:
            if size not in self.pools:
                self.pools[size] = []
            self.pools[size].append(buf)

#
# 2) OptimizedToom3 with Tiered Multiplication & Advanced Routines
#
class OptimizedToom3:
    """
    A CPU-first big-int multiplication system:
      - small => schoolbook
      - mid => karatsuba
      - large => block-based Toom-3
    Also includes advanced ops like exponent, factorial, fibonacci.
    """

    L1_CACHE = 32768  # 32KB
    CACHE_LINE = 64

    def __init__(self, pool: CPUMemoryPool):
        self.pool = pool
    def power(self, base: array.array, exponent: int) -> array.array:
        if exponent == 0:
            return array.array('Q', [1])
        temp_base = array.array(base.typecode, base)
        result = array.array('Q', [1])
        while exponent > 0:
            if exponent & 1:
                result = self.multiply(result, temp_base)
            temp_base = self.multiply(temp_base, temp_base)
            exponent >>= 1
        return result
    def multiply(self, a: array.array, b: array.array) -> array.array:
        """Dispatch to the appropriate multiplication algorithm."""
        n = max(len(a), len(b))
        if n < 32:
            t0 = time.perf_counter()
            out = self._multiply_schoolbook(a, b)
            self.pool.stats.time_spent['schoolbook'] += (time.perf_counter() - t0)
            return out
        elif n < 128:
            t0 = time.perf_counter()
            out = self._multiply_karatsuba(a, b)
            self.pool.stats.time_spent['karatsuba'] += (time.perf_counter() - t0)
            return out
        else:
            t0 = time.perf_counter()
            out = self._multiply_toom_3(a, b)
            self.pool.stats.time_spent['toom3'] += (time.perf_counter() - t0)
            return out

    # ----------------------
    # A) SCHOOLBOOK
    # ----------------------
    def _multiply_schoolbook(self, a: array.array, b: array.array) -> array.array:
        n = len(a)
        m = len(b)
        out = array.array('Q', [0]*(n+m))
        for i in range(n):
            carry = 0
            for j in range(m):
                s_val = out[i+j] + a[i]*b[j] + carry
                out[i+j] = s_val & ((1<<64)-1)
                carry = s_val >> 64
            if carry:
                out[i+m] += carry
        # Trim
        while len(out)>1 and out[-1]==0:
            out.pop()
        return out

    # ----------------------
    # B) KARATSUBA
    # ----------------------
    def _multiply_karatsuba(self, a: array.array, b: array.array) -> array.array:
        n = max(len(a), len(b))
        if n < 32:
            return self._multiply_schoolbook(a, b)

        half = n//2
        a0, a1 = a[:half], a[half:]
        b0, b1 = b[:half], b[half:]

        z0 = self._multiply_karatsuba(a0, b0)
        z2 = self._multiply_karatsuba(a1, b1)

        sum_a = self._add_arrays(a0, a1)
        sum_b = self._add_arrays(b0, b1)
        z1 = self._multiply_karatsuba(sum_a, sum_b)

        self._subtract_in_place(z1, z0)
        self._subtract_in_place(z1, z2)

        result = array.array('Q',[0]*(n*2))
        self._add_shifted(result, z0, 0)
        self._add_shifted(result, z1, half)
        self._add_shifted(result, z2, half*2)

        while len(result)>1 and result[-1]==0:
            result.pop()
        return result

    def _add_shifted(self, target: array.array, source: array.array, shift: int) -> None:
        """Wrapper for merges in Karatsuba; delegates to block-based approach with small block_size=64."""
        self._blocked_add_shift(target, source, shift, block_size=64)

    # ----------------------
    # C) TOOM-3 BLOCK-BASED
    # ----------------------
    def _multiply_toom_3(self, a: array.array, b: array.array) -> array.array:
        n = max(len(a), len(b))
        chunk_size = (n+2)//3
        block_size = self.L1_CACHE // (8*3)

        # 1) split
        a_chunks = self._blocked_split(a, chunk_size, 3, block_size)
        b_chunks = self._blocked_split(b, chunk_size, 3, block_size)

        # 2) evaluate
        t0 = time.perf_counter()
        a_evals = self._blocked_evaluate(a_chunks, block_size)
        b_evals = self._blocked_evaluate(b_chunks, block_size)
        self.pool.stats.time_spent['evaluation'] += (time.perf_counter() - t0)

        # 3) multiply sub-evals
        products = []
        for ae, be in zip(a_evals, b_evals):
            products.append(self.multiply(ae, be))

        # 4) interpolate
        t0 = time.perf_counter()
        result = self._blocked_interpolate(products, chunk_size, block_size)
        self.pool.stats.time_spent['interpolation'] += (time.perf_counter() - t0)

        return result

    # 1) splitting
    def _blocked_split(self, 
                       num: array.array,
                       chunk_size: int,
                       num_chunks: int,
                       block_size: int) -> List[array.array]:
        chunks = []
        for cidx in range(num_chunks):
            chunk = self.pool.get_buffer(chunk_size)
            base_offset = cidx*chunk_size
            for start in range(0, chunk_size, block_size):
                end = min(start+block_size, chunk_size)
                for i in range(start, end):
                    src_idx = base_offset + i
                    chunk[i] = num[src_idx] if src_idx<len(num) else 0
            chunks.append(chunk)
        return chunks

    # 2) evaluate at x=0,1,-1,2,∞
    def _blocked_evaluate(self, chunks: List[array.array], block_size: int) -> List[array.array]:
        out0 = chunks[0]
        out1 = self._blocked_sum(chunks, block_size)
        outm = self._blocked_alternating_sum(chunks, block_size)
        out2 = self._blocked_weighted_sum(chunks, 2, block_size)
        out_inf = chunks[-1]
        return [out0, out1, outm, out2, out_inf]

    # 3) interpolation
    def _blocked_interpolate(self, products: List[array.array],
                             chunk_size: int, block_size: int) -> array.array:
        p0, p1, pm1, p2, pinf = products

        # v2 = (p2 - p1)//3
        v2 = self._blocked_sub(p2, p1, block_size)
        self._blocked_div3(v2, block_size)
        # v3 = (p1 - pm1)//2
        v3 = self._blocked_sub(p1, pm1, block_size)
        self._blocked_div2(v3, block_size)
        # v1 = p1 - p0
        v1 = self._blocked_sub(p1, p0, block_size)
        v4 = pinf

        out_len = chunk_size*5
        result = array.array('Q', [0]*out_len)
        self._blocked_add_shift(result, p0, 0, block_size)
        self._blocked_add_shift(result, v1, chunk_size, block_size)
        self._blocked_add_shift(result, v2, chunk_size*2, block_size)
        self._blocked_add_shift(result, v3, chunk_size*3, block_size)
        self._blocked_add_shift(result, v4, chunk_size*4, block_size)

        while len(result)>1 and result[-1]==0:
            result.pop()
        return result

    # 4) block-based summation routines
    def _blocked_sum(self, arrays: List[array.array], block_size: int) -> array.array:
        if not arrays:
            return array.array('Q',[0])
        # create new array from arrays[0]
        result = array.array(arrays[0].typecode, arrays[0])
        length = len(result)
        for arr in arrays[1:]:
            for start in range(0, length, block_size):
                end = min(start+block_size, length)
                carry = 0
                for i in range(start, end):
                    s_val = result[i] + arr[i] + carry
                    result[i] = s_val & ((1<<64)-1)
                    carry = s_val>>64
                if carry and end<length:
                    result[end] += carry
        return result

    def _blocked_alternating_sum(self, arrays: List[array.array], block_size: int) -> array.array:
        if not arrays:
            return array.array('Q',[0])
        result = array.array(arrays[0].typecode, arrays[0])
        length = len(result)
        for idx, arr in enumerate(arrays[1:], 1):
            sign_add = ((idx & 1)==0)
            for start in range(0, length, block_size):
                end = min(start+block_size, length)
                carry = 0
                for i in range(start, end):
                    if sign_add:
                        s_val = result[i] + arr[i] + carry
                        result[i] = s_val & ((1<<64)-1)
                        carry = s_val>>64
                    else:
                        s_val = result[i] - arr[i] - carry
                        result[i] = s_val & ((1<<64)-1)
                        carry = (s_val>>64) & 1
                if carry and end<length:
                    idx2 = end
                    if sign_add:
                        # carry means increment next limb
                        while carry and idx2<length:
                            s_val = result[idx2] + carry
                            result[idx2] = s_val & ((1<<64)-1)
                            carry = s_val>>64
                            idx2+=1
                    else:
                        # carry is a borrow
                        while carry and idx2<length:
                            diff = result[idx2] - carry
                            result[idx2] = diff & ((1<<64)-1)
                            carry = (diff>>64) & 1
                            idx2+=1
        return result

    def _blocked_weighted_sum(self, arrays: List[array.array],
                              w: int, block_size: int) -> array.array:
        if not arrays:
            return array.array('Q',[0])
        result = array.array(arrays[0].typecode, arrays[0])
        length = len(result)
        cur_w = 1
        for arr in arrays[1:]:
            cur_w *= w
            for start in range(0, length, block_size):
                end = min(start+block_size, length)
                carry = 0
                c2 = 0
                for i in range(start, end):
                    prod = arr[i]*cur_w + c2
                    c2   = prod>>64
                    s_val = result[i] + (prod & ((1<<64)-1)) + carry
                    result[i] = s_val & ((1<<64)-1)
                    carry = s_val>>64
                idx = end
                while carry and idx<length:
                    s_val = result[idx] + carry
                    result[idx] = s_val & ((1<<64)-1)
                    carry = s_val>>64
                    idx+=1
                while c2 and idx<length:
                    s_val = result[idx] + c2
                    result[idx] = s_val & ((1<<64)-1)
                    c2 = s_val>>64
                    idx+=1
        return result

    # 5) block-based sub / div
    def _blocked_sub(self, a: array.array, b: array.array, block_size: int) -> array.array:
        length = max(len(a), len(b))
        out = array.array(a.typecode, a)
        if len(out) < length:
            out.extend([0]*(length-len(out)))
        carry = 0
        for start in range(0, length, block_size):
            end = min(start+block_size, length)
            for i in range(start, end):
                av = out[i] if i<len(out) else 0
                bv = b[i] if i<len(b) else 0
                diff = av - bv - carry
                out[i] = diff & ((1<<64)-1)
                carry  = (diff>>64) & 1
        return out

    def _blocked_div2(self, arr: array.array, block_size: int) -> None:
        carry = 0
        length = len(arr)
        for start in range(length-block_size, -block_size, -block_size):
            s_begin = max(0, start)
            for i in reversed(range(s_begin, s_begin+block_size)):
                if i<0 or i>=length: 
                    continue
                cur = (carry<<64) | arr[i]
                arr[i] = cur>>1
                carry  = cur & 1

    def _blocked_div3(self, arr: array.array, block_size: int) -> None:
        carry = 0
        length = len(arr)
        for start in range(length-block_size, -block_size, -block_size):
            s_begin = max(0, start)
            for i in reversed(range(s_begin, s_begin+block_size)):
                if i<0 or i>=length:
                    continue
                cur = (carry<<64) | arr[i]
                q   = cur//3
                r   = cur%3
                arr[i] = q
                carry  = r

    def _blocked_add_shift(self, target: array.array, source: array.array,
                           shift: int, block_size: int) -> None:
        carry = 0
        length = len(target)
        for start in range(0, len(source), block_size):
            end = min(start+block_size, len(source))
            for i in range(start, end):
                idx = i+shift
                if idx>=length:
                    break
                s_val = target[idx] + source[i] + carry
                target[idx] = s_val & ((1<<64)-1)
                carry = s_val>>64
            idx = shift+end
            while carry and idx<length:
                s_val = target[idx] + carry
                target[idx] = s_val & ((1<<64)-1)
                carry = s_val>>64
                idx+=1

    # ------------------------------------------------------
    # ADVANCED: power, factorial, fibonacci
    # ------------------------------------------------------
    def power(self, base: array.array, exponent: int) -> array.array:
        """
        Basic square-and-multiply exponentiation using self.multiply for big-int limbs.
        """
        if exponent == 0:
            return array.array('Q', [1])  # 1-limb array = 1

        # Make a fresh copy of base to avoid modifying the original in-place
        temp_base = array.array(base.typecode, base)
        result = array.array('Q', [1])  # start result as 1

        while exponent > 0:
            if (exponent & 1) == 1:  # if odd exponent
                result = self.multiply(result, temp_base)
            temp_base = self.multiply(temp_base, temp_base)
            exponent >>= 1

        return result
    def factorial(self, n: int) -> array.array:
        """Compute n! using repeated multiplication in limbs."""
        # Start from 1
        fact = array.array('Q',[1])
        for i in range(2, n+1):
            # Convert i to limbs
            i_val = array.array('Q', [i])
            fact = self.multiply(fact, i_val)
        return fact
    
    def fibonacci(self, n: int) -> array.array:
        """Compute F_n with matrix exponent or repeated addition approach."""
        if n==0:
            return array.array('Q',[0])
        elif n==1:
            return array.array('Q',[1])
        
        # Use repeated addition approach for demonstration (can do matrix exponent too)
        f0 = array.array('Q',[0])
        f1 = array.array('Q',[1])
        for i in range(2, n+1):
            temp = self.multiply(f0, array.array('Q',[0]))  # dummy usage or skip
            # Actually: f2 = f0 + f1
            f2 = self._add_arrays(f0, f1)
            f0 = f1
            f1 = f2
        return f1

    # ------------------------------------------------------
    # HELPER routines used by advanced methods
    # ------------------------------------------------------
    def _add_arrays(self, a: array.array, b: array.array) -> array.array:
        length = max(len(a), len(b))
        out = array.array('Q',[0]*length)
        carry = 0
        for i in range(length):
            av = a[i] if i<len(a) else 0
            bv = b[i] if i<len(b) else 0
            s_val = av + bv + carry
            out[i] = s_val & ((1<<64)-1)
            carry = s_val>>64
        if carry:
            out.append(carry)
        return out

    def _subtract_in_place(self, target: array.array, source: array.array) -> None:
        carry = 0
        for i in range(len(target)):
            sv = source[i] if i<len(source) else 0
            diff = target[i] - sv - carry
            target[i] = diff & ((1<<64)-1)
            carry = (diff>>64) & 1

#
# 3) Single-file DEMO
#
def verify_exp_with_mpmath(base_int: int, exponent_int: int) -> bool:
    # Convert Python int => mpmath.mpf for floating
    base_mpf = mpmath.mpf(base_int)
    # do exponent
    approx_val = mpmath.power(base_mpf, exponent_int)
    
    # Meanwhile, your code's integer exponent result:
    python_exact_val = pow(base_int, exponent_int)
    
    # For a magnitude check:
    approx_log10 = mpmath.log10(approx_val)
    exact_log10  = mpmath.log10(mpmath.mpf(python_exact_val))

    # Compare difference in logs as a measure
    diff = abs(approx_log10 - exact_log10)

    # If difference is extremely small (like < 1e-10), it's good
    return diff < 1e-10



def advanced_power_test_with_mpmath(base, exponent):
    base_val = limbs_to_int(base)
    python_result = pow(base_val, exponent)         # exact integer
    # approximate float check
    base_mpf = mpmath.mpf(base_val)
    float_approx = mpmath.power(base_mpf, exponent)
    
    # Compare magnitudes or leading digits
    magnitude_diff = abs(mpmath.log10(float_approx) - 
                         mpmath.log10(mpmath.mpf(python_result)))
    if magnitude_diff < 1e-10:
        print("Matches magnitude in mpmath!")
    else:
        print("Mismatch in magnitude vs. mpmath reference.")
def compare_power_with_mpmath(base_arr: array.array, exponent: int, big_int_obj) -> bool:
    """
    1) Convert base_arr to Python int
    2) Compute exponent with your big-int code
    3) Convert that result to Python int
    4) Also compute an approximate float with mpmath
    5) Compare log10 difference
    Returns True if magnitude difference is small, else False.
    """
    from mpmath import mp, mpf, log10, power
    
    # 1) Convert base_arr => python int
    base_val = limbs_to_int(base_arr)
    
    # 2) big-int exponent
    big_result_arr = big_int_obj.power(base_arr, exponent)
    
    # 3) Convert big_result_arr => python int
    actual_val = limbs_to_int(big_result_arr)
    
    # 4) mpmath approximate float
    base_mpf = mpf(base_val)
    approx_float = power(base_mpf, exponent)  # floating approximation
    
    # Compare log10 => difference
    # We'll do an absolute difference in log10. If it's < 1e-10, we call it good.
    # For extremely large exponent, you might set a less strict threshold if you prefer.
    approx_log10 = log10(approx_float)
    exact_log10  = log10(mpf(actual_val))
    diff = abs(approx_log10 - exact_log10)
    
    return (diff < 1e-10)
def demo():
    pool = CPUMemoryPool()
    big_int = OptimizedToom3(pool)

    print("=== BigInteger CPU-Only Implementation Stress Tests ===\n")

    print("Running basic multiplication tests...")
    test_bits = [64,128,256,512,1024,2048,4096]
    total_pass, total_fail = 0, 0
    for bits in test_bits:
        print(f"\nTesting {bits}-bit multiplication:")
        # 5 random tests
        for test in range(5):
            # random arrays
            limbs = (bits+63)//64
            a = array.array('Q',[random.getrandbits(64) for _ in range(limbs)])
            b = array.array('Q',[random.getrandbits(64) for _ in range(limbs)])
            # expected
            a_val = limbs_to_int(a)
            b_val = limbs_to_int(b)
            expected_val = a_val * b_val

            result = big_int.multiply(a,b)
            actual_val = limbs_to_int(result)
            if actual_val == expected_val:
                total_pass+=1
                print(f"  Test {test+1}: PASS")
            else:
                total_fail+=1
                print(f"  Test {test+1}: FAIL")

    print(f"\nSummary:\nPassed: {total_pass}\nFailed: {total_fail}")
    print(f"\nBasic tests complete - {total_pass+total_fail}/{total_pass+total_fail} passed")

    # Advanced tests demonstration
    print("\n=== Starting Advanced Stress Tests ===\n")    
    print("Running power tests...")
    # in your advanced tests:
    print("Running power tests...")
    for bits in [16, 32, 64, 128, 256, 512, 1024]:
        base_limbs = (bits + 63)//64
        base_arr = array.array('Q',[random.getrandbits(64) for _ in range(base_limbs)])
        exponent = 1024

        start_t = time.perf_counter()
        pow_result = big_int.power(base_arr, exponent)  # your big-int exponent code
        dur = time.perf_counter() - start_t

        # direct integer check
        base_val = limbs_to_int(base_arr)
        python_result = pow(base_val, exponent)
        actual_val = limbs_to_int(pow_result)
        pass_fail = "✓" if actual_val == python_result else "✗"

        print(f"  {bits}-bit base ^ {exponent}: {dur:.3f}s {pass_fail}")

        #  Additional approximate check with mpmath:
        mpmath_ok = compare_power_with_mpmath(base_arr, exponent, big_int)
        if mpmath_ok:
            print("     [mpmath magnitude check => PASS]")
        else:
            print("     [mpmath magnitude check => FAIL]")
    


    print("\nRunning factorial tests...")
    for n in [10,50,100,500,1000]:
        start_t=time.perf_counter()
        fact_res= big_int.factorial(n)
        dur= time.perf_counter()-start_t
        # check with python
        expected_val=1
        for i in range(2,n+1):
            expected_val *= i
        actual_val= limbs_to_int(fact_res)
        pass_fail="✓" if actual_val==expected_val else "✗"
        print(f"  {n}!: {dur:.3f}s {pass_fail}")

    print("\nRunning Fibonacci tests...")
    for n in [100,250,500,750,1000]:
        start_t= time.perf_counter()
        fib_res= big_int.fibonacci(n)
        dur= time.perf_counter()-start_t
        # check with python
        expected_val= fib_py(n)  # we'll define fib_py below
        actual_val= limbs_to_int(fib_res)
        pass_fail= "✓" if actual_val==expected_val else "✗"  # or just measure time
        print(f"  F_{n}: {dur:.3f}s {pass_fail}")

    # Memory usage stats
    print("\n=== Detailed Results ===")
    # Not a real track in this snippet, but you can read pool.stats
    print(f"\n=== Memory Statistics ===")
    print(f"Peak memory usage: {pool.stats.peak_memory/1024/1024:.1f}MB")
    hits=pool.stats.block_hits
    misses=pool.stats.cache_misses
    total= hits+misses
    rate= (hits/total*100) if total>0 else 0
    print(f"Buffer reuse rate: {rate:.1f}%")

    print("\n=== Test Suite Complete ===")

def limbs_to_int(limbs: array.array) -> int:
    """Convert array of 64-bit limbs => python int"""
    val=0
    for i, limb in enumerate(limbs):
        val += limb<<(64*i)
    return val

def fib_py(n: int) -> int:
    """Python reference fibonacci for validation"""
    if n<2:
        return n
    f0, f1=0,1
    for _ in range(2,n+1):
        f0, f1= f1, f0+f1
    return f1




def main():
    # Tweak these as desired
    initial_bits = 1024
    real_threshold = 2.0
    hard_limit = 6.0
    exponent = 1000
    demo()
    feasible_bits = dynamic_precision_test_stepwise(
        initial_bits=initial_bits,
        real_threshold=real_threshold,
        hard_limit=hard_limit,
        exponent=exponent
    )
    print(f"\n[FINAL] We concluded feasible bits ~ {feasible_bits}")

if __name__ == "__main__":
    main()

    # ----------------------------------------------------------------
    #    DEMO
    # ----------------------------------------------------------------
    print("\n=== Basic Demo ===")
    aS = "9999999999999999999999"
    bS = "1234567890123456789"

    print("a+b =", add_str(aS, bS))
    print("a-b =", sub_str(aS, bS))
    print("a*b =", mul_str(aS, bS))
    print("a//b=", div_str(aS, bS))
    print("sqrt(a) =", sqrt_str(aS))

    big_pow_result = pow_str(aS, "50")
    print("a^50 =>", big_pow_result[:200], "...(truncated)")

    # 2) Fraction-like input => shows the * 2^N approach
    cS = "12345.6789"
    sqrt_c = sqrt_str(cS)
    print("sqrt(12345.6789) =>", sqrt_c)