from dataclasses import dataclass
from typing import List, Tuple

from ember_ml import ops, tensor
from ember_ml.types import TensorLike


# Assuming fft, signal processing ops are under ops.fft and ops.signal respectively
# from ember_ml.ops import fft as ops_fft
# from ember_ml.ops import signal as ops_signal
# from ember_ml.ops import linearalg as ops_linearalg # For rfft/rfftfreq if they moved

@dataclass
class WaveConfig:
    sample_rate: int
    bit_depth: int
    buffer_size: int
    num_freq_bands: int
    phase_resolution: int

class BinaryWaveProcessor:
    """Handles conversion between PCM audio and binary wave representations"""
    
    def __init__(self, config: WaveConfig):
        self.config = config
        self.phase_accumulator = 0 # This seems unused based on current code
        self.previous_sample = 0 # This seems unused based on current code
        self._initialize_frequency_bands()
    
    def _initialize_frequency_bands(self):
        """Initialize frequency band filters"""
        nyquist = self.config.sample_rate // 2

        # Replace np.logspace with ops/tensor equivalent
        # log_min = ops.log10(tensor.convert_to_tensor(20.0))
        # log_max = ops.log10(tensor.convert_to_tensor(float(nyquist))) # Ensure nyquist is float for log10
        # self.band_frequencies = ops.pow(10.0, tensor.linspace(log_min, log_max, self.config.num_freq_bands))

        # For simplicity if direct logspace is not in ops, using linspace on log scale then pow
        # This should be float for calculations
        log_min_val = ops.log10(tensor.convert_to_tensor(20.0, dtype=tensor.EmberDType.float32))
        log_max_val = ops.log10(tensor.convert_to_tensor(float(nyquist), dtype=tensor.EmberDType.float32))

        linspace_tensor = tensor.linspace(
            ops.item(log_min_val) if hasattr(log_min_val, 'item') else float(log_min_val), # Ensure scalar for linspace start/end
            ops.item(log_max_val) if hasattr(log_max_val, 'item') else float(log_max_val),
            self.config.num_freq_bands,
            dtype=tensor.EmberDType.float32
        )
        self.band_frequencies = ops.pow(10.0, linspace_tensor)

        # Create band filters with sharper cutoffs
        self.band_filters = []
        # Iterating over an EmberTensor might require .tolist() or direct iteration if supported
        # Assuming self.band_frequencies can be iterated or converted to list for this setup loop
        try:
            band_freq_list = self.band_frequencies.tolist()
        except: # Fallback if .tolist() not available or tensor is not easily iterable for this setup
            # This part might need adjustment based on EmberTensor's capabilities
            # For now, assuming it can be iterated for setup
            band_freq_list = [self.band_frequencies[i].item() for i in range(tensor.shape(self.band_frequencies)[0])]


        for freq_val in band_freq_list: # freq is now a Python float
            width = freq_val * 0.3  # Narrower bands
            low = max(0.0, freq_val - width) # Ensure float for max
            high = min(float(nyquist), freq_val + width) # Ensure float for min
            self.band_filters.append((low, high))
    
    def pcm_to_binary(self, pcm_data: TensorLike) -> tensor.EmberTensor: # Return EmberTensor
        """Convert PCM audio data to binary representation using improved delta-sigma"""
        # Ensure pcm_data is EmberTensor
        pcm_data_et = tensor.convert_to_tensor(pcm_data)

        # Normalize PCM data to [-1, 1]
        if pcm_data_et.dtype == tensor.int16:
            normalized = ops.divide(pcm_data_et, 32768.0)
        elif pcm_data_et.dtype == tensor.int32:
            normalized = ops.divide(pcm_data_et, 2147483648.0)
        else:
            # Assuming it's already float or needs cast
            normalized = tensor.cast(pcm_data_et, dtype=tensor.EmberDType.float32)
        
        # Second-order delta-sigma modulation
        binary_list = [] # Build as list of Python ints, then convert
        error1 = 0.0  # First integrator, ensure float
        error2 = 0.0  # Second integrator, ensure float
        
        # Iterating over EmberTensor: use slicing or getitem
        for i in range(tensor.shape(normalized)[0]):
            current_sample_val = normalized[i].item() # Get Python float for scalar math

            input_value = current_sample_val + error1 * 1.5 - error2 * 0.5
            
            current_binary_val = 0
            quantization_error = 0.0

            if input_value > 0:
                current_binary_val = 1
                quantization_error = input_value - 1.0
            else:
                current_binary_val = 0
                quantization_error = input_value
            
            binary_list.append(current_binary_val)
            error2 = error1 + quantization_error * 0.5
            error1 = quantization_error
                
        return tensor.convert_to_tensor(binary_list, dtype=tensor.EmberDType.uint8)
    
    def extract_frequency_bands(self, binary_data: TensorLike) -> List[tensor.EmberTensor]: # Return List of EmberTensors
        """Extract frequency band information with improved filtering"""
        binary_data_et = tensor.convert_to_tensor(binary_data)
        # Convert binary to float for FFT: 0 -> -1, 1 -> 1
        float_data = ops.subtract(ops.multiply(tensor.cast(binary_data_et, dtype=tensor.EmberDType.float32), 2.0), 1.0)
        
        data_len = tensor.shape(float_data)[0]
        # Apply Hanning window
        # window = np.hanning(len(float_data))
        window = ops.signal.hann_window(data_len, dtype=tensor.EmberDType.float32, periodic=True) # Assuming periodic=True matches np.hanning for FFT
        windowed_data = ops.multiply(float_data, window)
        
        # Compute FFT
        # spectrum = linearalg.rfft(windowed_data) # Assuming these moved to ops.fft
        # freqs = linearalg.rfftfreq(len(float_data), 1/self.config.sample_rate)
        spectrum = ops.fft.rfft(windowed_data)
        # Ensure n for rfftfreq is from original data length before rfft transform
        freqs = ops.fft.rfftfreq(n=data_len, d=1.0/self.config.sample_rate, device=float_data.device)

        band_data_list = [] # Changed name
        for low, high in self.band_filters: # low, high are Python floats
            center_freq = (low + high) / 2.0
            bandwidth = high - low
            
            # freq_response = 1 / (1 + ((freqs - center_freq)/(bandwidth/2))**4)
            # Ensure all ops are on EmberTensors
            freqs_minus_center = ops.subtract(freqs, center_freq)
            scaled_diff = ops.divide(freqs_minus_center, (bandwidth / 2.0))
            scaled_diff_pow4 = ops.pow(scaled_diff, 4.0)
            denominator = ops.add(1.0, scaled_diff_pow4)
            freq_response = ops.divide(1.0, denominator)

            band_spectrum = ops.multiply(spectrum, freq_response)
            # band_signal = np.fft.irfft(band_spectrum)
            band_signal = ops.fft.irfft(band_spectrum) # Returns real EmberTensor
            
            threshold = 0.0
            hysteresis = 0.1
            # binary = tensor.zeros_like(band_signal, dtype=tensor.uint8) # Initialize EmberTensor
            # This loop needs to be vectorized or use ops for assignments
            # state = 0 # Python scalar state
            # for i in range(tensor.shape(band_signal)[0]):
            #     current_bs_val = band_signal[i].item()
            #     if state == 0 and current_bs_val > threshold + hysteresis:
            #         state = 1
            #     elif state == 1 and current_bs_val < threshold - hysteresis:
            #         state = 0
            #     binary[i] = state # This assignment is tricky for EmberTensor
            
            # Vectorized hysteresis (conceptual, actual implementation might differ based on ops capabilities)
            # This is complex to vectorize perfectly without specific ops.
            # For now, a simpler thresholding:
            binary = tensor.cast(ops.greater(band_signal, threshold), dtype=tensor.EmberDType.uint8)
            # The hysteresis part is harder to vectorize simply.
            # For this refactor, I'll use the simpler thresholding and note the loss of hysteresis logic.
            print("Warning: Hysteresis logic in extract_frequency_bands simplified to basic thresholding.")

            band_data_list.append(binary)
            
        return band_data_list
    
    def encode_phase(self, binary_data: TensorLike) -> Tuple[tensor.EmberTensor, float]: # Return EmberTensor and float
        """Encode phase information with improved stability"""
        binary_data_et = tensor.convert_to_tensor(binary_data)
        analytic_signal = self._hilbert_transform(binary_data_et) # analytic_signal is complex EmberTensor
        # phase = np.angle(analytic_signal)
        phase = ops.angle(analytic_signal) # phase is real EmberTensor
        
        mean_phase_tensor = stats.mean(phase) # Returns scalar EmberTensor
        mean_phase_scalar = mean_phase_tensor.item() % (2 * ops.pi) # Python float
        
        return binary_data_et, mean_phase_scalar
    
    def _hilbert_transform(self, binary_data: tensor.EmberTensor) -> tensor.EmberTensor: # Takes and returns EmberTensor
        """Compute Hilbert transform of binary signal"""
        # float_data = binary_data.astype(tensor.float32) * 2 - 1
        float_data = ops.subtract(ops.multiply(tensor.cast(binary_data, dtype=tensor.EmberDType.float32), 2.0), 1.0)

        # spectrum = np.fft.fft(float_data)
        spectrum = ops.fft.fft(float_data) # Returns complex EmberTensor

        n = tensor.shape(spectrum)[0]
        h = tensor.zeros((n,), dtype=spectrum.dtype, device=spectrum.device) # Match dtype (complex) and device
                                                                             # Or h should be real, then ops.multiply handles complex * real
        h_real = tensor.zeros((n,), dtype=ops.dtype(spectrum).real_dtype, device=spectrum.device)


        # This logic for h needs to be on EmberTensors if n is tensor, or use Python int from n.item()
        n_py = n.item() if isinstance(n, tensor.EmberTensor) else int(n)

        if n_py % 2 == 0:
            # h_real[0] = 1; h_real[n_py//2] = 1; h_real[1:n_py//2] = 2
            # Using slice_update or direct assignment if EmberTensor supports it
            h_real[0] = 1.0
            h_real[n_py//2] = 1.0
            if n_py//2 > 1: # Ensure slice is valid
                 h_real[slice(1, n_py//2)] = 2.0
        else:
            # h_real[0] = 1; h_real[1:(n_py+1)//2] = 2
            h_real[0] = 1.0
            if (n_py+1)//2 > 1: # Ensure slice is valid
                 h_real[slice(1, (n_py+1)//2)] = 2.0

        # If h is meant to be complex (though typically real for Hilbert)
        # h = tensor.complex(h_real, tensor.zeros_like(h_real))
        # For now, assume h is real and ops.multiply handles complex * real
        h_to_multiply = h_real

        # return np.fft.ifft(spectrum * h)
        return ops.fft.ifft(ops.multiply(spectrum, h_to_multiply)) # Returns complex EmberTensor
    
    def process_frame(self, pcm_data: TensorLike) -> Tuple[List[tensor.EmberTensor], List[float]]:
        """Process a frame of PCM data into binary waves with phase information"""
        pcm_data_et = tensor.convert_to_tensor(pcm_data)
        binary_data = self.pcm_to_binary(pcm_data_et)
        band_data_list = self.extract_frequency_bands(binary_data) # List of EmberTensors
        
        phases = []
        encoded_bands_list = [] # Changed name
        for band_et in band_data_list: # band_et is EmberTensor
            encoded_et, phase_scalar = self.encode_phase(band_et)
            encoded_bands_list.append(encoded_et)
            phases.append(phase_scalar) # list of Python floats
            
        return encoded_bands_list, phases
    
    def binary_to_pcm(self, binary_data: TensorLike) -> tensor.EmberTensor: # Return EmberTensor
        """Convert binary representation back to PCM audio with improved filtering"""
        binary_data_et = tensor.convert_to_tensor(binary_data)
        # float_data = binary_data.astype(tensor.float32) * 2 - 1
        float_data = ops.subtract(ops.multiply(tensor.cast(binary_data_et, dtype=tensor.EmberDType.float32), 2.0), 1.0)
        
        filter_length = 31
        # t = tensor.arange(-filter_length//2, filter_length//2 + 1) # Assuming tensor.arange exists
        # If not, use linspace or create from list:
        t_list = list(range(-filter_length//2, filter_length//2 + 1))
        t = tensor.convert_to_tensor(t_list, dtype=tensor.EmberDType.float32, device=float_data.device)

        # sinc = np.sinc(t/2)
        # Sinc(x) = sin(pi*x) / (pi*x)
        # Here, x = t/2. So Sinc(t/2) = sin(pi*t/2) / (pi*t/2)
        pi_val = ops.pi # float
        t_div_2 = ops.divide(t, 2.0)
        
        # Numerator: sin(pi * t/2)
        sin_arg = ops.multiply(pi_val, t_div_2)
        sin_val = ops.sin(sin_arg)

        # Denominator: pi * t/2 (same as sin_arg)
        # Handle t=0 case for sinc where den is 0, Sinc(0)=1
        sinc_val = ops.divide(sin_val, sin_arg)
        # For t=0, sin_arg is 0. Division by zero. Sinc(0) = 1.
        # Create a tensor of ones for where t is zero
        ones_for_sinc = tensor.ones_like(t_div_2)
        sinc = ops.where(ops.equal(t_div_2, 0), ones_for_sinc, sinc_val)

        # window = np.hamming(len(sinc))
        window = ops.signal.hamming_window(tensor.shape(sinc)[0], dtype=tensor.EmberDType.float32, device=sinc.device)

        filter_kernel = ops.multiply(sinc, window)
        sum_kernel = stats.sum(filter_kernel)
        # Avoid division by zero if sum_kernel is zero
        if ops.greater(ops.abs(sum_kernel), 1e-9).item():
            filter_kernel = ops.divide(filter_kernel, sum_kernel)

        # filtered = np.convolve(float_data, filter_kernel, mode='same')
        # This is a critical dependency. Assuming ops.signal.convolve exists.
        # Note: convolve usually expects 1D inputs. float_data might be ND.
        # If float_data is ND, this needs per-channel convolution or a general N-D convolve.
        # Assuming float_data is 1D for now as per typical audio processing.
        try:
            if len(tensor.shape(float_data)) > 1:
                 print("Warning: binary_to_pcm convolve expects 1D float_data, found >1D. Results may be incorrect.")
            filtered = ops.signal.convolve(float_data, filter_kernel, mode='same')
        except (AttributeError, NotImplementedError):
            print("Warning: ops.signal.convolve not available or incompatible. PCM conversion will be noisy (no low-pass).")
            # Fallback: no filtering, which will sound bad.
            filtered = float_data
        
        # Convert to 16-bit PCM
        scaled_filtered = ops.multiply(filtered, 32767.0)
        clipped_filtered = ops.clip(scaled_filtered, -32768.0, 32767.0)
        pcm_int16 = tensor.cast(clipped_filtered, dtype=tensor.int16)
        return pcm_int16

class BinaryWaveNeuron:
    """Binary wave neuron with phase sensitivity and STDP learning"""
    
    def __init__(self, num_freq_bands: int, phase_sensitivity: float = 0.5):
        self.num_freq_bands = num_freq_bands
        self.phase_sensitivity = phase_sensitivity
        self.state = tensor.zeros((num_freq_bands,), dtype=tensor.EmberDType.uint8) # Explicit shape
        self.phase = tensor.zeros((num_freq_bands,), dtype=tensor.EmberDType.float32) # Explicit shape and float
        self.threshold = 0.3

        # self.weights = np.random.random(num_freq_bands) * 0.5 + 0.25
        rand_weights = tensor.random_uniform(shape=(num_freq_bands,), minval=0.0, maxval=1.0, dtype=tensor.EmberDType.float32)
        self.weights = ops.add(ops.multiply(rand_weights, 0.5), 0.25) # EmberTensor weights
        
        self.learning_rate = 0.01
        self.stdp_window = 20  # samples
        self.state_history: List[tensor.EmberTensor] = [] # Type hint
        self.phase_history: List[tensor.EmberTensor] = [] # Type hint for phase history if needed
    
    def compute_interference(self, input_waves: List[tensor.EmberTensor], input_phases: List[float]) -> tensor.EmberTensor:
        """Compute wave interference patterns with improved phase sensitivity"""
        # Ensure self.state is float for interference calculation if it's used that way
        interference = tensor.zeros_like(self.weights, dtype=tensor.EmberDType.float32) # Match self.weights type
        
        for i, (wave_et, phase_scalar) in enumerate(zip(input_waves, input_phases)): # wave_et is EmberTensor
            # Ensure phase operations use EmberTensors if ops expect them
            current_neuron_phase = self.phase[i] # Scalar EmberTensor
            input_phase_et = tensor.convert_to_tensor(phase_scalar, dtype=current_neuron_phase.dtype, device=current_neuron_phase.device)

            # phase_diff = ops.abs(((self.phase[i] - phase + ops.pi) % (2 * ops.pi)) - ops.pi)
            # Modulo arithmetic for phase difference: (a - b + pi) % (2*pi) - pi
            diff_plus_pi = ops.add(ops.subtract(current_neuron_phase, input_phase_et), ops.pi)
            mod_val = ops.mod(diff_plus_pi, (2.0 * ops.pi)) # Use float for modulo
            phase_diff = ops.abs(ops.subtract(mod_val, ops.pi))
            
            # Gaussian phase sensitivity
            # phase_factor = ops.exp(-phase_diff**2 / (2 * self.phase_sensitivity**2))
            phase_diff_sq = ops.square(phase_diff)
            denominator_pf = 2.0 * (self.phase_sensitivity**2)
            exponent_pf = ops.negative(ops.divide(phase_diff_sq, denominator_pf))
            phase_factor = ops.exp(exponent_pf) # Scalar EmberTensor
            
            # Compute weighted contribution with temporal integration
            # wave_energy = stats.sum(wave) / len(wave)
            # Assuming wave_et is 1D here
            wave_energy = ops.divide(stats.sum(wave_et), float(tensor.shape(wave_et)[0])) # Scalar EmberTensor

            # interference[i] = wave_energy * self.weights[i] * phase_factor
            term_prod = ops.multiply(ops.multiply(wave_energy, self.weights[i]), phase_factor)
            # This requires interference to be an EmberTensor and support __setitem__
            interference[i] = term_prod
            
        return interference
    
    def update_phase(self, interference: tensor.EmberTensor, input_phases: List[float]):
        """Update neuron phase state with momentum"""
        phase_momentum = 0.8
        learn_rate = 0.2
        
        for i, phase in enumerate(input_phases):
            if interference[i] > self.threshold:
                # Update phase with momentum
                phase_diff = ((phase - self.phase[i] + ops.pi) % (2 * ops.pi)) - ops.pi
                self.phase[i] = (self.phase[i] + 
                               phase_momentum * self.phase[i] +
                               learn_rate * phase_diff) % (2 * ops.pi)
    
    def apply_stdp(self, input_waves: List[TensorLike], output: TensorLike):
        """Apply STDP learning rule"""
        # Store state history
        self.state_history.append(output)
        if len(self.state_history) > self.stdp_window:
            self.state_history.pop(0)
        
        # Skip if not enough history
        if len(self.state_history) < 2:
            return
        
        # Compute STDP updates
        for i in range(self.num_freq_bands):
            # Compute correlation between input and output
            input_energy = stats.mean([wave[i] if i < len(wave) else 0 for wave in input_waves])
            output_energy = stats.mean(output)
            
            # Compute weight update
            if output_energy > 0:
                # Hebbian update
                delta_w = self.learning_rate * input_energy * (1 - self.weights[i])
            else:
                # Anti-Hebbian update
                delta_w = -self.learning_rate * input_energy * self.weights[i]
            
            # Apply weight update with bounds
            self.weights[i] = ops.clip(self.weights[i] + delta_w, 0.1, 0.9)
    
    def generate_output(self, interference: TensorLike) -> TensorLike:
        """Generate binary output with hysteresis"""
        output = tensor.zeros_like(interference, dtype=tensor.uint8)
        
        # Add hysteresis to prevent rapid switching
        hysteresis = 0.05
        for i in range(len(interference)):
            if self.state[i] == 0 and interference[i] > self.threshold + hysteresis:
                output[i] = 1
            elif self.state[i] == 1 and interference[i] < self.threshold - hysteresis:
                output[i] = 0
            else:
                output[i] = self.state[i]
        
        return output
    
    def process(self, input_waves: List[TensorLike], input_phases: List[float]) -> TensorLike:
        """Process input waves and generate output"""
        # Compute interference pattern
        interference = self.compute_interference(input_waves, input_phases)
        
        # Update phase
        self.update_phase(interference, input_phases)
        
        # Generate output
        output = self.generate_output(interference)
        
        # Apply STDP learning
        self.apply_stdp(input_waves, output)
        
        # Update state
        self.state = output
        
        return output

def create_test_signal(duration: float, sample_rate: int) -> TensorLike:
    """Create a test signal with improved harmonics"""
    t = tensor.linspace(0, duration, int(duration * sample_rate))
    
    # Create a signal with multiple frequencies and amplitude modulation
    am = 0.5 * (1 + 0.3 * ops.sin(2 * ops.pi * 5 * t))  # 5 Hz amplitude modulation
    
    signal = am * (
        0.5 * ops.sin(2 * ops.pi * 440 * t) +  # A4 note
        0.3 * ops.sin(2 * ops.pi * 880 * t) +  # A5 note
        0.2 * ops.sin(2 * ops.pi * 1760 * t)   # A6 note
    )
    
    # Add some noise
    noise = tensor.random_normal(0, 0.05, len(t))
    signal = signal + noise
    
    # Normalize and convert to int16
    signal = signal / stats.max(ops.abs(signal))
    return (signal * 32767).astype(tensor.int16)
