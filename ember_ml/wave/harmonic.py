"""
Harmonic wave processing components.
"""

import math
from typing import List, Dict

from ember_ml import ops, stats
from ember_ml import tensor  # For tensor.EmberTensor, tensor.zeros etc.
from ember_ml.types import TensorLike  # For type hinting


class FrequencyAnalyzer:
    """Analyzer for frequency components of signals."""
    
    def __init__(self, sampling_rate: float, window_size: int = 1024, overlap: float = 0.5):
        """
        Initialize frequency analyzer.

        Args:
            sampling_rate: Sampling rate in Hz
            window_size: FFT window size
            overlap: Window overlap ratio
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
    def compute_spectrum(self, signal: tensor.EmberTensor) -> TensorLike: # Return backend tensor
        """
        Compute frequency spectrum.

        Args:
            signal: Input signal tensor (EmberTensor wrapper)

        Returns:
            Frequency spectrum (raw backend tensor)
        """
        # Apply Hann window
        # ops.signal.hann_window returns a backend tensor
        window = ops.signal.hann_window(self.window_size, dtype=tensor.EmberDType.float32, device=signal.device)

        # Calculate padding amount
        # signal.shape is a tuple, get last dim
        padding_amount = self.window_size - (tensor.shape(signal)[-1] % self.window_size)
        if padding_amount == self.window_size: # if signal length is a multiple of window_size
             padding_amount = 0
        
        # F.pad equivalent: tensor.pad. PyTorch F.pad(signal, (pad_left, pad_right))
        # tensor.pad expects paddings as [[before, after], [before, after], ...]
        # For 1D signal, it would be [[0, padding_amount]]
        # Ensure signal is at least 1D for tensor.pad
        if len(tensor.shape(signal)) == 0: # scalar
            padded_signal = tensor.pad(tensor.reshape(signal, (1,)), [[0, padding_amount]])
        else:
            paddings = [[0,0]] * (len(tensor.shape(signal)) -1) + [[0, padding_amount]]
            padded_signal = tensor.pad(signal, paddings)

        # Compute FFT
        # Assuming window is compatible for multiplication (broadcast or same shape)
        # ops.multiply might be needed if '*' is not overloaded for EmberTensor with backend tensor
        product = ops.multiply(padded_signal, window)
        spectrum_complex = ops.fft.rfft(product)
        return ops.abs(spectrum_complex)
        
    def find_peaks(self, signal: tensor.EmberTensor, threshold: float = 0.1, tolerance: float = 0.01) -> List[Dict[str, float]]:
        """
        Find peak frequencies.

        Args:
            signal: Input signal tensor
            threshold: Peak detection threshold
            tolerance: Frequency matching tolerance

        Returns:
            List of peak information dictionaries
        """
        spectrum = self.compute_spectrum(signal)
        # Assuming self.window_size is correct for rfftfreq and spectrum length
        # And assuming 1/self.sampling_rate is correct for 'd' parameter
        # spectrum is a backend tensor. tensor.shape works on backend tensors too via common.shape
        n_for_rfftfreq = tensor.shape(spectrum)[-1] * 2 - 2 if tensor.shape(spectrum)[-1] > 1 else self.window_size
        freqs = ops.fft.rfftfreq(n_for_rfftfreq, d=1.0/self.sampling_rate, device=ops.get_device_of_tensor(spectrum)) # Get device from backend tensor

        # Find peaks
        peaks = []
        spectrum_max = stats.max(spectrum) # spectrum_max is a backend tensor (scalar)

        for i in range(1, tensor.shape(spectrum)[-1]-1): # Iterate over the last dimension
            # spectrum and freqs are backend tensors. Indexing them returns backend tensors.
            current_val = spectrum[i]
            prev_val = spectrum[i-1]
            next_val = spectrum[i+1]

            # ops.greater and ops.multiply work on backend tensors
            if tensor.item(ops.greater(current_val, prev_val)) and \
               tensor.item(ops.greater(current_val, next_val)): # Convert boolean tensor to Python bool
                if tensor.item(ops.greater(current_val, ops.multiply(threshold, spectrum_max))):
                    freq = freqs[i]
                    rounded_freq = round(tensor.item(freq)) # Use tensor.item()
                    peaks.append({
                        'frequency': rounded_freq,
                        'amplitude': tensor.item(ops.divide(current_val, spectrum_max)) # Use tensor.item()
                    })
        
        # Merge peaks within tolerance
        merged_peaks = {}
        for peak in peaks:
            freq = peak['frequency']
            amp = peak['amplitude']
            found = False
            for existing_freq in list(merged_peaks.keys()):
                if abs(freq - existing_freq) <= tolerance:
                    if amp > merged_peaks[existing_freq]['amplitude']:
                        merged_peaks[existing_freq] = peak
                    found = True
                    break
            if not found:
                merged_peaks[freq] = peak
                
        return sorted(merged_peaks.values(), key=lambda x: x['amplitude'], reverse=True)
        
    def harmonic_ratio(self, signal: tensor.convert_to_tensor) -> float:
        """
        Compute harmonic to noise ratio.

        Args:
            signal: Input signal tensor

        Returns:
            Harmonic ratio value
        """
        spectrum = self.compute_spectrum(signal)
        peaks = self.find_peaks(signal)
        
        if not peaks:
            return 0.0
            
        # Sum peak amplitudes
        peak_sum = sum(p['amplitude'] for p in peaks)
        # Assuming spectrum is 1D for this logic
        total_sum = stats.sum(spectrum) # total_sum is a backend tensor (scalar)
        
        # ops.greater returns backend boolean tensor, convert to Python bool with tensor.item()
        return tensor.item(ops.divide(peak_sum, tensor.item(total_sum))) if tensor.item(ops.greater(total_sum, 0.0)) else 0.0


class WaveSynthesizer:
    """Synthesizer for harmonic wave generation."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize wave synthesizer.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def sine_wave(self,
                 frequency: float,
                 duration: float,
                 amplitude: float = 1.0,
                 phase: float = 0.0) -> TensorLike: # Return backend tensor
        """
        Generate sine wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude
            phase: Initial phase in radians

        Returns:
            Sine wave (raw backend tensor)
        """
        num_samples = int(duration * self.sampling_rate)
        # tensor.linspace returns backend tensor
        t = tensor.linspace(0, duration, num_samples, dtype=tensor.EmberDType.float32)

        term1 = ops.multiply(t, 2 * math.pi * frequency) # ops.* returns backend tensor
        term2 = ops.add(term1, phase)
        sin_wave = ops.sin(term2)
        return ops.multiply(amplitude, sin_wave) # Returns backend tensor
        
    def harmonic_wave(self,
                     frequencies: List[float],
                     amplitudes: List[float],
                     duration: float) -> TensorLike: # Return backend tensor
        """
        Generate wave with harmonics.

        Args:
            frequencies: List of frequencies
            amplitudes: List of amplitudes
            duration: Signal duration in seconds

        Returns:
            Harmonic wave (raw backend tensor)
        """
        total_amp = sum(abs(a) for a in amplitudes)
        if total_amp > 0:
            norm_amplitudes = [a / total_amp for a in amplitudes]
        else:
            norm_amplitudes = amplitudes
            
        num_samples = int(duration * self.sampling_rate)
        # tensor.zeros returns backend tensor
        wave = tensor.zeros((num_samples,), dtype=tensor.EmberDType.float32)

        for freq, amp in zip(frequencies, norm_amplitudes):
            # self.sine_wave now returns backend tensor
            sine_component = self.sine_wave(freq, duration, amp)
            wave = ops.add(wave, sine_component) # ops.add returns backend tensor
            
        return wave # Returns backend tensor
        
    def apply_envelope(self,
                      wave: TensorLike, # Expect backend tensor
                      envelope: TensorLike) -> TensorLike: # Return backend tensor
        """
        Apply amplitude envelope.

        Args:
            wave: Input wave (raw backend tensor)
            envelope: Amplitude envelope (raw backend tensor)

        Returns:
            Modulated wave (raw backend tensor)
        """
        # tensor.shape works on backend tensors
        wave_len = tensor.shape(wave)[-1]
        env_len = tensor.shape(envelope)[-1]

        processed_envelope = envelope # backend tensor
        if env_len != wave_len:
            # tensor.reshape returns backend tensor
            envelope_reshaped = tensor.reshape(envelope, (1, 1, env_len))
            try:
                # ops.image.resize returns backend tensor
                resized_envelope_squeezable = ops.image.resize(envelope_reshaped,
                                                              size=(1, wave_len),
                                                              mode='linear')
                # tensor.squeeze returns backend tensor
                processed_envelope = tensor.squeeze(resized_envelope_squeezable)
            except AttributeError:
                print("Warning: ops.image.resize with linear mode for 1D not found. Envelope may not be applied correctly.")

        abs_envelope = ops.abs(processed_envelope) # backend tensor
        max_abs_env = stats.max(abs_envelope)   # backend tensor (scalar)

        # tensor.item(ops.greater(...)) for Python bool condition
        if tensor.item(ops.greater(max_abs_env, 0.0)):
            normalized_envelope = ops.divide(abs_envelope, max_abs_env)
        else:
            normalized_envelope = abs_envelope
        
        modulated = ops.multiply(wave, normalized_envelope)
        
        signed_wave = ops.sign(wave)
        abs_modulated = ops.abs(modulated)
        abs_wave = ops.abs(wave)
        min_abs = ops.minimum(abs_modulated, abs_wave)
        final_modulated = ops.multiply(signed_wave, min_abs)
            
        return final_modulated # Returns backend tensor

class HarmonicProcessor:
    """Processor for harmonic signal analysis and manipulation."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize harmonic processor.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.analyzer = FrequencyAnalyzer(sampling_rate)
        self.synthesizer = WaveSynthesizer(sampling_rate)
        
    def decompose(self, signal: tensor.EmberTensor) -> Dict[str, List[float]]:
        """
        Decompose signal into harmonic components.

        Args:
            signal: Input signal tensor

        Returns:
            Dictionary of frequency components
        """
        peaks = self.analyzer.find_peaks(signal)
        return {
            'frequencies': [p['frequency'] for p in peaks],
            'amplitudes': [p['amplitude'] for p in peaks]
        }
        
    def reconstruct(self,
                   frequencies: List[float],
                   amplitudes: List[float],
                   duration: float) -> TensorLike: # Return backend tensor
        """
        Reconstruct signal from components.

        Args:
            frequencies: List of frequencies
            amplitudes: List of amplitudes
            duration: Signal duration in seconds

        Returns:
            Reconstructed signal (raw backend tensor)
        """
        freq_amp = sorted(zip(frequencies, amplitudes), key=lambda x: x[0])
        frequencies = [f for f, _ in freq_amp]
        amplitudes = [a for _, a in freq_amp]
        
        # self.synthesizer.harmonic_wave returns backend tensor
        signal = self.synthesizer.harmonic_wave(frequencies, amplitudes, duration)
        
        max_amp = max(abs(a) for a in amplitudes) if amplitudes else 1.0
        if max_amp > 0:
            # ops.multiply with scalar and backend tensor
            signal = ops.multiply(signal, max_amp)
            
        return signal # Returns backend tensor
        
    def filter_harmonics(self,
                        signal: TensorLike, # Expect backend tensor
                        keep_frequencies: List[float],
                        tolerance: float = 0.1) -> TensorLike: # Return backend tensor
        """
        Filter specific harmonics.

        Args:
            signal: Input signal (raw backend tensor)
            keep_frequencies: Frequencies to keep
            tolerance: Frequency matching tolerance

        Returns:
            Filtered signal (raw backend tensor)
        """
        # ops.abs and stats.max operate on and return backend tensors
        original_max = stats.max(ops.abs(signal)) # original_max is backend scalar tensor
        
        duration = tensor.shape(signal)[-1] / self.sampling_rate
        # tensor.zeros_like returns backend tensor
        filtered_signal = tensor.zeros_like(signal)
        
        if keep_frequencies:
            for freq in keep_frequencies:
                # self.synthesizer.sine_wave returns backend tensor
                # tensor.item(original_max) to use scalar float value
                sine_component = self.synthesizer.sine_wave(freq, duration, tensor.item(original_max))
                filtered_signal = ops.add(filtered_signal, sine_component)
            
            filtered_signal = ops.divide(filtered_signal, float(len(keep_frequencies)))
            
        return filtered_signal # Returns backend tensor
