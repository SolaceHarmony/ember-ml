"""
Quantum wave processing components.
"""

import math
# import cmath # Not directly used for tensor ops, math.pi is fine
from typing import List, Dict, Optional, Tuple, Union

from ember_ml import ops, stats
from ember_ml import tensor # For tensor.EmberTensor, tensor.zeros etc.
from ember_ml.types import TensorLike # For type hinting
from ember_ml.nn.modules import Module, Parameter # For Module and Parameter
from ember_ml.nn.layers import Linear # For Linear layer
# Assuming stats and linearalg are accessible via ops
# from ember_ml.ops import stats, linearalg


# Helper to create complex number if not directly supported by multiplying float with 1j
def _complex_scalar_multiply(scalar_tensor: TensorLike, complex_py_num: complex) -> TensorLike: # Takes and returns backend tensor
    real_part = ops.multiply(scalar_tensor, complex_py_num.real)
    imag_part = ops.multiply(scalar_tensor, complex_py_num.imag)
    return tensor.complex(real_part, imag_part)

class WaveFunction:
    """Quantum wave function representation. Holds backend tensors."""
    
    def __init__(self, amplitudes: TensorLike, phases: TensorLike):
        """
        Initialize wave function.

        Args:
            amplitudes: Probability amplitudes (real backend tensor)
            phases: Phase angles (real backend tensor)
        """
        # Ensure inputs are backend tensors
        self.amplitudes = amplitudes.to_backend_tensor() if isinstance(amplitudes, tensor.EmberTensor) else amplitudes
        self.phases = phases.to_backend_tensor() if isinstance(phases, tensor.EmberTensor) else phases
        
    def to_complex(self) -> TensorLike: # Returns backend tensor
        """
        Convert to complex representation: A * exp(j * phi).

        Returns:
            Complex backend tensor representation
        """
        cos_phi = ops.cos(self.phases)
        sin_phi = ops.sin(self.phases)
        exp_j_phi = tensor.complex(cos_phi, sin_phi) # tensor.complex returns backend tensor

        return ops.multiply(self.amplitudes, exp_j_phi) # ops.multiply returns backend tensor
        
    def probability_density(self) -> TensorLike: # Returns backend tensor
        """
        Compute probability density.

        Returns:
            Probability density (raw backend tensor)
        """
        return ops.square(self.amplitudes) # ops.square returns backend tensor
        
    def normalize(self) -> 'WaveFunction':
        """
        Normalize wave function. Operates on and returns WaveFunction with backend tensors.

        Returns:
            Normalized wave function
        """
        prob_sum = stats.sum(self.probability_density()) # backend tensor (scalar)
        norm = ops.sqrt(prob_sum) # backend tensor (scalar)
        
        # tensor.item for condition
        if tensor.item(ops.greater(norm, 1e-12)):
            normalized_amplitudes = ops.divide(self.amplitudes, norm) # backend tensor
        else:
            normalized_amplitudes = self.amplitudes
        # Creates new WaveFunction with backend tensors
        return WaveFunction(normalized_amplitudes, self.phases)

    def evolve(self, hamiltonian: TensorLike, dt: float) -> 'WaveFunction': # hamiltonian is backend tensor
        """
        Time evolution under Hamiltonian.

        Args:
            hamiltonian: Hamiltonian operator (complex tensor)
            dt: Time step (float)

        Returns:
            Evolved wave function
        """
        psi_complex = self.to_complex() # Complex tensor
        
        # -1j * hamiltonian * dt
        # Create complex scalar -1j
        neg_j_dt = tensor.complex(tensor.zeros_like(dt), tensor.full_like(dt, -1.0 * dt)) # Assuming dt is scalar tensor or float
                                                                                     # If dt is float, need to make it tensor first
        if not isinstance(dt, tensor.EmberTensor):
            dt_tensor = tensor.convert_to_tensor(dt, device=hamiltonian.device, dtype=ops.dtype(hamiltonian).real_dtype if hasattr(ops.dtype(hamiltonian), 'real_dtype') else tensor.EmberDType.float32) # Match hamiltonian's float type
        else:
            dt_tensor = dt

        # -1j
        neg_j = tensor.complex(tensor.zeros_like(dt_tensor), tensor.full_like(dt_tensor, -1.0))


        term = ops.multiply(hamiltonian, dt_tensor) # Hamiltonian * dt
        exponent = ops.multiply(neg_j, term) # -j * Hamiltonian * dt

        U = ops.linearalg.expm(exponent) # Matrix exponential, assumes complex support

        # ops.matmul for complex tensors
        evolved_complex = ops.matmul(U, psi_complex)

        # Convert back to amplitude/phase
        # ops.abs for complex gives magnitude (real tensor)
        new_amplitudes = ops.abs(evolved_complex)
        # ops.angle for complex gives phase angle (real tensor)
        new_phases = ops.angle(evolved_complex)

        return WaveFunction(new_amplitudes, new_phases)

class QuantumState:
    """Quantum state with qubit operations."""
    
    def __init__(self, num_qubits: int, device: Optional[str] = None): # device is string now
        """
        Initialize quantum state.

        Args:
            num_qubits: Number of qubits
            device: Computation device (string, e.g., "cpu", "gpu")
        """
        self.num_qubits = num_qubits
        self.device = device if device is not None else ops.get_device() # Use global default if None
        self.dim = 2 ** num_qubits
        
        # Initialize to |0...0⟩ state
        self.amplitudes = tensor.zeros((self.dim,), dtype=tensor.complex64, device=self.device) # backend tensor
        
        # Replace direct item assignment with tensor_scatter_nd_update
        init_val_real = tensor.convert_to_tensor(1.0, device=self.device, dtype=tensor.EmberDType.float32)
        init_val_imag = tensor.convert_to_tensor(0.0, device=self.device, dtype=tensor.EmberDType.float32)
        init_val_complex = tensor.complex(init_val_real, init_val_imag) # backend tensor (scalar complex)

        indices_tensor = tensor.convert_to_tensor([[0]], dtype=tensor.EmberDType.int64, device=self.device)
        updates_tensor = tensor.reshape(init_val_complex, (1,)) # Reshape scalar to (1,) for update
        self.amplitudes = tensor.tensor_scatter_nd_update(self.amplitudes, indices_tensor, updates_tensor)

    def apply_gate(self, gate: TensorLike, qubits: List[int]): # gate is backend tensor
        """
        Apply quantum gate.

        Args:
            gate: Gate unitary matrix (complex backend tensor)
            qubits: Target qubit indices
        """
        # Construct full operator
        # Note: This part is algorithmically complex and hard to make efficient
        # without good sparse tensor tools or direct element-wise __setitem__ support in EmberTensor
        # that translates to efficient backend operations.
        op_real = tensor.zeros((self.dim, self.dim), dtype=ops.dtype(gate).real_dtype, device=self.device)
        op_imag = tensor.zeros((self.dim, self.dim), dtype=ops.dtype(gate).real_dtype, device=self.device)

        # Initialize op as identity matrix parts
        eye_real_diag = tensor.ones((self.dim,), dtype=ops.dtype(gate).real_dtype, device=self.device)
        op_real = ops.linearalg.diag_embed(eye_real_diag) # Creates diagonal matrix from vector

        # The loop is very slow. This is a placeholder for a more efficient sparse construction.
        # The following logic attempts to replicate the PyTorch index-based assignment.
        # This will be extremely slow if done element by element via scatter updates in a loop.
        # A better approach would be to use kronecker products and permutations if available in ops.
        # For this refactoring, I will highlight this as a major performance bottleneck / feasibility issue.
        print(f"Warning: QuantumState.apply_gate uses a slow element-wise construction loop. Needs optimization with sparse ops.")
        for i_int in range(self.dim):
            for j_int in range(self.dim):
                if all((i_int >> q) & 1 == (j_int >> q) & 1 for q in range(self.num_qubits) if q not in qubits):
                    idx_i = sum(((i_int >> q) & 1) << n for n, q in enumerate(qubits))
                    idx_j = sum(((j_int >> q) & 1) << n for n, q in enumerate(qubits))

                    # gate_val is complex EmberTensor. op_real/op_imag need real values.
                    gate_val = gate[idx_i, idx_j] # This itself returns an EmberTensor (complex scalar)

                    # Manually assign real and imaginary parts using tensor_scatter_nd_update
                    idx_to_update = tensor.convert_to_tensor([[i_int, j_int]], dtype=tensor.EmberDType.int64, device=self.device)

                    val_to_update_real = tensor.reshape(ops.real(gate_val), (1,)) # Ensure correct shape for update
                    op_real = tensor.tensor_scatter_nd_update(op_real, idx_to_update, val_to_update_real)

                    val_to_update_imag = tensor.reshape(ops.imag(gate_val), (1,)) # Ensure correct shape for update
                    op_imag = tensor.tensor_scatter_nd_update(op_imag, idx_to_update, val_to_update_imag)

        op = tensor.complex(op_real, op_imag) # op is backend tensor
                    
        # Apply operator
        self.amplitudes = ops.matmul(op, self.amplitudes) # self.amplitudes remains backend tensor
        
    def measure(self, qubit: int) -> Tuple[int, float]:
        """
        Measure single qubit.

        Args:
            qubit: Qubit index to measure

        Returns:
            Tuple of (measurement result, probability)
        """
        # Compute probabilities
        # probs should be a real tensor
        probs_real = tensor.zeros((2,), dtype=tensor.EmberDType.float32, device=self.device)

        # This loop is also potentially slow. Can be optimized with gather/scatter/sum_reduce if available.
        for i in range(self.dim):
            bit = (i >> qubit) & 1
            # self.amplitudes[i] is a backend complex scalar. ops.abs then ops.square give backend real scalar.
            prob_contrib = ops.square(ops.abs(self.amplitudes[i])) # backend scalar
            
            idx_to_update_probs = tensor.convert_to_tensor([[bit]], dtype=tensor.EmberDType.int64, device=self.device)
            current_prob_val = tensor.gather(probs_real, idx_to_update_probs, axis=0) # Gather based on bit
            updated_prob_val = ops.add(current_prob_val, prob_contrib)
            probs_real = tensor.tensor_scatter_nd_update(probs_real, idx_to_update_probs, updated_prob_val)

        # Sample result
        # probs_real[1] is a backend scalar. Use tensor.item() for p parameter.
        # tensor.random_bernoulli returns backend tensor.
        # Reshape probs_real[1] to be a valid shape for bernoulli if it expects non-scalar prob input.
        # For now, assume p can be scalar Python float.
        prob_of_one = tensor.item(probs_real[1])
        result_tensor = tensor.random_bernoulli(shape=(1,), p=prob_of_one, device=self.device) # Get (1,) shape
        result = int(tensor.item(result_tensor))
        
        # Project state
        prob_for_result = probs_real[result] # backend scalar
        norm_val = ops.sqrt(prob_for_result) # backend scalar

        if tensor.item(ops.greater(norm_val, 1e-12)):
            # Create temporary lists for real/imag parts to build the new state
            new_amp_re_list = []
            new_amp_im_list = []
            for i in range(self.dim):
                if ((i >> qubit) & 1) == result:
                    val_to_assign = ops.divide(self.amplitudes[i], norm_val) # backend complex scalar
                    new_amp_re_list.append(ops.real(val_to_assign))
                    new_amp_im_list.append(ops.imag(val_to_assign))
                else:
                    zero_scalar_dtype = ops.dtype(self.amplitudes).real_dtype if hasattr(ops.dtype(self.amplitudes), 'real_dtype') else tensor.EmberDType.float32
                    zero_scalar = tensor.zeros((), dtype=zero_scalar_dtype, device=self.device)
                    new_amp_re_list.append(zero_scalar)
                    new_amp_im_list.append(zero_scalar)

            # Stack lists of scalar tensors into 1D tensors, then combine into complex
            self.amplitudes = tensor.complex(tensor.stack(new_amp_re_list), tensor.stack(new_amp_im_list))
        
        return result, tensor.item(prob_for_result)
        
    def get_probabilities(self) -> TensorLike: # Returns real backend tensor
        """
        Get state probabilities.

        Returns:
            Probability distribution tensor
        """
        return ops.square(ops.abs(self.amplitudes)) # |amplitudes|^2

class QuantumWave(Module): # Inherit from ember_ml.nn.modules.Module
    """Neural quantum wave processor."""
    
    def __init__(self, num_qubits: int, hidden_size: int):
        """
        Initialize quantum wave processor.

        Args:
            num_qubits: Number of qubits
            hidden_size: Hidden layer dimension
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.hidden_size = hidden_size
        
        # Learnable gates - Parameters should store EmberTensors
        # Create real and imaginary parts and combine
        # Assuming default device is handled by Parameter or global settings
        rand_real_single = tensor.random_normal((num_qubits, 2, 2), dtype=tensor.EmberDType.float32)
        rand_imag_single = tensor.random_normal((num_qubits, 2, 2), dtype=tensor.EmberDType.float32)
        self.single_qubit_gates = Parameter(
            tensor.complex(rand_real_single, rand_imag_single)
        )

        rand_real_ent = tensor.random_normal((num_qubits-1, 4, 4), dtype=tensor.EmberDType.float32)
        rand_imag_ent = tensor.random_normal((num_qubits-1, 4, 4), dtype=tensor.EmberDType.float32)
        self.entangling_gates = Parameter(
            tensor.complex(rand_real_ent, rand_imag_ent)
        )
        
        # Classical processing - ember_ml.nn.container.Linear
        self.pre_quantum = Linear(hidden_size, num_qubits * 2)
        self.post_quantum = Linear(2 ** num_qubits, hidden_size)
        
    def _make_unitary(self, matrix: TensorLike) -> TensorLike: # matrix is complex backend tensor, returns backend
        """Make matrix unitary using QR decomposition."""
        Q, R = ops.linearalg.qr(matrix) # Q, R are backend tensors
        diag_R = ops.linearalg.diag_part(R) # backend tensor

        # Handle potential zero values in diag_R before division to avoid NaN/Inf
        abs_diag_R = ops.abs(diag_R)
        # Create a tensor of ones with the same shape and type as abs_diag_R for where condition
        ones_like_abs_diag_R = tensor.ones_like(abs_diag_R)
        # Create a small epsilon tensor
        epsilon_val = 1e-9 # Or some small float
        epsilon_tensor = tensor.full_like(abs_diag_R, epsilon_val)

        # Replace zeros in abs_diag_R with epsilon to avoid division by zero
        # Note: ops.where(condition, x, y) -> x if condition true, y if false
        # We want abs_diag_R if abs_diag_R > epsilon, else epsilon.
        # Or, more simply, add epsilon to abs_diag_R if that's acceptable.
        # For stability: safe_abs_diag_R = ops.maximum(abs_diag_R, epsilon_tensor)
        # For this example, let's assume ops.divide handles near-zero denominators gracefully or we filter.
        # A common way is to add a small epsilon to the denominator.
        safe_denominator = ops.add(abs_diag_R, epsilon_tensor)
        phases_complex = ops.divide(diag_R, safe_denominator) # backend tensor

        return ops.multiply(Q, tensor.expand_dims(phases_complex, -2)) # backend tensor
        
    def quantum_layer(self, state: QuantumState) -> QuantumState: # Takes and returns QuantumState (holding backend tensors)
        """
        Apply quantum circuit layer.
        """
        for i in range(self.num_qubits):
            # .data of Parameter returns backend tensor
            gate_param_tensor = self.single_qubit_gates.data[i]
            gate = self._make_unitary(gate_param_tensor) # gate is backend tensor
            state.apply_gate(gate, [i]) # apply_gate expects backend tensor
            
        for i in range(self.num_qubits - 1):
            gate_param_tensor = self.entangling_gates.data[i]
            gate = self._make_unitary(gate_param_tensor)
            state.apply_gate(gate, [i, i+1])
            
        return state # Returns QuantumState
        
    def forward(self, x: TensorLike) -> TensorLike: # x is real backend tensor, returns backend tensor
        """
        Process input through quantum circuit.
        """
        # x is already a backend tensor if called from another ember_ml layer/op
        # tensor.shape and .device work on backend tensors via common functions
        batch_size = tensor.shape(x)[0]
        current_device = ops.get_device_of_tensor(x)
        
        params = self.pre_quantum(x) # params is backend tensor
        
        outputs_probs_list = [] # List of backend tensors
        for i in range(batch_size):
            # params[i] will be a backend tensor slice
            current_params_slice = params[i]
            state = self.get_quantum_state(current_params_slice, device=current_device)
            state = self.quantum_layer(state)
            probs = state.get_probabilities() # probs is backend tensor
            outputs_probs_list.append(probs)
            
        stacked_probs = tensor.stack(outputs_probs_list, axis=0) # backend tensor
        return self.post_quantum(stacked_probs) # backend tensor
        
    def get_quantum_state(self, params: TensorLike, device: str) -> QuantumState: # params is backend tensor
        """
        Create quantum state from parameters.

        Args:
            params: State parameters [num_qubits * 2] (real tensor)
            device: Device string

        Returns:
            Initialized quantum state
        """
        state = QuantumState(self.num_qubits, device=device) # Pass device string
        
        # Apply initialization gates
        for i in range(self.num_qubits):
            theta = params[2*i]  # Scalar EmberTensor
            phi = params[2*i + 1]    # Scalar EmberTensor

            # Create rotation gate components using ops
            cos_theta_half = ops.cos(ops.divide(theta, 2.0))
            sin_theta_half = ops.sin(ops.divide(theta, 2.0))
            
            # exp(-1j*phi) = cos(-phi) + j*sin(-phi) = cos(phi) - j*sin(phi)
            # exp( 1j*phi) = cos(phi) + j*sin(phi)
            cos_phi = ops.cos(phi)
            sin_phi = ops.sin(phi)

            # Gate elements (complex scalars as EmberTensors)
            # All inputs to tensor.complex must be real EmberTensors
            # Ensure theta, phi are scalar float tensors if ops need that. .item() might be needed if they are not already.
            # Assuming theta, phi are 0-dim tensors.

            # Element [0,0]: cos(theta/2) -> complex(cos(theta/2), 0)
            el00 = tensor.complex(cos_theta_half, tensor.zeros_like(cos_theta_half))

            # Element [0,1]: -exp(-1j*phi)*sin(theta/2) = - (cos(phi) - j*sin(phi)) * sin(theta/2)
            # = -cos(phi)sin(theta/2) + j*sin(phi)sin(theta/2)
            term_cosphi_sintheta = ops.multiply(cos_phi, sin_theta_half)
            term_sinphi_sintheta = ops.multiply(sin_phi, sin_theta_half)
            el01_real = ops.negate(term_cosphi_sintheta)
            el01_imag = term_sinphi_sintheta
            el01 = tensor.complex(el01_real, el01_imag)

            # Element [1,0]: exp(1j*phi)*sin(theta/2) = (cos(phi) + j*sin(phi)) * sin(theta/2)
            # = cos(phi)sin(theta/2) + j*sin(phi)sin(theta/2)
            el10_real = term_cosphi_sintheta
            el10_imag = term_sinphi_sintheta
            el10 = tensor.complex(el10_real, el10_imag)

            # Element [1,1]: cos(theta/2) -> complex(cos(theta/2), 0)
            el11 = tensor.complex(cos_theta_half, tensor.zeros_like(cos_theta_half))

            # Create gate matrix using tensor.stack or creating from list of lists if convert_to_tensor supports it
            # For a 2x2 matrix: tensor.stack([tensor.stack([el00, el01]), tensor.stack([el10, el11])])
            # This assumes elXX are 0-dim (scalar) tensors.
            # If tensor.convert_to_tensor can take list of list of EmberTensor scalars:
            gate_data_list = [[el00, el01], [el10, el11]]
            # This is tricky. convert_to_tensor might expect Python numbers or backend tensors.
            # A robust way is to stack real and imaginary parts separately and then combine.
            gate_real = tensor.stack([tensor.stack([ops.real(el00), ops.real(el01)]),
                                      tensor.stack([ops.real(el10), ops.real(el11)])])
            gate_imag = tensor.stack([tensor.stack([ops.imag(el00), ops.imag(el01)]),
                                      tensor.stack([ops.imag(el10), ops.imag(el11)])])
            gate = tensor.complex(gate_real, gate_imag)
            
            state.apply_gate(gate, [i])
            
        return state