"""Hybrid LTC / Attention / LSTM composite modules.

This file was refactored to remove deprecated ``EmberDType`` and legacy
``EmberTensor`` wrapper references. It now relies solely on backend-agnostic
functional tensor utilities exposed through :mod:`ember_ml.tensor` and
operations from :mod:`ember_ml.ops`.
"""

from typing import Dict, Any, List

from ember_ml import tensor, ops
from ember_ml.types import TensorLike
from ember_ml.nn.modules import Module, activations
from ember_ml.nn.modules.rnn import LSTM


class AttentionLayer(Module):
    """Simple single-head attention layer for temporal processing."""

    def __init__(self, query_dim: int, key_dim: int, value_dim: int, hidden_dim: int):
        from ember_ml.nn.layers import Linear  # Local import to avoid circulars
        super().__init__()
        self.query_map = Linear(query_dim, hidden_dim)
        self.key_map = Linear(key_dim, hidden_dim)
        self.value_map = Linear(value_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query: TensorLike, key: TensorLike, value: TensorLike) -> TensorLike:
        q_mapped = tensor.expand_dims(self.query_map(query), axis=1)
        k_mapped_transposed = tensor.transpose(self.key_map(key), axes=(0, 2, 1))
        v_mapped = self.value_map(value)
        scores_unscaled = ops.matmul(q_mapped, k_mapped_transposed)
        scores_scaled = ops.divide(scores_unscaled, self.scale)
        attn_weights = ops.softmax(scores_scaled, axis=-1)
        output_expanded = ops.matmul(attn_weights, v_mapped)
        return tensor.squeeze(output_expanded, axis=1)


class HybridNeuron(Module):
    """Hybrid neuron combining LTC dynamics with attention mechanisms."""
    
    def __init__(self,
                 neuron_id: int, # Assuming neuron_id is not used by base Module if not passed
                 tau: float = 1.0,
                 dt: float = 0.01,
                 hidden_size: int = 64,
                 attention_heads: int = 4): # attention_heads not used in current code
        """
        Initialize hybrid neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau: Time constant
            dt: Time step for numerical integration
            hidden_size: Hidden state dimension
            attention_heads: Number of attention heads
        """
        super().__init__() # Adjusted super call if base Module doesn't take these args
        self.neuron_id = neuron_id # Store if needed
        self.tau = tau
        self.dt = dt
        self.hidden_size = hidden_size
        
        # Attention mechanism
        self.attention = AttentionLayer( # This is a local class, also a Module
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size
        )
        
        # Memory buffer for temporal attention
        self.memory_buffer: List[TensorLike] = [] # Stores backend tensors
        self.max_memory_size = 100
        self.state = self._initialize_state() # Initialize state (backend tensor)
        self.history: List[TensorLike] = [] # Initialize history (stores backend tensors)
        
    def _initialize_state(self) -> TensorLike:
        """Initialize neuron state backend tensor."""
        return tensor.zeros((self.hidden_size,), dtype=tensor.float32)
        
    def update(self,
               input_signal: TensorLike,
               **kwargs) -> TensorLike:
        """
        Update neuron state using hybrid processing.

        Args:
            input_signal: Input tensor (backend tensor) [hidden_size]
            **kwargs: Additional parameters

        Returns:
            Updated state tensor (backend tensor) [hidden_size]
        """
        # Append latest input to temporal memory
        current_input_backend = input_signal
        self.memory_buffer.append(current_input_backend)

        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)
            
        # tensor.stack now takes and returns backend tensors (if list contains backend tensors)
        memory = tensor.stack(self.memory_buffer, axis=0)
        
        # Apply attention over memory
        # self.state is a backend tensor
        query_state = tensor.expand_dims(self.state, axis=0)
        # memory is [memory_size, hidden_size], need [1, memory_size, hidden_size] for key/value
        key_value_memory = tensor.expand_dims(memory, axis=0)

        attended = self.attention(
            query_state,
            key_value_memory,
            key_value_memory
        ) # Output of attention is [1, hidden_size]

        attended_squeezed = tensor.squeeze(attended, axis=0) # Remove batch dim -> [hidden_size]
        
        # LTC update with attention-modulated input
        # dh = (1.0 / self.tau) * (attended.squeeze(0) - self.state)
        # self.state = self.state + self.dt * dh
        diff_term = ops.subtract(attended_squeezed, self.state)
        scaled_diff = ops.multiply(1.0 / self.tau, diff_term)
        dh = scaled_diff

        self.state = ops.add(self.state, ops.multiply(self.dt, dh))
        
        # Store history
        self.history.append(tensor.copy(self.state))
        
        return self.state
        
    def save_state(self) -> Dict[str, Any]:
        """Save neuron state and parameters."""
        state_dict = super().save_state() # Assuming base Module has save_state
        # If not, initialize state_dict = {} or handle accordingly
        # state_dict = {} # If super().save_state() doesn't exist or is not what we want
        state_dict.update({
            'neuron_id': self.neuron_id,
            'tau': self.tau,
            'dt': self.dt,
            'hidden_size': self.hidden_size,
            # Convert backend tensors to lists for serialization
            'memory_buffer': [tensor.tolist(m) for m in self.memory_buffer],
            'max_memory_size': self.max_memory_size,
            'state': tensor.tolist(self.state),
            'history': [tensor.tolist(h) for h in self.history]
        })
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load neuron state and parameters."""
        super().load_state(state_dict)
        self.neuron_id = state_dict['neuron_id']
        self.tau = state_dict['tau']
        self.dt = state_dict['dt']
        self.hidden_size = state_dict['hidden_size']
        # tensor.convert_to_tensor returns backend tensors
        self.memory_buffer = [tensor.convert_to_tensor(m) for m in state_dict['memory_buffer']]
        self.max_memory_size = state_dict['max_memory_size']
        self.state = tensor.convert_to_tensor(state_dict['state'])
        self.history = [tensor.convert_to_tensor(h) for h in state_dict['history']]



class HybridLNNModel(Module):
    """
    Hybrid architecture combining LTC networks with LSTM and attention mechanisms.
    Implements the enhanced model from the LNN-CNN analysis.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 lstm_hidden_size: int,
                 output_size: int,
                 parallel_chains: int,
                 attention_hidden_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.parallel_chains = parallel_chains
        
        self.ltc_cells = [ # This should be ModuleList or similar if parameters need registration
            ImprovedLiquidTimeConstantCell(input_size, hidden_size)
            for _ in range(parallel_chains)
        ]
        
        self.lstm = LSTM( # Assuming ember_ml LSTM
            lstm_hidden_size, # input_size for LSTM after LTC layers
            lstm_hidden_size, # hidden_size for LSTM
            batch_first=True
        )
        
        self.attention = AttentionLayer( # Local class
            lstm_hidden_size,
            lstm_hidden_size,
            lstm_hidden_size,
            attention_hidden_dim
        )
        
        from ember_ml.nn.layers import Linear # Local import
        self.output_layer = Linear(lstm_hidden_size, output_size)
        
    def forward(self,
                input_sequence: TensorLike,
                times: TensorLike) -> TensorLike:
        # Shape / device / dtype discovery
        batch_size, seq_len, _ = tensor.shape(input_sequence)
        current_device = ops.get_device(input_sequence)
        input_dtype = tensor.dtype(input_sequence)

        x0_shape = (batch_size, self.hidden_size * self.parallel_chains)
        # tensor.zeros returns backend tensor
        x0 = tensor.zeros(x0_shape, dtype=input_dtype, device=current_device)

        outputs: List[TensorLike] = [] # Stores backend tensors

        for t_idx in range(seq_len - 1):
            # times[t_idx] if times is backend tensor and supports __getitem__ returning backend tensor
            # Or use tensor.slice_tensor for robustness.
            # For simplicity, assuming times[t_idx] gives a scalar backend tensor.
            t_start_val = tensor.item(times[t_idx])
            t_end_val = tensor.item(times[t_idx + 1])
            
            t_span_list = [t_start_val, t_end_val]
            # tensor.convert_to_tensor returns backend tensor
            t_span = tensor.convert_to_tensor(t_span_list, dtype=input_dtype, device=current_device)
            
            u = input_sequence[:, t_idx, :] # u is a backend tensor slice
            
            try:
             # tensor.split returns list of backend tensors
             x0_split = tensor.split(x0, num_splits=self.parallel_chains, axis=1)
            except AttributeError:
                 print("Warning: tensor.split failed, attempting manual slicing for chunks.")
                 split_size = x0.shape[1] // self.parallel_chains
                 x0_split = [
                     tensor.slice_tensor(x0, starts=[0, i * split_size], sizes=[batch_size, split_size])
                     for i in range(self.parallel_chains)
                 ]

            x_new_chains: List[TensorLike] = [] # Stores backend tensors
            for chain_idx in range(self.parallel_chains):
                chain_cell = self.ltc_cells[chain_idx]
                x_chain_current = x0_split[chain_idx] # backend tensor
                u_for_chain = u
                
                # _integrate_ode returns backend tensor
                x_integrated = self._integrate_ode(
                    chain_cell,
                    x_chain_current,
                    t_span,
                    u_for_chain
                )
                x_new_chains.append(x_integrated[-1]) # x_integrated is sequence of backend_tensors
            
            x0 = tensor.concatenate(x_new_chains, axis=1) # x0 is backend tensor
            
            y = self.output_layer(x0) # y is backend tensor
            outputs.append(y)
            
        outputs_stacked = tensor.stack(outputs, axis=1) # backend tensor
        
        return outputs_stacked # Returns backend tensor
    
    def _integrate_ode(self,
                      cell: Module,
                      x0: TensorLike, # Expect backend tensor
                      t_span: TensorLike, # Expect backend tensor
                      u: TensorLike, # Expect backend tensor
                      method: str = 'rk4',
                      options: dict = {'step_size': 0.1}) -> TensorLike: # Returns backend tensor
        """
        Integrate ODE for LTC cell.
        NOTE: This requires a backend-agnostic ODE solver in ember_ml.ops.
        """
        # from torchdiffeq import odeint # Removed
        try:
            # Placeholder for ember_ml's ODE solver
            # The 'args' parameter in torchdiffeq.odeint is passed as a tuple to the cell's forward.
            # Ensure ops.integrate.odeint has a similar mechanism or adapt cell.forward.
            # Current cell.forward is (t, x, u). odeint usually provides (t,x) and other args via tuple.
            # Modifying cell to be func(t, y, *args) where y is state, args is (u,)

            # Wrapper for cell to match expected signature if needed by hypothetical ops.integrate.odeint
            # def cell_wrapper(t, x_state): # u is captured from outer scope
            #    return cell(t, x_state, u)
            # solution = ops.integrate.odeint(func=cell_wrapper, y0=x0, t=t_span, method=method, options=options)

            # Assuming cell's forward signature is (t, x, u_arg) and odeint passes u as an arg
            # This is a conceptual replacement.
            if not hasattr(ops, 'integrate') or not hasattr(ops.integrate, 'odeint'):
                raise NotImplementedError("ops.integrate.odeint is not implemented.")

            # The arguments to cell for torchdiffeq are (t, x) then *args which is (u,)
            # So cell.forward(t, x, u_from_args_tuple)
            # Let's assume ops.integrate.odeint passes 'args' similarly
            solution = ops.integrate.odeint(func=cell, y0=x0, t=t_span, method=method, options=options, args=(u,))
            return solution

        except (AttributeError, NotImplementedError) as e:
            print(f"Warning: ODE integration is not available. Returning initial state. Error: {e}")
            # Fallback: return sequence of x0 for each time point in t_span (or just x0 if t_span is just start/end)
            # This is a dummy behavior.
            num_time_points = tensor.shape(t_span)[0]
            return tensor.stack([x0] * num_time_points, axis=0)


class ImprovedLiquidTimeConstantCell(Module):
    """Enhanced LTC cell with nonlinear dynamics."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        from ember_ml.nn.layers import Linear
        from ember_ml.nn.modules import Parameter
        self.W = Linear(hidden_size, hidden_size)
        self.U = Linear(input_size, hidden_size)
        # Parameters store backend tensors internally if Parameter class was also refactored.
        # tensor.zeros/ones return backend tensors.
        self.b = Parameter(tensor.zeros((hidden_size,), dtype=tensor.float32))
        self.tau = Parameter(tensor.ones((hidden_size,), dtype=tensor.float32))
        
    def forward(self,
                t: TensorLike, # time, often scalar backend tensor
                x: TensorLike, # state, backend tensor
                u: TensorLike  # input, backend tensor
                ) -> TensorLike: # Returns backend tensor
        # x, u, t are backend tensors. Parameter.data should also be backend tensor.
        # All ops.* and activations.* return backend tensors.

        # Ensure 1.0 is treated compatibly, e.g. if x is backend tensor, ops.add should handle scalar.
        # Or convert explicitly:
        one_val = tensor.convert_to_tensor(
            1.0,
            dtype=tensor.dtype(x),
            device=ops.get_device(x)
        )
        nonlinear_term = ops.sqrt(ops.add(x, one_val))

        neg_x = ops.negative(x)
        # .data of Parameter gives the backend tensor
        term1 = ops.divide(neg_x, self.tau.data)

        sum_linear = ops.add(ops.add(self.W(x), self.U(u)), self.b.data)
        term2 = activations.tanh(sum_linear)

        dxdt_unscaled = ops.add(term1, term2)
        
        dxdt = ops.multiply(dxdt_unscaled, nonlinear_term)
        
        return dxdt # Returns backend tensor