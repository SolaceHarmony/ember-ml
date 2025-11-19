"""Demonstration of spherical (non-Euclidean) LTC neurons."""

# import torch # Removed
import matplotlib.pyplot as plt

from ember_ml import ops  # For ops.multiply, ops.sin, ops.pi, ops.linearalg.norm etc.
from ember_ml import tensor  # For tensor.EmberTensor, tensor.zeros etc.
# Updated import path for Spherical LTC components
from ember_ml.nn.modules.rnn.spherical_ltc import (
    SphericalLTCConfig,
    SphericalLTCChain
)


def generate_input_signal(
    total_time: float = 10.0,
    pattern_time: float = 5.0,
    dt: float = 0.01,
    freq: float = 1.0
) -> tensor.EmberTensor: # Updated return type hint
    """Generate sinusoidal input signal.
    
    Args:
        total_time: Total simulation time
        pattern_time: Duration of input pattern
        dt: Time step
        freq: Signal frequency
        
    Returns:
        Input signal tensor (num_steps, 3)
    """
    num_steps = int(total_time / dt)
    pattern_steps = int(pattern_time / dt)
    
    # Create time array
    t = tensor.linspace(0, total_time, num_steps, dtype=tensor.EmberDType.float32) # Use tensor.linspace
    
    # Generate signal
    signal_1d = tensor.zeros((num_steps,), dtype=tensor.EmberDType.float32) # Explicit shape tuple
    
    if pattern_steps > 0:
        # Slice of time array for the pattern
        t_slice = tensor.slice_tensor(t, starts=[0], sizes=[pattern_steps])

        # Argument for sin: 2 * pi * freq * t_slice
        arg_mult1 = ops.multiply(2.0, ops.pi) # 2 * pi
        arg_mult2 = ops.multiply(arg_mult1, freq) # 2 * pi * freq
        arg = ops.multiply(arg_mult2, t_slice) # 2 * pi * freq * t_slice

        sin_values = ops.sin(arg)

        # Update the signal tensor using slice_update
        # signal_1d[:pattern_steps] = sin_values
        # Assuming signal_1d is 1D, key for slice_update would be a slice object or compatible.
        # For simplicity, if slice_update can take typical Python slice syntax for EmberTensor:
        # However, explicit slice_update is safer if __setitem__ is not fully robust or clear.
        # Using slice_update:
        # Need to ensure indices are correct. If starts=[0], updates=sin_values (which has length pattern_steps)
        # This implies replacing the first pattern_steps elements.
        # Assuming slice_update(original_tensor, start_indices_per_dim, update_tensor_or_values)
        # This might be more like a scatter update.
        # A direct way if __setitem__ is implemented:
        # for i in range(pattern_steps):
        #    signal_1d[i] = sin_values[i]
        # Or, if slice_update is for replacing a slice with another tensor of same shape:
        signal_1d = tensor.slice_update(signal_1d, [slice(0, pattern_steps)], sin_values)


    # Convert to 3D
    input_3d = tensor.zeros((num_steps, 3), dtype=tensor.EmberDType.float32) # Explicit shape tuple
    # Project onto x-axis. This requires updating a slice of input_3d.
    # input_3d[:, 0] = signal_1d
    # Using slice_update: update the 0-th column of all rows.
    # indices for slice_update would be (slice(None), 0)
    # update values need to be reshaped to (num_steps, 1) for some backends if signal_1d is (num_steps,)
    signal_1d_col = tensor.reshape(signal_1d, (num_steps, 1))
    input_3d = tensor.slice_update(input_3d, [slice(None), 0], tensor.squeeze(signal_1d_col, axis=1)) # Squeeze back if slice_update needs 1D update for 1D slice target
    
    return input_3d

def run_simulation(
    chain: SphericalLTCChain,
    input_signal: tensor.EmberTensor, # Updated type hint
    batch_size: int = 1
) -> tensor.EmberTensor: # Updated return type hint
    """Run simulation of spherical LTC chain.
    
    Args:
        chain: Spherical LTC chain
        input_signal: Input signal (num_steps, 3)
        batch_size: Batch size for parallel simulation
        
    Returns:
        State history tensor (batch_size, num_steps, num_neurons, 3)
    """
    num_steps = tensor.shape(input_signal)[0] # Use tensor.shape
    states_history = []
    
    # Reset chain
    chain.reset_states(batch_size) # Assuming this takes care of device for states
    
    # Run simulation
    for step in range(num_steps):
        # current_slice = input_signal[step:step+1] -> tensor.slice_tensor
        # Assuming input_signal is (num_steps, num_features=3)
        # Slice for one time step: starts=[step, 0], sizes=[1, num_features]
        current_slice = tensor.slice_tensor(input_signal, starts=[step, 0], sizes=[1, tensor.shape(input_signal)[1]])

        # expand(batch_size, -1) -> tensor.tile
        # current_slice shape is (1, num_features). Tile to (batch_size, num_features)
        # reps would be (batch_size, 1)
        input_batch = tensor.tile(current_slice, reps=(batch_size, 1))

        states, _ = chain(input_batch) # chain.forward should handle EmberTensors
        states_history.append(states) # states should be EmberTensor
        
    return tensor.stack(states_history, axis=1) # Use tensor.stack, ensure states_history contains EmberTensors

def plot_results(
    states: tensor.EmberTensor, # Updated type hint
    input_signal: tensor.EmberTensor, # Updated type hint
    pattern_time: float,
    dt: float
):
    """Plot simulation results.
    """
    num_steps = tensor.shape(states)[1] # Use tensor.shape
    num_neurons = tensor.shape(states)[2] # Use tensor.shape

    # time = tensor.arange(num_steps) * dt
    # Assuming tensor.arange exists like np.arange or torch.arange
    # If tensor.arange is not available, use tensor.linspace or np.arange and convert
    # For now, assume tensor.arange(0, num_steps) * dt works or:
    time_values = [i * dt for i in range(num_steps)]

    # Plot norms (should stay close to 1)
    plt.figure(figsize=(12, 6))

    # norms = torch.norm(states[0], dim=-1)
    # Slice first batch item: states[0] -> tensor.slice_tensor(states, [0], [1]) -> shape (1, time, num_neurons, 3)
    # Then squeeze batch dim -> tensor.squeeze(..., axis=0) -> shape (time, num_neurons, 3)
    states_0_squeezed = tensor.squeeze(tensor.slice_tensor(states, starts=[0,0,0,0], sizes=[1,num_steps,num_neurons,3]), axis=0)
    norms = ops.linearalg.norm(states_0_squeezed, axis=-1) # Norm along the last dimension (coords)
    
    for i in range(num_neurons):
        # norms[:, i] -> tensor.slice_tensor for columns or direct slicing if EmberTensor supports
        # norms_i = norms[:, i] # Assuming norms is (time, num_neurons)
        # For plotting, convert to numpy
        plt.plot(time_values, tensor.to_numpy(norms[:, i]), label=f'LTC {i+1} Norm') # norms[:,i] if supported
        
    plt.axvline(pattern_time, color='r', linestyle='--', label='Pattern End')
    plt.title('Spherical LTC Chain - State Vector Norms')
    plt.xlabel('Time (s)')
    plt.ylabel('||x||')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot x-axis projections
    plt.figure(figsize=(12, 6))
    
    # Input signal: input_signal[:, 0]
    # Convert to numpy for plotting
    plt.plot(time_values, tensor.to_numpy(input_signal[:, 0]), label='Input', linewidth=2) # input_signal[:,0] if supported
    
    # Neuron states: states[0, :, i, 0]
    for i in range(num_neurons):
        # Slice to get the x-component of the i-th neuron for the first batch item
        # state_proj_i = states[0, :, i, 0] # Assuming this slicing works on EmberTensor
        # Convert to numpy for plotting
        plt.plot(
            time_values,
            tensor.to_numpy(states[0, :, i, 0]), # state_proj_i
            label=f'LTC {i+1}',
            alpha=0.8
        )
        
    plt.axvline(pattern_time, color='r', linestyle='--', label='Pattern End')
    plt.title('Spherical LTC Chain - X-axis Projections')
    plt.xlabel('Time (s)')
    plt.ylabel('x(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    # Parameters
    total_time = 10.0
    pattern_time = 5.0
    dt = 0.01
    num_neurons = 3
    
    # Create config
    config = SphericalLTCConfig(
        tau=1.0,
        gleak=0.5,
        dt=dt
    )
    
    # Create chain
    chain = SphericalLTCChain(num_neurons, config)
    
    # Generate input
    input_signal = generate_input_signal(
        total_time=total_time,
        pattern_time=pattern_time,
        dt=dt
    )
    
    # Run simulation
    states = run_simulation(chain, input_signal)
    
    # Plot results
    plot_results(states, input_signal, pattern_time, dt)
    
    # Compute forgetting times
    forgetting_times = chain.get_forgetting_times(
        states,
        threshold=0.05
    )
    
    # Print results
    print("\nForgetting Analysis:")
    for i, time in forgetting_times.items():
        if time is not None:
            print(f"LTC {i+1} forgot pattern after {time:.2f}s")
        else:
            print(f"LTC {i+1} maintained pattern")
            
if __name__ == "__main__":
    main()
