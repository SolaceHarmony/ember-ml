import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
# Removed: import torch
# Removed: from sklearn.metrics.pairwise import cosine_similarity (unused)
from ember_ml import ops, stats
# Ensure stats ops are accessible if stats.mean is used later
# from ember_ml.ops import stats # Or access via stats.mean
from ember_ml import tensor # Ensure tensor is imported for tensor.EmberTensor, tensor.stack etc.
from typing import List, Optional
def harmonic_wave(params: tensor.EmberTensor, t: tensor.EmberTensor, batch_size: int) -> tensor.EmberTensor:
    """
    Generate a harmonic wave based on parameters.
    Handles batch processing for multiple embeddings.
    """
    harmonics_list = [] # Changed name to avoid conflict
    for i in range(batch_size):
        # Assuming params[i] yields an EmberTensor for a single sample
        amplitudes, frequencies, phases = tensor.split_tensor(params[i], num_splits=3) # Use num_splits

        # Ensure t is compatible for broadcasting, e.g., shape (num_time_steps,)
        # amplitudes, frequencies, phases are likely (num_harmonics_per_param_group,)
        # We need amplitudes/frequencies/phases to be (num_harmonics, 1) and t to be (1, num_time_steps) for broadcasting,
        # or ensure ops handle broadcasting correctly.
        # The original code `amplitudes[:, None]` suggests this.

        # Reshape for broadcasting: (num_harmonics,) -> (num_harmonics, 1)
        amp_reshaped = tensor.reshape(amplitudes, (-1, 1))
        freq_reshaped = tensor.reshape(frequencies, (-1, 1))
        phases_reshaped = tensor.reshape(phases, (-1, 1))
        t_reshaped = tensor.reshape(t, (1, -1)) # (num_time_steps,) -> (1, num_time_steps)

        term_freq_t = ops.multiply(freq_reshaped, t_reshaped)
        term_2pi_freq_t = ops.multiply(ops.multiply(2.0, ops.pi), term_freq_t) # ops.pi should be float or EmberTensor
        sin_input = ops.add(term_2pi_freq_t, phases_reshaped)
        sin_output = ops.sin(sin_input)

        harmonic_components = ops.multiply(amp_reshaped, sin_output) # (num_harmonics, num_time_steps)

        # Summing over harmonics (axis=0)
        current_harmonic = stats.sum(harmonic_components, axis=0) # Ensure sum is over correct axis
        harmonics_list.append(current_harmonic)

    # Stack along batch dimension (axis=0)
    return tensor.stack(harmonics_list, axis=0)

# Load transformer model and tokenizer - This part remains dependent on 'transformers'
# and its underlying backend (e.g., PyTorch or TensorFlow).
# The goal is to make the *output* of this function an EmberTensor.
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name) # Renamed to avoid conflict if 'model' is used elsewhere

def generate_embeddings(texts: List[str]) -> Optional[tensor.EmberTensor]:
    """
    Generate embeddings for a list of texts using a pretrained transformer.
    Output is an EmberTensor.
    """
    if not texts:
        # Return an empty EmberTensor with appropriate shape (0, embedding_dim)
        # Need to know embedding_dim, or return None / raise error.
        # For now, returning None if texts is empty.
        # A more robust solution might involve getting embedding_dim from hf_model.config.hidden_size
        # and returning tensor.zeros((0, hf_model.config.hidden_size))
        return None

    embeddings_list = []
    # The HuggingFace model call itself will use its own backend (PyTorch in this case)
    # The conversion to EmberTensor happens immediately after.
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # This is a PyTorch-specific block
        # No gradients needed for inference with pretrained model
        # import torch # Not needed at top level anymore
        # with torch.no_grad():
        pt_outputs = hf_model(**inputs) # hf_model is the PyTorch model

        # Extract CLS token embedding (PyTorch tensor)
        pt_cls_embedding = pt_outputs.last_hidden_state[:, 0, :]

        # Convert the PyTorch tensor to EmberTensor.
        # Assuming tensor.convert_to_tensor can handle a PyTorch tensor directly,
        # or by first converting pt_cls_embedding to NumPy array.
        # For safety, converting to NumPy first is often more portable if direct PT->Ember isn't guaranteed.
        # requires_grad=False is good practice for embeddings not needing further training here.
        try:
            # Attempt direct conversion if supported, or via numpy
            # If pt_cls_embedding is on GPU, it needs .cpu() first for numpy conversion
            np_embedding = pt_cls_embedding.cpu().detach().numpy()
            ember_cls_embedding = tensor.convert_to_tensor(np_embedding, requires_grad=False)
        except AttributeError as e: # If .cpu().detach().numpy() fails (e.g. not a PT tensor)
            # This path indicates an issue with the assumption about hf_model output type
            print(f"Error converting transformer output: {e}. Ensure 'transformers' is using PyTorch and output is as expected.")
            # Fallback or re-raise:
            # For now, if conversion fails for one, we might want to skip or error out for all.
            # Propagating None or raising an error.
            return None # Or raise specific error

        embeddings_list.append(ember_cls_embedding)

    if not embeddings_list: # Should not happen if texts was not empty and no errors
        return None

    # Stack the list of EmberTensor embeddings into a single EmberTensor
    return tensor.stack(embeddings_list, axis=0)

# Example texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "AI is transforming the world of technology.",
    "Deep learning enables powerful language models."
]

# Generate embeddings (now returns EmberTensor or None)
embeddings = generate_embeddings(texts)

def map_embeddings_to_harmonics(embeddings: tensor.EmberTensor) -> tensor.EmberTensor:
    """
    Initialize harmonic parameters for all embeddings in a batch.
    """
    if embeddings is None:
        # Or handle error appropriately
        raise ValueError("Input embeddings cannot be None")

    batch_size, embedding_dim = tensor.shape(embeddings) # Use tensor.shape
    params_list = [] # Changed name
    for _ in range(batch_size): # Use _ if i is not used
        # ops.multiply needs tensor inputs if it's strict, or handles scalars.
        # Assuming num_params is int here.
        num_params = 3 * embedding_dim
        params_list.append(tensor.random_normal(shape=(num_params,), dtype=embeddings.dtype, device=embeddings.device))
    return tensor.stack(params_list, axis=0) # Use tensor.stack

def loss_function(params: tensor.EmberTensor, t: tensor.EmberTensor, target_embedding: tensor.EmberTensor) -> tensor.EmberTensor:
    """
    Compute the loss between the target embedding and the generated harmonic wave.
    Uses Mean Squared Error (MSE) as the metric.
    """
    # Generate harmonic wave for the given parameters
    # Assuming params is (num_harmonics_params), target_embedding is (embedding_dim)
    # harmonic_wave expects batched params. Here we process one sample.
    # So, params needs to be (1, num_harmonics_params) for harmonic_wave if it expects batch.
    # Or, adjust harmonic_wave or this function.
    # For now, assuming params is for a single sample, and harmonic_wave can handle it or is adapted.
    # If harmonic_wave is called with batch_size=1:
    
    # The original harmonic_wave sums over harmonics.
    # amplitudes, frequencies, phases = tensor.split_tensor(params, 3)
    # harmonic_summed_over_harmonics = (
    #     amplitudes[:, None] * ops.sin(2 * ops.pi * frequencies[:, None] * t + phases[:, None])
    # ).sum(axis=0) # This results in (num_time_steps,) or (embedding_dim,) if num_time_steps == embedding_dim

    # Let's call the batched harmonic_wave with batch_size=1 by unsqueezing params
    # and then squeezing the output.
    params_batched = tensor.expand_dims(params, axis=0)
    generated_wave_batched = harmonic_wave(params_batched, t, batch_size=1)
    generated_wave = tensor.squeeze(generated_wave_batched, axis=0) # Shape: (embedding_dim,)

    diff = ops.subtract(target_embedding, generated_wave)
    squared_diff = ops.square(diff)
    loss = stats.mean(squared_diff)
    return loss


def compute_gradients(params: tensor.EmberTensor, t: tensor.EmberTensor, target_embedding: tensor.EmberTensor, epsilon: float =1e-5) -> tensor.EmberTensor:
    """
    Compute numerical gradients for the harmonic parameters using finite differences.
    """
    gradients = tensor.zeros_like(params)
    # Convert epsilon to a tensor of same dtype as params for ops.add/subtract
    epsilon_t = tensor.convert_to_tensor(epsilon, dtype=params.dtype, device=params.device)

    # Assuming params is 1D (num_params_per_sample)
    for i in range(tensor.shape(params)[0]): # Iterate over elements of the param tensor for one sample
        params_step_plus = tensor.copy(params)
        params_step_minus = tensor.copy(params)

        # Add epsilon to element i
        # This relies on EmberTensor __setitem__ being correctly implemented via slice_update / scatter
        current_val_plus = ops.add(params_step_plus[i], epsilon_t)
        params_step_plus[i] = current_val_plus
        loss_plus = loss_function(params_step_plus, t, target_embedding)

        # Subtract epsilon from element i (original params[i] - epsilon)
        current_val_minus = ops.subtract(params[i], epsilon_t) # params.copy()[i] might be safer if params[i] is view
                                                              # but copy is at start of loop.
        params_step_minus[i] = current_val_minus
        loss_minus = loss_function(params_step_minus, t, target_embedding)

        loss_diff = ops.subtract(loss_plus, loss_minus)
        denominator = ops.multiply(2.0, epsilon_t)
        gradients[i] = ops.divide(loss_diff, denominator)
    return gradients


def train_harmonic_embeddings(embeddings: tensor.EmberTensor, t: tensor.EmberTensor, batch_size: int, learning_rate: float =0.01, epochs: int =100) -> tensor.EmberTensor:
    """
    Train harmonic wave parameters to match transformer embeddings.
    Handles multiple embeddings in batch.
    """
    params = map_embeddings_to_harmonics(embeddings)  # Random initialization
    for epoch in range(epochs):
        total_loss = 0
        for i in range(batch_size):
            # Compute loss
            loss = loss_function(params[i], t, embeddings[i])
            
            # Compute gradients
            gradients = compute_gradients(params[i], t, embeddings[i])
            
            # Update parameters using ops
            update_step = ops.multiply(learning_rate, gradients)
            # Assuming direct modification works, see note in compute_gradients
            params[i] = ops.subtract(params[i], update_step)

            # Accumulate loss using ops.add
            # Ensure total_loss is initialized as a tensor or 0.0
            if i == 0 and epoch == 0: # Initialize total_loss correctly on first step
                 total_loss = loss # Assign first loss
            else:
                 total_loss = ops.add(total_loss, loss)

        # Calculate average loss using ops.divide
        avg_loss = ops.divide(total_loss, tensor.convert_to_tensor(float(batch_size)))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {tensor.item(avg_loss)}") # Use tensor.item() to print scalar loss
    return params


# Visualize embeddings vs harmonic waves
def visualize_embeddings(target, learned):
    """
    Visualize target embeddings and learned harmonic embeddings.
    """
    plt.figure(figsize=(12, 6))

    # Plot target embeddings
    plt.subplot(211)
    plt.imshow(target, aspect="auto", cmap="viridis")
    plt.title("Target Embeddings")
    plt.colorbar()

    # Plot learned harmonic embeddings (reshaped)
    plt.subplot(212)
    plt.imshow(learned, aspect="auto", cmap="viridis")
    plt.title("Learned Harmonic Embeddings")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    if __name__ == "__main__":
        # Remove numpy import
        # import numpy as np

        # Generate time steps using tensor.linspace
        # Assuming embeddings is already an EmberTensor or compatible
        num_time_steps = tensor.shape(embeddings)[1]
        t = tensor.linspace(0.0, 5.0, num_time_steps) # Use tensor.linspace

        # Train harmonic embeddings
        batch_size = embeddings.shape[0]  # Number of embeddings (batch size)
        params = train_harmonic_embeddings(embeddings, t, batch_size)
    
        # Generate learned harmonic waves
        learned_harmonic_wave = harmonic_wave(params, t, batch_size)
    
        # Reshape learned harmonic wave to match embeddings
        if learned_harmonic_wave.shape == embeddings.shape:
            # Use tensor.reshape function
            learned_harmonic_wave = tensor.reshape(learned_harmonic_wave, embeddings.shape)
        else:
            raise ValueError(
                f"Shape mismatch: learned wave shape {learned_harmonic_wave.shape}, "
                f"expected {embeddings.shape}"
            )
    
        # Visualize the results
        visualize_embeddings(embeddings, learned_harmonic_wave)