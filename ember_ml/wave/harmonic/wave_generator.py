import numpy as np

def harmonic_wave(params, t, batch_size):
    """
    Generate a harmonic wave based on parameters.
    Handles batch processing for multiple embeddings.
    
    Args:
        params (np.ndarray): Array of shape (batch_size, 3*n_components) containing
                           amplitudes, frequencies, and phases for each component
        t (np.ndarray): Time points at which to evaluate the wave
        batch_size (int): Number of waves to generate in parallel
        
    Returns:
        np.ndarray: Generated harmonic waves of shape (batch_size, len(t))
    """
    harmonics = []
    for i in range(batch_size):
        amplitudes, frequencies, phases = np.split(params[i], 3)
        harmonic = (
            amplitudes[:, None] * np.sin(2 * np.pi * frequencies[:, None] * t + phases[:, None])
        )
        harmonics.append(harmonic.sum(axis=0))
    return np.vstack(harmonics)

def map_embeddings_to_harmonics(embeddings):
    """
    Initialize harmonic parameters for all embeddings in a batch.
    
    Args:
        embeddings (np.ndarray): Input embeddings of shape (batch_size, embedding_dim)
        
    Returns:
        np.ndarray: Initialized parameters of shape (batch_size, 3*embedding_dim)
                   containing amplitudes, frequencies, and phases
    """
    batch_size, embedding_dim = embeddings.shape
    params = []
    for i in range(batch_size):
        params.append(np.random.rand(3 * embedding_dim))  # Amplitudes, Frequencies, Phases
    return np.vstack(params)