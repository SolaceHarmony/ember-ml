"""PyTorch tensor random operations."""

import torch
from typing import Union, Optional

from ember_ml.backend.torch.types import TensorLike, DType, Shape
from ember_ml.backend.torch.tensor.ops.utility import _create_new_tensor # Import helper

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a normal distribution.
    Dtype/Device handling deferred to caller via convert_to_tensor.
    Create a tensor with random values from a normal distribution using the helper.
    """
    # torch.normal takes mean, std, size (shape). Use kwargs.
    # Helper handles dtype and device.
    # Note: shape is passed as 'shape' kwarg - the helper will extract it and pass as 'size' positional arg
    return _create_new_tensor(torch.normal, dtype=dtype, device=device, shape=shape, mean=mean, std=stddev)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor with random uniform values
    """
    # Use the helper function directly, passing torch.rand and scaling parameters
    # This is simpler and more consistent with the MLX implementation
    return _create_new_tensor(torch.rand, dtype=dtype, device=device, shape=shape) * (maxval - minval) + minval

def random_binomial(shape: Shape, p: float = 0.5,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with random values from a binomial distribution using the helper.
    """
    # torch.bernoulli takes probability tensor. Create it first.
    # Helper handles device. Dtype defaults usually work for probs (float).
    prob_tensor = _create_new_tensor(torch.full, shape=shape, fill_value=p, device=device)

    # Use helper for bernoulli call, passing the probability tensor.
    # Let helper resolve final dtype (likely bool or float).
    return _create_new_tensor(torch.bernoulli, dtype=dtype, device=device, input=prob_tensor) # Pass prob_tensor via input kwarg

def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a gamma distribution using the helper.
    """
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")

    # PyTorch uses concentration (alpha) and rate (1/beta)
    rate = 1.0 / beta if beta != 0 else float('inf') # Avoid division by zero

    # Create tensors for alpha and rate using helper to handle dtype/device
    alpha_tensor = _create_new_tensor(torch.full, shape=shape, fill_value=alpha, dtype=dtype, device=device)
    rate_tensor = _create_new_tensor(torch.full, shape=shape, fill_value=rate, dtype=dtype, device=device)

    # Use torch.distributions.Gamma.sample() - this doesn't fit _create_new_tensor well.
    # Call directly after ensuring parameters are tensors.
    alpha_tensor_clamped = torch.clamp(alpha_tensor, min=1e-9)
    rate_tensor_clamped = torch.clamp(rate_tensor, min=1e-9)

    # Ensure tensors are on the correct device
    target_device = alpha_tensor_clamped.device # Use device from created tensors
    gamma_dist = torch.distributions.gamma.Gamma(alpha_tensor_clamped.to(target_device), rate_tensor_clamped.to(target_device))

    # Sample returns tensor on the correct device
    result = gamma_dist.sample() # Sample shape is implicitly handled by alpha/rate shape
    return result

def random_exponential(shape: Shape, scale: float = 1.0,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from an exponential distribution using the helper.
    """
    # PyTorch exponential takes rate (lambda = 1/scale)
    rate = 1.0 / scale if scale != 0 else float('inf')
    # Use helper for torch.empty to get tensor of correct shape/dtype/device
    empty_tensor = _create_new_tensor(torch.empty, shape=shape, dtype=dtype, device=device)
    # Call exponential_ directly on the created tensor
    return empty_tensor.exponential_(lambd=rate)

def random_poisson(shape: Shape, lam: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate random values from a Poisson distribution using the helper.
    """
    # Create rate tensor using helper
    rate_tensor = _create_new_tensor(torch.full, shape=shape, fill_value=lam, dtype=dtype, device=device)
    # Clamp rate tensor to be non-negative
    rate_tensor_clamped = torch.clamp(rate_tensor, min=0)

    # Use helper for poisson call
    return _create_new_tensor(torch.poisson, input=rate_tensor_clamped, dtype=dtype, device=device) # Pass rate tensor as input

def random_categorical(data: TensorLike, num_samples: int,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Draw samples from a categorical distribution.
    Dtype/Device handling deferred to caller via convert_to_tensor.
    """
    from ember_ml.backend.torch.tensor.tensor import TorchTensor # Lazy import - corrected path assuming tensor.py is one level up
    # Instantiate TorchTensor
    tensor_ops = TorchTensor()
    # Use convert_to_tensor for input data, ensuring float and correct device
    logits_tensor = tensor_ops.convert_to_tensor(data, dtype=torch.float32, device=device)
    target_device = logits_tensor.device

    # Convert to probabilities
    probs = torch.softmax(logits_tensor, dim=-1)

    # Sample from the categorical distribution
    samples = torch.multinomial(probs, num_samples, replacement=True)
    return samples

def random_permutation(data: Union[int, TensorLike],
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate a random permutation.
    Dtype/Device handling deferred to caller via convert_to_tensor.
    """
    from ember_ml.backend import get_device # Lazy import
    target_device = device if device is not None else get_device()

    if isinstance(data, int):
        # Generate random permutation on target device (returns int64)
        perm = torch.randperm(data, device=target_device)
        return perm
    else:
        from ember_ml.backend.torch.tensor.tensor import TorchTensor # Lazy import - corrected path assuming tensor.py is one level up
        # Instantiate TorchTensor
        tensor_ops = TorchTensor()
        # Ensure input tensor is on the correct device
        tensor_data = tensor_ops.convert_to_tensor(data, device=target_device)
        shape = tensor_data.shape
        if len(shape) == 0 or shape[0] <= 1:
            return tensor_data

        # Generate indices on the same device
        indices = torch.randperm(shape[0], device=tensor_data.device)
        # Gather
        return tensor_data[indices]

def shuffle(data: TensorLike) -> torch.Tensor:
    """
    Randomly shuffle a tensor along the first dimension.
    Input tensor `data` should be converted and placed on device by the caller.
    """
    from ember_ml.backend.torch.tensor.tensor import TorchTensor # Lazy import - corrected path assuming tensor.py is one level up
    # Instantiate TorchTensor
    tensor_ops = TorchTensor()
    # Convert data just in case, though frontend should handle this
    tensor_data = tensor_ops.convert_to_tensor(data)

    shape = tensor_data.shape
    if len(shape) == 0 or shape[0] <= 1:
        return tensor_data

    # Generate random indices on the same device
    indices = torch.randperm(shape[0], device=tensor_data.device)
    # Gather
    return tensor_data[indices]

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Check if MPS (Metal Performance Shaders) is available in this PyTorch version
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
         torch.mps.manual_seed(seed)

def get_seed() -> Optional[int]:
    """
    Get the current random seed state (as initial seed).
    """
    return torch.initial_seed()

__all__ = [
    "random_normal",
    "random_uniform",
    "random_binomial",
    "random_gamma",
    "random_exponential",
    "random_poisson",
    "random_categorical",
    "random_permutation",
    "shuffle",
    "set_seed",
    "get_seed",
]