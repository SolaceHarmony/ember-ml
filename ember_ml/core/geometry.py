"""Geometric operations for non-Euclidean neural computations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def normalize_sphere(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors to the unit sphere.
    
    Args:
        vec: Input vectors (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized vectors on unit sphere
    """
    norm = torch.norm(vec, dim=-1, keepdim=True)
    mask = norm > eps
    return torch.where(mask, vec / norm, vec)

def log_map_sphere(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """Logarithmic map on unit sphere (Log_p(q)).
    
    Maps point q to the tangent space at p.
    
    Args:
        p: Base point(s) (..., 3)
        q: Target point(s) (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Tangent vector(s) at p
    """
    # Normalize inputs to unit sphere
    p_n = normalize_sphere(p, eps)
    q_n = normalize_sphere(q, eps)
    
    # Compute angle between p and q
    dot_prod = torch.sum(p_n * q_n, dim=-1, keepdim=True)
    dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
    theta = torch.arccos(dot_prod)
    
    # Handle small angles
    small_angle = theta < eps
    if small_angle.any():
        return torch.zeros_like(p)
    
    # Compute direction in tangent space
    perp = q_n - dot_prod * p_n
    perp_norm = torch.norm(perp, dim=-1, keepdim=True)
    perp_mask = perp_norm > eps
    
    # Combine results
    dir_vec = torch.where(perp_mask, perp / perp_norm, torch.zeros_like(perp))
    return dir_vec * theta

def exp_map_sphere(
    p: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """Exponential map on unit sphere (Exp_p(v)).
    
    Maps tangent vector v at p to the sphere.
    
    Args:
        p: Base point(s) (..., 3)
        v: Tangent vector(s) at p (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Point(s) on sphere
    """
    # Get vector norm
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    small_norm = v_norm < eps
    
    if small_norm.any():
        return p
    
    # Normalize base point and direction
    p_n = normalize_sphere(p, eps)
    dir_vec = v / v_norm
    
    # Remove component along p
    proj_p = torch.sum(dir_vec * p_n, dim=-1, keepdim=True) * p_n
    dir_vec = dir_vec - proj_p
    dir_vec = normalize_sphere(dir_vec, eps)
    
    # Compute new point
    new_point = torch.cos(v_norm) * p_n + torch.sin(v_norm) * dir_vec
    return normalize_sphere(new_point, eps)

def parallel_transport_sphere(
    p: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """Parallel transport tangent vector v from p to q on sphere.
    
    Args:
        p: Start point(s) (..., 3)
        q: End point(s) (..., 3)
        v: Tangent vector(s) at p (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Transported vector(s) at q
    """
    # Normalize points
    p_n = normalize_sphere(p, eps)
    q_n = normalize_sphere(q, eps)
    
    # Get geodesic
    dot_prod = torch.sum(p_n * q_n, dim=-1, keepdim=True)
    dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
    theta = torch.arccos(dot_prod)
    
    # Handle small angles
    small_angle = theta < eps
    if small_angle.any():
        return v
        
    # Get transport direction
    transport_dir = q_n - dot_prod * p_n
    transport_dir = normalize_sphere(transport_dir, eps)
    
    # Transport v
    v_proj_p = torch.sum(v * p_n, dim=-1, keepdim=True) * p_n
    v_perp = v - v_proj_p
    
    transported = (
        torch.cos(theta) * v_perp +
        torch.sin(theta) * torch.cross(transport_dir, v_perp)
    )
    
    return transported

class SphericalLinear(nn.Module):
    """Linear transformation in spherical geometry.
    
    Maps tangent vectors between spherical tangent spaces.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(
        self,
        x: torch.Tensor,
        base_point: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spherical linear transformation.
        
        Args:
            x: Input tangent vectors at base_point
            base_point: Point on sphere where input vectors live
            
        Returns:
            (output tangent vectors, new base point)
        """
        # Linear transform in tangent space
        output = F.linear(x, self.weight, self.bias)
        
        # Map result to sphere
        new_point = exp_map_sphere(base_point, output)
        
        # Get output in tangent space at new point
        output_tangent = log_map_sphere(new_point, output)
        
        return output_tangent, new_point