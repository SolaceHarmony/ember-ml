"""
Adam optimizer for ember_ml.

This module provides a backend-agnostic implementation of the Adam optimizer
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Dict, List, Optional, Union, Any, Callable

from ember_ml import ops
from ember_ml.training.optimizer.base import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer.
    
    This optimizer implements the Adam algorithm.
    """
    
    def __init__(
        self,
        params=None,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False
    ):
        """
        Initialize the Adam optimizer.
        
        Args:
            params: Parameters to optimize (Module or list of parameters)
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use the AMSGrad variant
        """
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        self.defaults = defaults
        self.state = {}
        self.param_groups = []
        
        if params is not None:
            self.add_param_group(params)
    
    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = ops.add(grad, ops.multiply(weight_decay, param.data))
                
                # Get state for this parameter
                state = self.state.get(param, {})
                
                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = tensor.zeros_like(param.data)
                    state['exp_avg_sq'] = tensor.zeros_like(param.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = tensor.zeros_like(param.data)
                
                # Update step count
                state['step'] += 1
                
                # Update biased first moment estimate (momentum)
                exp_avg = state['exp_avg']
                exp_avg = ops.add(
                    ops.multiply(beta1, exp_avg),
                    ops.multiply(1.0 - beta1, grad)
                )
                state['exp_avg'] = exp_avg
                
                # Update biased second raw moment estimate
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq = ops.add(
                    ops.multiply(beta2, exp_avg_sq),
                    ops.multiply(1.0 - beta2, ops.square(grad))
                )
                state['exp_avg_sq'] = exp_avg_sq
                
                # Compute bias correction
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                
                # Apply AMSGrad if enabled
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = ops.maximum(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = ops.sqrt(max_exp_avg_sq)
                else:
                    denom = ops.sqrt(exp_avg_sq)
                
                # Apply bias correction to denominator
                denom = ops.divide(denom, ops.sqrt(bias_correction2))
                
                # Compute step size
                step_size = lr / bias_correction1
                
                # Update parameter
                param.data = ops.subtract(
                    param.data,
                    ops.divide(
                        ops.multiply(step_size, exp_avg),
                        ops.add(denom, eps)
                    )
                )
                
                # Update state
                self.state[param] = state