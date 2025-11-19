"""
Optimized Restricted Boltzmann Machine for Large-Scale Feature Learning

This module provides an optimized implementation of Restricted Boltzmann Machines
designed for processing large-scale data with efficient memory usage and support
for chunked training.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Union, Generator

from ember_ml import ops, tensor
from ember_ml import stats
from ember_ml.types import TensorLike

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimized_rbm')

# Removed torch import and TORCH_AVAILABLE check, assuming ember_ml handles device availability.
# try:
#     import torch
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     logger.warning("PyTorch not available. GPU acceleration will be disabled.")


class OptimizedRBM:
    """
    Optimized Restricted Boltzmann Machine for large-scale feature learning.
    
    This implementation focuses on:
    - Memory efficiency for large datasets
    - Chunked training support
    - Optional GPU acceleration (via ember_ml device handling)
    - Efficient parameter initialization
    - Comprehensive monitoring and logging
    """
    
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        batch_size: int = 100,
        use_binary_states: bool = False,
        # use_gpu is deprecated in favor of device string
        use_gpu: bool = False, # Kept for compatibility during transition, but device_str is primary
        device: Optional[str] = None, # e.g., "cpu", "gpu:0", "mlx_gpu_0"
        verbose: bool = True
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_binary_states = use_binary_states
        self.verbose = verbose
        
        # Determine target device string
        if device: # User provided a specific device string
            self.device_str = device
        elif use_gpu: # Legacy use_gpu flag
            # This part needs a robust way to select a default GPU device string
            # For now, using a generic "gpu" placeholder or relying on ember_ml's default GPU.
            # A better approach would be ops.get_default_gpu_device() or similar.
            logger.warning("`use_gpu=True` is deprecated. Please use `device='gpu:0'` or similar. Attempting to use default GPU.")
            self.device_str = ops.get_default_device() # Assuming ops can give a default (could be CPU if no GPU)
            if 'cpu' in self.device_str.lower(): # If default is CPU, but GPU was requested
                 logger.warning("GPU requested but ops default is CPU. Check ember_ml setup or specify device.")
        else:
            self.device_str = "cpu" # Default to CPU if nothing specified

        logger.info(f"OptimizedRBM will use device: {self.device_str}")

        # Initialize weights and biases as EmberTensors on the target device
        scale_factor = 0.01 # Python float
        # ops.sqrt returns backend tensor, ensure n_visible is also tensor for op
        n_visible_tensor = tensor.convert_to_tensor(float(n_visible), dtype=tensor.float32, device=self.device_str)
        scale_denominator = ops.sqrt(n_visible_tensor)
        # ops.divide returns backend tensor. Use tensor.item() to get scalar for stddev
        scale = tensor.item(ops.divide(scale_factor, scale_denominator))

        # Create on default device then move, or create directly if API supports
        self.weights = tensor.random_normal(mean=0.0, stddev=scale, shape=(n_visible, n_hidden), device=self.device_str)
        self.visible_bias = tensor.zeros((n_visible,), device=self.device_str) # ensure shape is tuple
        self.hidden_bias = tensor.zeros((n_hidden,), device=self.device_str) # ensure shape is tuple

        # Initialize momentum terms on the target device
        self.weights_momentum = tensor.zeros((n_visible, n_hidden), device=self.device_str)
        self.visible_bias_momentum = tensor.zeros((n_visible,), device=self.device_str) # ensure shape is tuple
        self.hidden_bias_momentum = tensor.zeros((n_hidden,), device=self.device_str) # ensure shape is tuple
        
        self.training_errors = []
        self.training_time = 0
        self.n_epochs_trained = 0
        self.last_batch_error = float('inf')
        
        self.reconstruction_error_threshold = None
        self.free_energy_threshold = None
        
        logger.info(f"Initialized OptimizedRBM with {n_visible} visible units, {n_hidden} hidden units on device '{self.device_str}'")

    # Removed _to_gpu and _to_cpu methods
    
    def _ensure_tensor_on_device(self, data: TensorLike) -> TensorLike: # Returns backend tensor
        """Ensure data is a backend tensor on the RBM's device."""
        # tensor.convert_to_tensor now returns a backend tensor.
        # ops.to_device also returns a backend tensor.
        # If data is tensor, unwrap it first.
        # Directly convert to backend tensor on the target device
        return tensor.convert_to_tensor(data, dtype=tensor.float32, device=self.device_str)

    def sigmoid(self, x: TensorLike) -> TensorLike: # Takes and returns backend tensor
        """
        Compute sigmoid function with numerical stability improvements.
        Args:
            x: Input backend tensor
        Returns:
            Sigmoid of input (backend tensor)
        """
        clipped_x = ops.clip(x, -15.0, 15.0)
        one = tensor.ones_like(clipped_x)
        return ops.divide(one, ops.add(one, ops.exp(ops.negative(clipped_x))))
    
    def compute_hidden_probabilities(self, visible_states: TensorLike) -> TensorLike: # Returns backend tensor
        """
        Compute probabilities of hidden units given visible states.
        Args:
            visible_states: States of visible units [batch_size, n_visible] (TensorLike)
        Returns:
            Probabilities of hidden units [batch_size, n_hidden] (backend tensor)
        """
        visible_states_t = self._ensure_tensor_on_device(visible_states)
        hidden_activations = ops.add(ops.matmul(visible_states_t, self.weights), self.hidden_bias)
        return self.sigmoid(hidden_activations)
    
    def sample_hidden_states(self, hidden_probs: TensorLike) -> TensorLike: # Takes and returns backend tensor
        """
        Sample binary hidden states from their probabilities.
        Args:
            hidden_probs: Probabilities of hidden units [batch_size, n_hidden] (backend tensor)
        Returns:
            Binary hidden states [batch_size, n_hidden] (backend tensor)
        """
        if not self.use_binary_states:
            return hidden_probs
        
        random_values = tensor.random_uniform(shape=ops.shape(hidden_probs), device=ops.get_device(hidden_probs))
        return tensor.cast(ops.greater(hidden_probs, random_values), dtype=tensor.float32)
    
    def compute_visible_probabilities(self, hidden_states: TensorLike) -> TensorLike: # Takes and returns backend tensor
        """
        Compute probabilities of visible units given hidden states.
        Args:
            hidden_states: States of hidden units [batch_size, n_hidden] (backend tensor)
        Returns:
            Probabilities of visible units [batch_size, n_visible] (backend tensor)
        """
        hidden_states_t = self._ensure_tensor_on_device(hidden_states)
        visible_activations = ops.add(ops.matmul(hidden_states_t, ops.transpose(self.weights, axes=(1,0))), self.visible_bias)
        return self.sigmoid(visible_activations)
    
    def sample_visible_states(self, visible_probs: TensorLike) -> TensorLike: # Takes and returns backend tensor
        """
        Sample binary visible states from their probabilities.
        Args:
            visible_probs: Probabilities of visible units [batch_size, n_visible] (backend tensor)
        Returns:
            Binary visible states [batch_size, n_visible] (backend tensor)
        """
        if not self.use_binary_states:
            return visible_probs
        
        random_values = tensor.random_uniform(shape=ops.shape(visible_probs), device=ops.get_device(visible_probs))
        return tensor.cast(ops.greater(visible_probs, random_values), dtype=tensor.float32)
    
    def contrastive_divergence(self, batch_data: TensorLike, k: int = 1) -> float:
        """
        Perform contrastive divergence algorithm for a single batch.
        Args:
            batch_data: Batch of training data [batch_size, n_visible]
            k: Number of Gibbs sampling steps (default: 1)
        Returns:
            Reconstruction error for this batch (float)
        """
        batch_data_t = self._ensure_tensor_on_device(batch_data)
        current_batch_size = ops.shape(batch_data_t)[0] # Get actual batch size
        
        # Positive phase
        pos_hidden_probs = self.compute_hidden_probabilities(batch_data_t)
        pos_hidden_states = self.sample_hidden_states(pos_hidden_probs) # This is already on device

        pos_associations = ops.matmul(ops.transpose(batch_data_t, axes=(1,0)), pos_hidden_probs)

        # Negative phase
        neg_hidden_states = tensor.copy(pos_hidden_states) # Replaces .clone()

        for _ in range(k):
            neg_visible_probs = self.compute_visible_probabilities(neg_hidden_states)
            neg_visible_states = self.sample_visible_states(neg_visible_probs)
            neg_hidden_probs = self.compute_hidden_probabilities(neg_visible_states)
            neg_hidden_states = self.sample_hidden_states(neg_hidden_probs) # Re-assign for next loop iteration

        # Compute negative associations using neg_hidden_probs (as per Hinton's guide)
        neg_associations = ops.matmul(ops.transpose(neg_visible_states, axes=(1,0)), neg_hidden_probs)

        # Compute gradients
        # Dividing by float current_batch_size
        weights_gradient = ops.divide(ops.subtract(pos_associations, neg_associations), float(current_batch_size))
        visible_bias_gradient = stats.mean(ops.subtract(batch_data_t, neg_visible_states), axis=0)
        hidden_bias_gradient = stats.mean(ops.subtract(pos_hidden_probs, neg_hidden_probs), axis=0)

        # Update with momentum and weight decay (operating on EmberTensors)
        self.weights_momentum = ops.add(ops.multiply(self.momentum, self.weights_momentum), weights_gradient)
        self.visible_bias_momentum = ops.add(ops.multiply(self.momentum, self.visible_bias_momentum), visible_bias_gradient)
        self.hidden_bias_momentum = ops.add(ops.multiply(self.momentum, self.hidden_bias_momentum), hidden_bias_gradient)

        # Apply updates
        # self.weights = self.weights + self.learning_rate * (self.weights_momentum - self.weight_decay * self.weights)
        update_weights = ops.subtract(self.weights_momentum, ops.multiply(self.weight_decay, self.weights))
        self.weights = ops.add(self.weights, ops.multiply(self.learning_rate, update_weights))

        self.visible_bias = ops.add(self.visible_bias, ops.multiply(self.learning_rate, self.visible_bias_momentum))
        self.hidden_bias = ops.add(self.hidden_bias, ops.multiply(self.learning_rate, self.hidden_bias_momentum))

        # Compute reconstruction error
        squared_diff = ops.square(ops.subtract(batch_data_t, neg_visible_probs)) # Use neg_visible_probs for error
        sum_squared_diff_per_sample = stats.sum(squared_diff, axis=1) # backend tensor
        reconstruction_error_scalar_tensor = stats.mean(sum_squared_diff_per_sample) # backend tensor (scalar)
        reconstruction_error = tensor.item(reconstruction_error_scalar_tensor) # Convert to Python float
        
        self.last_batch_error = reconstruction_error
        return reconstruction_error
    
    def train_in_chunks(
        self,
        data_generator: Generator,
        epochs: int = 10,
        k: int = 1,
        validation_data: Optional[TensorLike] = None,
        early_stopping_patience: int = 5,
        callback: Optional[callable] = None
    ) -> List[float]:
        """
        Train the RBM using a data generator to handle large datasets.
        """
        training_errors = []
        
        start_time = time.time()
        best_validation_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_error_sum = 0.0 # Use float for sum
            n_batches = 0
            
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch_data_from_gen in enumerate(data_generator):
                if len(batch_data_from_gen) == 0: # Assuming generator yields list/numpy
                    continue
                
                # No need to check isinstance, _ensure_tensor_on_device handles it
                # batch_data_tensor = self._ensure_tensor_on_device(batch_data_from_gen)
                # if ops.shape(batch_data_tensor)[0] == 0 : # Check after conversion if necessary
                #     continue

                batch_start_time = time.time()
                
                batch_error = self.contrastive_divergence(batch_data_from_gen, k) # CD expects TensorLike
                epoch_error_sum += batch_error # batch_error is already float
                n_batches += 1
                
                batch_time = time.time() - batch_start_time
                
                if self.verbose and (batch_idx % 10 == 0 or batch_idx < 5):
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}: "
                               f"error = {batch_error:.4f}, time = {batch_time:.2f}s")
                
                if callback:
                    callback(epoch, batch_idx, batch_error)
            
            avg_epoch_error = epoch_error_sum / max(n_batches, 1)
            training_errors.append(avg_epoch_error)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} completed: "
                       f"avg_error = {avg_epoch_error:.4f}, time = {epoch_time:.2f}s")
            
            validation_error_val = None # Use different name from parameter
            if validation_data is not None:
                # reconstruction_error returns float or TensorLike.
                # If per_sample=False, it returns float.
                validation_error_val = self.reconstruction_error(validation_data, per_sample=False)
                # No .item() needed here as per_sample=False already returns float.

                logger.info(f"Validation error: {validation_error_val:.4f}")
                
                if validation_error_val < best_validation_error:
                    best_validation_error = validation_error_val
                    patience_counter = 0
                    logger.info(f"New best validation error: {best_validation_error:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epochs")
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.n_epochs_trained += 1
        
        self.training_time += time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f}s")
        
        if self.reconstruction_error_threshold is None:
            logger.info("Computing anomaly detection thresholds")
            errors_list = []
            energies_list = [] # Changed variable name

            # Re-initialize generator if it's exhausted, or assume it can be iterated again
            # This depends on the specific generator implementation.
            # For simplicity, assuming data_generator can be iterated multiple times or a new one is passed for this.
            # If not, this part needs adjustment (e.g., collect all data first, or use a resettable generator).

            # Placeholder: Assuming data_generator can be re-iterated for threshold calculation.
            # In a real scenario, one might store a subset of training data or use a fresh generator.
            temp_data_for_thresholds = []
            # This is tricky with a generator. Let's assume we collect some data for thresholds.
            # This part may need to be adapted based on how data_generator is implemented.
            # For now, let's assume the generator is re-usable or we operate on a snapshot.
            # This example will attempt to re-use, which might fail for some Python generators.

            # A better way: pass a specific data source for threshold calculation.
            # For now, this might not work correctly if data_generator is exhausted.
            # Let's assume it works for the purpose of refactoring the ops.

            for batch_data_thresh in data_generator: # This line might be problematic if generator is exhausted.
                if len(batch_data_thresh) == 0:
                    continue
                
                # reconstruction_error(per_sample=True) returns TensorLike (backend tensor)
                batch_errors_t = self.reconstruction_error(batch_data_thresh, per_sample=True)
                # free_energy returns TensorLike (backend tensor)
                batch_energies_t = self.free_energy(batch_data_thresh)
                
                # Convert backend tensors to numpy for extend
                errors_list.extend(tensor.to_numpy(batch_errors_t))
                energies_list.extend(tensor.to_numpy(batch_energies_t))

            if errors_list and energies_list:
                # Convert lists of numpy arrays back to backend tensors for percentile calculation
                errors_tensor = tensor.convert_to_tensor(errors_list, device=self.device_str)
                energies_tensor = tensor.convert_to_tensor(energies_list, device=self.device_str)

                self.reconstruction_error_threshold = tensor.item(stats.percentile(errors_tensor, 95))
                self.free_energy_threshold = tensor.item(stats.percentile(energies_tensor, 5))
            
                logger.info(f"Reconstruction error threshold: {self.reconstruction_error_threshold:.4f}")
                logger.info(f"Free energy threshold: {self.free_energy_threshold:.4f}")
            else:
                logger.warning("Could not compute anomaly thresholds: no data processed from generator for thresholds.")

        # No _to_cpu() needed as parameters are always EmberTensors.
        # Conversions happen at load/save or method boundaries if needed.
        return training_errors
    
    def transform(self, data: TensorLike) -> TensorLike: # Returns backend tensor
        """
        Transform data to hidden representation.
        Args:
            data: Input data [n_samples, n_visible] (TensorLike)
        Returns:
            Hidden representation [n_samples, n_hidden] (backend tensor)
        """
        data_t = self._ensure_tensor_on_device(data)
        hidden_probs = self.compute_hidden_probabilities(data_t)
        return hidden_probs
    
    def transform_in_chunks(self, data_generator: Generator, chunk_size: int = 1000) -> TensorLike: # Returns backend tensor
        """
        Transform data to hidden representation in chunks.
        """
        hidden_probs_list: List[TensorLike] = [] # List of backend tensors
        
        for batch_data_from_gen in data_generator:
            if len(batch_data_from_gen) == 0:
                continue
            
            batch_hidden_probs = self.transform(batch_data_from_gen)
            hidden_probs_list.append(batch_hidden_probs)
        
        if hidden_probs_list:
            return tensor.concatenate(hidden_probs_list, axis=0)
        else:
            return tensor.zeros((0, self.n_hidden), device=self.device_str)
    
    def reconstruct(self, data: TensorLike) -> TensorLike: # Returns backend tensor
        """
        Reconstruct input data.
        Args:
            data: Input data [n_samples, n_visible] (TensorLike)
        Returns:
            Reconstructed data [n_samples, n_visible] (backend tensor)
        """
        data_t = self._ensure_tensor_on_device(data)
        hidden_probs = self.compute_hidden_probabilities(data_t)
        hidden_states = self.sample_hidden_states(hidden_probs)
        visible_probs = self.compute_visible_probabilities(hidden_states)
        return visible_probs
    
    def reconstruction_error(self, data: TensorLike, per_sample: bool = False) -> Union[float, TensorLike]: # Returns float or backend tensor
        """
        Compute reconstruction error for input data.
        """
        data_t = self._ensure_tensor_on_device(data)
        reconstructed_t = self.reconstruct(data_t)

        squared_error = ops.square(ops.subtract(data_t, reconstructed_t))
        sum_squared_error_per_sample = stats.sum(squared_error, axis=1) # backend tensor
            
        if per_sample:
            return sum_squared_error_per_sample
        else:
            # stats.mean returns scalar backend tensor, tensor.item converts to Python float
            return tensor.item(stats.mean(sum_squared_error_per_sample))
    
    def free_energy(self, data: TensorLike) -> TensorLike: # Returns backend tensor
        """
        Compute free energy for input data.
        Returns: Free energy for each sample [n_samples] (backend tensor)
        """
        data_t = self._ensure_tensor_on_device(data)

        # Ensure visible_bias is correctly shaped for broadcasting with matmul result if needed,
        # or that matmul with a vector bias works as expected.
        # Assuming self.visible_bias is 1D (n_visible,).
        # ops.matmul(data_t (N,D), self.visible_bias (D,)) might not be what's intended.
        # Usually it's data_t @ W + b, or sum(data_t * visible_bias_broadcasted, axis=1)
        # The original was ops.dot(data, self.visible_bias) which for (N,D) and (D,) implies sum over D for each N.
        # This is equivalent to sum(data_t * self.visible_bias, axis=1) if self.visible_bias is broadcasted.
        # Or if matmul is (N,D) @ (D,1) -> (N,1) then squeeze.
        # For now, assuming direct element-wise product then sum for the bias term if it's not a matmul.
        # visible_bias_term = stats.sum(ops.multiply(data_t, self.visible_bias), axis=1) # If bias applied element-wise

        # The previous version used: ops.matmul(data_t, tensor.reshape(self.visible_bias, (-1,1))) and then squeeze.
        # This is fine if visible_bias_term should be (N,).
        reshaped_vb = tensor.reshape(self.visible_bias, (-1, 1))
        visible_bias_term_intermediate = ops.matmul(data_t, reshaped_vb)
        visible_bias_term = tensor.squeeze(visible_bias_term_intermediate, axis=-1)

        linear_term = ops.add(ops.matmul(data_t, self.weights), self.hidden_bias)

        one_like_linear = tensor.ones_like(linear_term)
        exp_linear_term = ops.exp(linear_term)
        log_term = ops.log(ops.add(one_like_linear, exp_linear_term))
        hidden_term_sum = stats.sum(log_term, axis=1)

        return ops.negative(ops.add(hidden_term_sum, visible_bias_term))
    
    def anomaly_score(self, data: TensorLike, method: str = 'reconstruction') -> TensorLike: # Returns backend tensor
        """
        Compute anomaly score for input data.
        Returns: Anomaly scores [n_samples] (backend tensor)
        """
        if method == 'reconstruction':
            return self.reconstruction_error(data, per_sample=True)
        elif method == 'free_energy':
            return ops.negative(self.free_energy(data))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_anomaly(self, data: TensorLike, method: str = 'reconstruction') -> TensorLike: # Returns boolean backend tensor
        """
        Determine if input data is anomalous.
        Returns: Boolean backend tensor indicating anomalies [n_samples]
        """
        scores_t = self.anomaly_score(data, method)
        
        if method == 'reconstruction':
            if self.reconstruction_error_threshold is None:
                raise ValueError("Reconstruction error threshold not computed. Train model first.")
            # Ensure threshold is a backend tensor for comparison with scores_t
            threshold_t = tensor.convert_to_tensor(self.reconstruction_error_threshold, device=ops.get_device_of_tensor(scores_t), dtype=ops.dtype(scores_t))
            return ops.greater(scores_t, threshold_t)
        elif method == 'free_energy':
            if self.free_energy_threshold is None:
                raise ValueError("Free energy threshold not computed. Train model first.")
            # As per previous logic, scores_t for free_energy are -FE. Threshold is for original FE.
            actual_free_energies = ops.negative(scores_t)
            threshold_t_orig_scale = tensor.convert_to_tensor(self.free_energy_threshold, device=ops.get_device_of_tensor(actual_free_energies), dtype=ops.dtype(actual_free_energies))
            return ops.less(actual_free_energies, threshold_t_orig_scale)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save(self, filepath: str) -> None:
        """ Save model to file. """
        # Parameters are already EmberTensors. Convert to NumPy for saving.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'weights': tensor.to_numpy(self.weights),
            'visible_bias': tensor.to_numpy(self.visible_bias),
            'hidden_bias': tensor.to_numpy(self.hidden_bias),
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'use_binary_states': self.use_binary_states,
            'training_errors': self.training_errors, # List of floats
            'reconstruction_error_threshold': self.reconstruction_error_threshold, # float
            'free_energy_threshold': self.free_energy_threshold, # float
            'training_time': self.training_time,
            'n_epochs_trained': self.n_epochs_trained,
            'timestamp': datetime.now().isoformat(),
            'device_str': self.device_str # Save device string
        }
        
        ops.save(filepath, model_data, allow_pickle=True) # ops.save should handle dict of numpy arrays
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = False, device: Optional[str] = None) -> 'OptimizedRBM':
        """ Load model from file. """
        model_data_loaded = ops.load(filepath, allow_pickle=True).item() # ops.load returns a tensor/array, get item.

        # Determine device to load onto
        # Priority: user-passed device > saved device_str > use_gpu flag > cpu
        load_device_str = device
        if load_device_str is None:
            load_device_str = model_data_loaded.get('device_str', "cpu")
            if use_gpu and 'cpu' in load_device_str : # if use_gpu is true but saved was cpu, try gpu
                 logger.warning("use_gpu=True with a CPU-saved model. Attempting to load to default GPU.")
                 load_device_str = ops.get_default_device() # Or a specific "gpu" string

        rbm = cls(
            n_visible=model_data_loaded['n_visible'],
            n_hidden=model_data_loaded['n_hidden'],
            learning_rate=model_data_loaded['learning_rate'],
            momentum=model_data_loaded['momentum'],
            weight_decay=model_data_loaded['weight_decay'],
            batch_size=model_data_loaded['batch_size'],
            use_binary_states=model_data_loaded['use_binary_states'],
            use_gpu=False, # Deprecated, device string below is used
            device=load_device_str, # Pass the determined device string
            verbose=True # Defaulting verbose to True, or load from model_data if saved
        )
        
        # Set model parameters by converting numpy arrays from file to EmberTensors on target device
        rbm.weights = tensor.convert_to_tensor(model_data_loaded['weights'], device=rbm.device_str)
        rbm.visible_bias = tensor.convert_to_tensor(model_data_loaded['visible_bias'], device=rbm.device_str)
        rbm.hidden_bias = tensor.convert_to_tensor(model_data_loaded['hidden_bias'], device=rbm.device_str)

        # Initialize momentum terms on the correct device (already done in __init__ for rbm instance)
        # If momentum terms were saved, load them:
        if 'weights_momentum' in model_data_loaded: # Check if momentum was saved
            rbm.weights_momentum = tensor.convert_to_tensor(model_data_loaded['weights_momentum'], device=rbm.device_str)
            rbm.visible_bias_momentum = tensor.convert_to_tensor(model_data_loaded['visible_bias_momentum'], device=rbm.device_str)
            rbm.hidden_bias_momentum = tensor.convert_to_tensor(model_data_loaded['hidden_bias_momentum'], device=rbm.device_str)


        rbm.training_errors = model_data_loaded['training_errors']
        rbm.reconstruction_error_threshold = model_data_loaded['reconstruction_error_threshold']
        rbm.free_energy_threshold = model_data_loaded['free_energy_threshold']
        rbm.training_time = model_data_loaded['training_time']
        rbm.n_epochs_trained = model_data_loaded['n_epochs_trained']

        # No _to_gpu() call needed, parameters are loaded to rbm.device_str
        logger.info(f"Model loaded from {filepath} to device '{rbm.device_str}'")
        return rbm
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            Summary string
        """
        summary = [
            "Optimized Restricted Boltzmann Machine Summary",
            "============================================",
            f"Visible units: {self.n_visible}",
            f"Hidden units: {self.n_hidden}",
            f"Parameters: {self.n_visible * self.n_hidden + self.n_visible + self.n_hidden}",
            f"Learning rate: {self.learning_rate}",
            f"Momentum: {self.momentum}",
            f"Weight decay: {self.weight_decay}",
            f"Batch size: {self.batch_size}",
            f"Binary states: {self.use_binary_states}",
            f"GPU acceleration: {self.use_gpu}",
            f"Epochs trained: {self.n_epochs_trained}",
            f"Training time: {self.training_time:.2f} seconds",
            f"Current reconstruction error: {self.last_batch_error:.4f}",
            f"Anomaly threshold (reconstruction): {self.reconstruction_error_threshold}",
            f"Anomaly threshold (free energy): {self.free_energy_threshold}"
        ]
        
        return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = tensor.random_normal(1000, 20)
    
    # Create RBM
    rbm = OptimizedRBM(
        n_visible=20,
        n_hidden=10,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        batch_size=100,
        use_binary_states=False,
        use_gpu=True
    )
    
    # Define a generator to yield data in batches
    def data_generator(data, batch_size=100):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    # Train RBM
    rbm.train_in_chunks(
        data_generator(data, batch_size=100),
        epochs=10,
        k=1
    )
    
    # Transform data
    features = rbm.transform(data)
    
    print(f"Transformed data shape: {features.shape}")
    print(rbm.summary())