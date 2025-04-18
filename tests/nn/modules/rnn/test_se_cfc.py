"""
Tests for the Spatially Embedded Closed-form Continuous-time (seCfC) neural network.
"""

import unittest
import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import EnhancedNCPMap
from ember_ml.nn.modules.rnn import seCfC

class TestSeCfC(unittest.TestCase):
    """Test suite for the seCfC neural network."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        tensor.set_seed(42)
        # No need for np.random.seed as we're using tensor operations
        
        # Create a small neuron map for testing
        self.neuron_map = EnhancedNCPMap(
            inter_neurons=4,
            command_neurons=2,
            motor_neurons=2,
            sensory_neurons=3,
            neuron_type="cfc",
            time_scale_factor=1.0,
            activation="tanh",
            recurrent_activation="sigmoid",
            sparsity_level=0.5,
            seed=42
        )
        
        # Create a seCfC model for testing
        self.model = seCfC(
            neuron_map=self.neuron_map,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            regularization_strength=0.01
        )
        
        # Create test data
        self.batch_size = 2
        self.time_steps = 5
        self.input_features = 3
        self.output_features = 2
        
        self.inputs = tensor.random_normal(
            (self.batch_size, self.time_steps, self.input_features))
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, seCfC)
        self.assertEqual(self.model.neuron_map.units, 8)  # 4 + 2 + 2
        self.assertEqual(self.model.neuron_map.output_dim, 2)
        self.assertEqual(self.model.neuron_map.input_dim, 3)
    
    def test_build(self):
        """Test that the model builds correctly."""
        # Build the model
        self.model.build(tensor.shape(self.inputs))
        
        # Check that the model is built
        self.assertTrue(self.model.built)
        
        # Check that the parameters are initialized
        self.assertIsNotNone(self.model.kernel)
        self.assertIsNotNone(self.model.recurrent_kernel)
        self.assertIsNotNone(self.model.bias)
        
        # Check parameter shapes
        self.assertEqual(tensor.shape(self.model.kernel.data), (3, 32))  # (input_dim, units * 4)
        self.assertEqual(tensor.shape(self.model.recurrent_kernel.data), (8, 32))  # (units, units * 4)
        self.assertEqual(tensor.shape(self.model.bias.data), (32,))  # (units * 4,)
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Forward pass
        outputs = self.model(self.inputs)
        
        # Check output shape
        self.assertEqual(tensor.shape(outputs), (self.batch_size, self.time_steps, self.output_features))
    
    def test_return_state(self):
        """Test that the model can return state."""
        # Create a model that returns state
        model = seCfC(
            neuron_map=self.neuron_map,
            return_sequences=True,
            return_state=True,
            go_backwards=False,
            regularization_strength=0.01
        )
        
        # Forward pass
        outputs, states = model(self.inputs)
        
        # Check output shape
        self.assertEqual(tensor.shape(outputs), (self.batch_size, self.time_steps, self.output_features))
        
        # Check state shape
        self.assertEqual(len(states), 2)
        self.assertEqual(tensor.shape(states[0]), (self.batch_size, 8))  # (batch_size, units)
        self.assertEqual(tensor.shape(states[1]), (self.batch_size, 8))  # (batch_size, units)
    
    def test_go_backwards(self):
        """Test that the model can process sequences backwards."""
        # Create a model that processes sequences backwards
        model = seCfC(
            neuron_map=self.neuron_map,
            return_sequences=True,
            return_state=False,
            go_backwards=True,
            regularization_strength=0.01
        )
        
        # Forward pass
        outputs = model(self.inputs)
        
        # Check output shape
        self.assertEqual(tensor.shape(outputs), (self.batch_size, self.time_steps, self.output_features))
    
    def test_regularization_loss(self):
        """Test that the model computes regularization loss."""
        # Build the model
        self.model.build(tensor.shape(self.inputs))
        
        # Compute regularization loss
        reg_loss = self.model.get_regularization_loss()
        
        # Check that the loss is a scalar
        self.assertEqual(tensor.shape(reg_loss), ())
        
        # Check that the loss is non-negative
        self.assertGreaterEqual(reg_loss.numpy(), 0.0)
    
    def test_reset_state(self):
        """Test that the model can reset state."""
        # Reset state
        states = self.model.reset_state(batch_size=3)
        
        # Check state shape
        self.assertEqual(len(states), 2)
        self.assertEqual(tensor.shape(states[0]), (3, 8))  # (batch_size, units)
        self.assertEqual(tensor.shape(states[1]), (3, 8))  # (batch_size, units)
        
        # Check that the state is zeros
        self.assertTrue(np.allclose(tensor.to_numpy(states[0]), 0.0))
        self.assertTrue(np.allclose(tensor.to_numpy(states[1]), 0.0))
    
    def test_get_config(self):
        """Test that the model can get its configuration."""
        # Get config
        config = self.model.get_config()
        
        # Check that the config contains the expected keys
        self.assertIn("neuron_map", config)
        self.assertIn("return_sequences", config)
        self.assertIn("return_state", config)
        self.assertIn("go_backwards", config)
        self.assertIn("regularization_strength", config)
        
        # Check that the config values are correct
        self.assertEqual(config["return_sequences"], True)
        self.assertEqual(config["return_state"], False)
        self.assertEqual(config["go_backwards"], False)
        self.assertEqual(config["regularization_strength"], 0.01)
    
    def test_training(self):
        """Test that the model can be trained."""
        # Create target data
        targets = tensor.random_normal((self.batch_size, self.time_steps, self.output_features))
        
        # Create optimizer
        optimizer = ops.optimizers.Adam(learning_rate=0.01)
        
        # Initial forward pass
        with ops.GradientTape() as tape:
            outputs = self.model(self.inputs)
            loss = ops.mse(targets, outputs)
        
        # Initial loss - use tensor operations to get the value
        initial_loss = tensor.to_numpy(loss)
        
        # Train for a few steps
        for _ in range(5):
            with ops.GradientTape() as tape:
                outputs = self.model(self.inputs)
                loss = ops.mse(targets, outputs)
            
            # Compute gradients
            gradients = tape.gradient(loss, self.model.parameters())
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, self.model.parameters()))
        
        # Final forward pass
        with ops.GradientTape() as tape:
            outputs = self.model(self.inputs)
            loss = ops.mse(targets, outputs)
        
        # Final loss - use tensor operations to get the value
        final_loss = tensor.to_numpy(loss)
        
        # Check that the loss decreased
        self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()