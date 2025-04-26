# API Reference

This section contains detailed API documentation for Ember ML.

## Core Modules

-   **`ember_ml.ops`**: [Core Operations](ops.md) - Mathematical, device, comparison, I/O, loss, and vector operations.
-   **`ember_ml.ops.linearalg`**: [Linear Algebra Operations](ops_linearalg.md) - Matrix decompositions, solvers, norms, etc.
-   **`ember_ml.ops.stats`**: [Statistical Operations](ops_stats.md) - Mean, variance, median, sum, sorting, etc.
-   **`ember_ml.ops.bitwise`**: [Bitwise Operations](ops_bitwise.md) - Bitwise AND, OR, XOR, NOT operations.
-   **`ember_ml.backend`**: Backend abstraction system (NumPy, PyTorch, MLX).
-   **`ember_ml.training`**: [Training Module](training.md) - Optimizers, loss functions, and metrics for training and evaluating models.

## Neural Network Components

-   **`ember_ml.nn.tensor`**: [Tensor Module](nn_tensor.md) - Backend-agnostic tensor creation, manipulation, random ops, and data types.
-   **`ember_ml.nn.features`**: [Feature Extraction](nn_features.md) - PCA, normalization, standardization, one-hot encoding, specialized extractors.
-   **`ember_ml.nn.modules`**: [Core NN Modules](nn_modules.md) - Base `Module`, `Parameter`, `Dense`, `NCP`.
-   **`ember_ml.nn.modules.containers`**: [Container Modules](nn_containers.md) - Sequential, ModuleList, etc.
-   **`ember_ml.nn.modules.activations`**: [Activation Modules](nn_activations.md) - ReLU, Tanh, Sigmoid, Dropout, etc.
-   **`ember_ml.nn.modules.rnn`**: [Recurrent NN Modules](nn_modules_rnn.md) - RNN, LSTM, GRU, CfC, LTC, Stride-Aware variants.
-   **`ember_ml.nn.modules.rnn`**: [Quantum-Inspired Neural Networks](nn_modules_rnn_quantum.md) - LQNet, CTRQNet, quantum-classical hybrid neurons.
-   **`ember_ml.nn.modules.wiring`**: [Neuron Maps (Wiring)](nn_modules_wiring.md) - Connectivity patterns, including enhanced and spatial wiring (NCPMap, FullyConnectedMap, RandomMap, EnhancedNeuronMap, EnhancedNCPMap, etc.).

## Other Modules

-   **`ember_ml.initializers`**: Weight initialization (Glorot, Binomial, etc.).
-   **`ember_ml.utils`**: [Utility Functions](utilities.md) - Metrics, visualization, math helpers, etc.
-   **`ember_ml.models`**: Higher-level model implementations (RBM, Liquid).
-   **`ember_ml.wave`**: [Wave Module](wave/index.md) - Wave-based models and operations.


## Comprehensive Guides

-   [Frontend Usage Guide](frontend_usage_guide.md): Comprehensive guide on using the Ember ML frontend.
-   [Tensor Operations Architecture](tensor_architecture.md): Detailed explanation of the tensor operations architecture.

For detailed documentation on specific functions and classes, refer to the linked module pages or the docstrings in the source code.
