# API Reference

This section contains detailed API documentation for Ember ML.

## Core Modules

-   **`ember_ml.ops`**: [Core Operations](ops.md) - Mathematical, device, comparison, I/O, loss, and vector operations.
-   **`ember_ml.ops.linearalg`**: [Linear Algebra Operations](linearalg.md) - Matrix decompositions, solvers, norms, etc.
-   **`ember_ml.ops.stats`**: [Statistical Operations](stats.md) - Mean, variance, median, sum, sorting, etc.
-   **`ember_ml.backend`**: Backend abstraction system (NumPy, PyTorch, MLX).
-   **`ember_ml.training`**: [Training Module](training.md) - Optimizers, loss functions, and metrics for training and evaluating models.

## Neural Network Components

-   **`ember_ml.nn.tensor`**: [Tensor Module](nn_tensor.md) - Backend-agnostic tensor creation, manipulation, random ops, and data types.
-   **`ember_ml.nn.features`**: [Feature Extraction](nn_features.md) - PCA, normalization, standardization, one-hot encoding, specialized extractors.
-   **`ember_ml.nn.modules`**: [Core NN Modules](nn_modules.md) - Base `Module`, `Parameter`, `Dense`, `NCP`.
-   **`ember_ml.nn.modules.activations`**: [Activation Modules](nn_activations.md) - ReLU, Tanh, Sigmoid, Dropout, etc.
-   **`ember_ml.nn.modules.rnn`**: [Recurrent NN Modules](nn_modules_rnn.md) - RNN, LSTM, GRU, CfC, LTC, Stride-Aware variants.
-   **`ember_ml.nn.modules.rnn`**: [Quantum-Inspired Neural Networks](nn_modules_rnn_quantum.md) - LQNet, CTRQNet, quantum-classical hybrid neurons.
-   **`ember_ml.nn.modules.wiring`**: [Neuron Maps (Wiring)](nn_modules_wiring.md) - Connectivity patterns (NCPMap, FullyConnectedMap, RandomMap).

## Other Modules

-   **`ember_ml.initializers`**: Weight initialization (Glorot, Binomial, etc.).
-   **`ember_ml.utils`**: Utility functions (metrics, visualization, math helpers).
-   **`ember_ml.models`**: Higher-level model implementations (RBM, Liquid).

## Theoretical Frameworks

-   [Grand Unified Cognitive Equation (GUCE)](guce.md): A theoretical framework that treats the universe and all matter and energy as a neural system.
-   [GUCE Architecture](guce_architecture.md): Implementation design for the GUCE framework within Ember ML.
-   [Abacus Neural Architecture](abacus_neural_architecture.md): A novel approach to neural network design that organizes neurons in layered 1D spaces.
-   [Boltzmann-Hebbian Dynamics](boltzmann_hebbian_dynamics.md): A framework that balances stochastic exploration with deterministic stability.
-   [Spatial Hebbian Network](spatial_hebbian_network.md): A 3D neural architecture that mimics biological brain development with proximity-based connectivity.
-   [Telomere Memory System](telomere_memory_system.md): A biologically-inspired approach to neural network memory management with age-based decay.
-   [Age Constant Memory](age_constant_memory.md): A paradigm shift from time-based to usage-based memory decay for more resilient and contextual memory.
-   [Retinal Flash Architecture](retinal_flash_architecture.md): A system combining parallel input processing with sequential attention for efficient handling of massive data streams.
-   [Prefrontal Attention Control](prefrontal_attention_control.md): A hierarchical control system inspired by the prefrontal cortex that orchestrates attention, working memory, and metacognition.
-   [Hamiltonian Cognitive Dynamics](hamiltonian_cognitive_dynamics.md): A framework modeling cognition as a physical system governed by Hamiltonian mechanics, where thought processes evolve as wave functions.
-   [Fractal Harmonic Embedding](fractal_harmonic_embedding.md): A revolutionary approach to high-dimensional vector compression using fractal mathematics and harmonic analysis.
-   [Liquid CFC xLSTM](liquid_cfc_xlstm.md): A hybrid neural architecture combining continuous-time dynamics with extended LSTM gating and Metal-accelerated tile-based processing.
-   [Metal Kernel Implementation](metal_kernel_implementation.md): Detailed explanation of the Metal shader implementation for the Liquid CFC xLSTM with tile-based asynchronous processing.
-   [Extended Long Short-Term Memory (xLSTM)](xlstm.md): A significant advancement in recurrent neural networks featuring exponential gating, novel memory structures, and efficient implementation techniques.
-   [GUCE-CFC Integration](guce_cfc_integration.md): A unified neural architecture that combines the theoretical foundations of GUCE with the practical temporal processing capabilities of CFC.
-   [Weight Transfer and Model Bootstrapping](weight_transfer.md): Techniques for transferring weights and bootstrapping models between different neural network architectures, with a focus on leveraging pre-trained xLSTM weights.

## Comprehensive Guides

-   [Frontend Usage Guide](frontend_usage_guide.md): Comprehensive guide on using the Ember ML frontend.
-   [Tensor Operations Architecture](tensor_architecture.md): Detailed explanation of the tensor operations architecture.

For detailed documentation on specific functions and classes, refer to the linked module pages or the docstrings in the source code.
