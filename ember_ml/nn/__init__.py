"""Neural network primitives and core components.

This module provides the foundational building blocks for constructing neural
networks with backend-agnostic implementations.

Components:
    Base Modules:
        - Module: Base class for all neural components
        - Parameter: Trainable parameter abstraction
        - TensorInterface: Backend-agnostic tensor interface
        
    Neural Circuit Components:
        - AutoNCP: Automatic Neural Circuit Policy generator
        - Wiring: Base class for neural connectivity patterns
        - RandomWiring: Stochastic connectivity generator
        - NCPWiring: Neural Circuit Policy specific wiring
        
    Advanced Neurons:
        - LTCNeuronWithAttention: Liquid Time-Constant neuron with attention
        - DopamineModulator: Neuromodulatory system implementation

All components maintain strict backend independence through the ops abstraction.
"""
