# Synthesis of Tailored Architectures (STAR)

*Note: This document combines information from the published STAR paper (Thomas et al., 2024) with implementation details from Liquid AI's production systems. Sections marked with notes indicate where implementation-specific details extend beyond the published research.*

## Overview

Synthesis of Tailored Architectures (STAR) is a framework for automated neural network architecture optimization. STAR combines a novel search space based on the theory of linear input-varying systems with gradient-free evolutionary algorithms to synthesize architectures optimized for multiple objectives such as model quality, parameter efficiency, and inference cache size. In production systems, STAR principles are extended with concepts from Liquid Neural Networks and Liquid Time-Constant Networks to create architectures that are both evolutionarily optimized and dynamically adaptive.

Unlike previous approaches to neural architecture search, STAR provides a comprehensive and well-conditioned search space that enables the discovery of diverse computational patterns beyond manually designed architectures like Transformers and hybrid models.

```
┌─────────────────────────────────────────────────────────────┐
│                  STAR Framework                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ LIV Search Space │ │ STAR Genome │ │ Evolutionary    │
│                  │ │             │ │ Optimization    │
└──────────────────┘ └─────────────┘ └─────────────────┘
```

## Core Principles

1. **Linear Input-Varying Systems (LIVs)**: A generalized framework that encompasses most computational units used in deep learning, including attention variants, linear recurrences, convolutions, and other structured operators. In production systems, LIVs enable a rich search space for synthesizing tailored architectures with diverse computational patterns.

2. **Hierarchical Search Space**: A taxonomized space of computational units and their composition, enabling systematic exploration. This hierarchical organization allows for the integration of specialized computational primitives such as Hyena-style gated convolutions and S4 (Structured State Space) blocks.

3. **Numerical Genome Encoding**: A hierarchical numerical representation of architectures that can be optimized with evolutionary algorithms. This encoding captures both static structural properties and dynamic routing capabilities, enabling the discovery of architectures that are both optimized through evolution and adaptive during execution.

4. **Multi-Objective Optimization**: Capability to optimize for multiple metrics simultaneously (quality, parameter count, inference cache size). Production implementations extend this to include hardware-specific optimizations such as kernel fusion and memory layout optimizations.

5. **Dynamic Computation**: Advanced implementations incorporate dynamic routing between blocks based on Liquid Time-Constant (LTC) equations, which adaptively route information flow based on input characteristics. This is complemented by multi-path processing with weight mixers that combine outputs from different computational paths.

## Mathematical Foundation

### Linear Input-Varying Systems

The class of inputs under consideration are sequences of vectors {x₀, x₁, ..., xₗ} where each element xᵢ is referred to as a token. Each token xᵢ is a real-valued vector in ℝᵈ, represented as xᵢ = (x⁰ᵢ, x¹ᵢ, ..., xᵈ⁻¹ᵢ). The individual components xᵅᵢ of each token are called channels.

LIVs can be expressed in their most general form as:

$$y^{\alpha}_i = \sum_{j \in [\ell]} \sum_{\beta \in [d]} T^{\alpha\beta}_{ij}(x)x^{\beta}_j$$

Where T(x) is a linear operator whose action is determined by the input itself. This framework builds on previous work in structured matrices and efficient sequence modeling, generalizing many computational units:

- **Attention**: $T_{ij} = \sigma(q^{\top}_i k_j)V^{\alpha\beta}$ where $(q_i, k_i) = (\phi(x_i), \psi(x_i))$ (dense attention)
- **Linear Attention**: $T_{ij} = C_iB_j$ (low-rank linear attention)
- **Linear Recurrence**: $T_{ij} = C_iA_{i-1} \cdots A_{j+1}B_j$ (semi-separable linear recurrence)
- **Gated Convolution**: $T_{ij} = C_iK_{i-j}B_j$ (scaled Toeplitz gated convolution)
- **Memoryless System**: $T_{ij} = \sigma(C)$ if $i = j$, 0 otherwise (diagonal memoryless system)

### Structure and Featurization

LIVs are characterized by two key aspects:

1. **Operator Structure**: Defined by token-mixing and channel-mixing structures
   - **Token-mixing structure** ($T^{\alpha\beta} \in \mathbb{R}^{\ell \times \ell}$): Determines how tokens interact for each pair of input/output channels
   - **Channel-mixing structure** ($T_{ij} \in \mathbb{R}^{d \times d}$): Determines how channels interact for each pair of input/output tokens

2. **Featurization**: How feature groups are obtained to modulate the computation
   - **Direct parametrization**: Parameters are learned directly
   - **Reparametrization**: Parameters are computed through intermediate functions
   - **Input-dependent parametrization**: Parameters are computed from the input (e.g., attention)

### Composition

LIVs can be composed in various ways, with production implementations extending beyond basic composition patterns:

1. **Sequential Stacking**: Traditional composition where LIVs are stacked sequentially. In advanced implementations, this is enhanced with dynamic routing capabilities that adaptively determine the flow of information through the network.

2. **Featurizer Sharing**: Sharing featurizer weights between different LIVs
   ```
   T_{ij} = φ(x_i^{(m)}; θ)B_j    S_{ij} = φ(x_i^{(n)}; θ)F_j
   ```
   This mechanism is crucial for parameter efficiency in large-scale models, reducing redundancy while maintaining representational capacity.

3. **Feature Group Sharing**: Directly sharing feature groups between different LIVs
   ```
   T_{ij} = C_iB_j    S_{ij} = E_iB_j
   ```
   Production systems leverage this for efficient key-value cache sharing and other memory optimizations.

4. **Multi-Path Processing**: Advanced implementations support parallel processing paths where the same input is processed through different computational units, with their outputs combined through learned weight mixers. This enhances the model's ability to capture diverse patterns and representations.

5. **Dynamic Routing**: Using Liquid Time-Constant (LTC) equations, blocks can learn which downstream blocks to activate based on input characteristics. This creates an adaptive computation graph where the flow of information is determined dynamically during inference.

## STAR Genome

The STAR genome is a hierarchical numerical representation of architectures:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Backbone Genome                            │
│  21211-31112-21221-32112 (example of a 4-LIV backbone)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Operator Genome                            │
│  Each LIV class (e.g., "2") expands to a 5-number sequence      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Featurizer Genome                           │
│  Defines structure and parameters of feature computation         │
└─────────────────────────────────────────────────────────────────┘
```

### Backbone Genome

The backbone genome represents a set of integer-valued sequences of length five, one for each LIV of the backbone:

1. **LIV class**: Integer summary of operator and featurizer genomes
2. **Featurizer sharing**: Determines weight-sharing structure between featurizers
3. **Featurization sharing strategy**: Defines how featurizer sharing is implemented
4. **Feature group sharing**: Determines which LIVs share feature groups directly
5. **Feature group sharing strategy**: Defines which feature groups are shared

**Example**: `21211-31112-21221-32112` is a backbone genome with 4 LIVs (this example reflects actual implementation patterns observed in Liquid AI models):
- First and third LIVs belong to class "2" and share featurizer weights (sharing group "1")
- Second and fourth LIVs belong to class "3" and share feature groups directly

### Operator Genome

The operator genome is a 5-number sequence that defines the structure of each LIV:

1. **Featurization**: Indicates the specific featurizer class
2. **Token-mixing structure**: Defines the linear token-mixing structure (e.g., dense, low-rank, semi-separable)
3. **Structured sparsity**: Defines any structured sparsity masks (e.g., banded, block-diagonal)
4. **Nonlinearity**: Specifies any nonlinearity applied to the token-mixing structure
5. **Channel-mixing structure**: Defines the LIV's channel-mixing structure

### Featurizer Genome

The featurizer genome defines for each feature group:

1. **Token-mixing structure**: How tokens interact in the featurizer
2. **Channel-mixing structure**: How channels interact in the featurizer
3. **Expansion factor**: Ratio of feature group channel dimension to input channel dimension
4. **Repeat factor**: How many times feature groups are replicated across the channel dimension

## Evolutionary Optimization and Mechanistic Design

STAR employs gradient-free evolutionary algorithms to optimize architectures, building on principles from Mechanistic Architecture Design (MAD):

The evolutionary approach represents a natural progression from MAD's principled hybrid architecture design. While MAD established theoretical foundations for combining different computational primitives based on their mechanistic properties, STAR extends this through automated discovery of optimal combinations and compositions.

### Key Steps

1. **Assessment**: Evaluating the quality of each genome in the population
   - Models are trained for a fixed number of tokens (e.g., 1.3B tokens)
   - Performance is measured on validation data (e.g., perplexity)
   - Efficiency metrics are computed (parameter count, cache size)

2. **Pairing**: Selecting parent genomes through tournament selection
   - k genomes are randomly selected from the population
   - The genome with the highest quality is chosen as a parent

3. **Recombination**: Generating new candidates through k-point crossover
   - Genetic material from two parents is exchanged between k randomly chosen points
   - Default: 2-point crossover

4. **Mutation**: Introducing random mutations to maintain diversity
   - Random mutations are applied with a probability (e.g., 10%)
   - Mutations respect the constraints of the genome position

5. **Repair**: Ensuring valid configurations after mutation and recombination
   - Invalid configurations are detected and repaired
   - Incompatible sharing strategies are fixed by re-sampling

### Evolutionary Algorithms

STAR has been evaluated with multiple evolutionary algorithms:

1. **Non-dominated Sorting Genetic Algorithm II (NSGA-2)**
   - Optimizes for multiple objectives simultaneously
   - Maintains diversity through non-dominated sorting and crowding distance
   - Preferred for multi-objective optimization (e.g., quality and size)

2. **Genetic Algorithm (GA)**
   - Optimizes for a single objective or weighted combination
   - Slightly better performance but less diversity than NSGA-2

3. **Firefly Algorithm (FA)**
   - Less effective for STAR optimization

## Implementation Details

### Model Configuration and Optimization

*Note: As described in the paper, STAR evolutions were performed at 125M-parameter model scale, where backbones contain 24 LIVs at a width of 768 dimensions, with populations of 16 genomes evolved for 18 generations. During each evolution, the depth and width of the backbone remained fixed. All experiments were performed in autoregressive language modeling on the RedPajama dataset at a sequence length of 4096 tokens.*

- **Standard Configuration**: 24 LIVs at a width of 768 dimensions (125M parameters)
- **Scaled Configuration**: 48 LIVs at a width of 2048 dimensions (1B parameters)
- **Population Size**: 16 genomes
- **Evolution Generations**: 18 generations
- **Training**: AdamW optimizer with peak learning rate of 0.0008
- **Batch Size**: 0.25M tokens (125M models), 0.75M tokens (1B models)
- **Sequence Length**: 4096 tokens

Production implementations extend these configurations with additional optimizations:

- **Kernel Fusion**: Combining multiple operations into single optimized kernels to reduce memory bandwidth requirements and improve computational efficiency
- **Memory Layout Optimizations**: Customizing memory layouts to maximize hardware utilization and minimize cache misses
- **Hardware-Specific Acceleration**: Leveraging specialized hardware features such as tensor cores for matrix multiplication and other operations
- **Dynamic Batch Sizing**: Adapting batch sizes based on sequence length and available memory to maximize throughput
- **Multi-Path Computation**: Enabling parallel processing paths with dynamic weight mixing to capture diverse patterns efficiently
### LIV Classes

*Note: The following LIV class designations represent specific implementations that have been optimized for production systems.*


The STAR implementation includes various LIV classes:

1. **Self-Attention Variants (SA)**
   - SA-1: Standard multi-head attention
   - SA-2: Linear attention
   - SA-3: Grouped query attention

2. **Recurrence Variants (Rec)**
   - Rec-1: Semi-separable linear recurrence
   - Rec-2: Parallel scan-based implementation

3. **Convolution Variants (GConv)**
   - GConv-1: Gated short convolutions, building on the principles of local feature extraction
   - GConv-2: Gated long convolutions, extending Hyena's approach to efficient long-range dependencies through structured convolutional patterns

4. **Memoryless Systems (GMemless)**
   - SwiGLU and variants

5. **Structured State Space Variants (S4)**
   - S4-1: Standard S4 blocks with fixed state matrices, building on the structured state space sequence modeling approach
   - S4-2: Liquid S4 blocks with dynamic state adaptation, incorporating principles from Liquid Time-Constant Networks to enable adaptive computation
   - S4-Diff: Differential S4 variants that compute the difference between parallel S4 computations, enhancing gradient flow and representational capacity
   - S4-Monarch: S4 variants that leverage Monarch matrices for efficient computation, combining insights from Monarch Mixer with state space modeling

6. **Differential Variants**
   - SA-1-Diff, SA-2-Diff, SA-3-Diff: Differential attention variants
   - Rec-1-Diff, Rec-2-Diff: Differential recurrence variants
   - GConv-1-Diff, GConv-2-Diff: Differential convolution variants

## Performance Metrics

*Note: The following performance metrics combine results from the published paper with specific measurements from Liquid AI's internal implementation and evaluation.*

### Quality Optimization

*Note: Figure 5.3 in the paper shows the distribution of genome scores during STAR evolution when optimizing for quality, demonstrating how the population evolves toward higher quality (lower perplexity) solutions.*

When optimizing solely for quality, STAR-evolved architectures achieve:

- Reduction in perplexity by 1.0 PPL points compared to initial populations
- Outperformance of parameter-matched Transformer++ and StripedMamba on downstream benchmarks
- Improvements on benchmark averages 2x larger than the improvement of hybrids over Transformers

### Quality and Size Optimization

When optimizing for quality and parameter count, STAR-evolved architectures achieve:

- Reduction in parameter count by up to 13% compared to Transformer++
- Reduction in parameter count by up to 8% compared to StripedMamba
- Maintained or improved performance on downstream benchmarks

### Quality and Cache Optimization

*Note: Figure 5.4 in the paper illustrates genome scores during STAR evolution when optimizing for quality and cache size (left), and shows cache scaling with increasing input sequence length for different models (right). The STAR-evolved architectures achieve significantly smaller cache sizes than Transformers and StripedMamba models while maintaining competitive performance.*

When optimizing for quality and cache size, STAR-evolved architectures achieve:

- Reduction in cache size by up to 90% compared to Transformers
- Reduction in cache size by up to 37% compared to StripedMamba
- Maintained competitive performance on downstream benchmarks

### Scaling Performance

When scaling from 125M to 1B parameters, STAR-evolved architectures:

- Match or outperform parameter-matched Transformer++ and StripedMamba models
- Maintain cache size advantages (90% smaller than Transformers, 37% smaller than StripedMamba)
- Show consistent performance across benchmarks including ARC-Challenge

## Emerging Architectural Motifs

*Note: The following architectural patterns are derived from actual evolutionary runs as shown in Figure 5.5 of the paper, which tracks the evolution of LIV types, connectivity patterns, and distances between connected LIVs across generations.*

Through evolutionary optimization, STAR discovers effective architectural motifs:

1. **Preference for Specific LIV Classes**
   - Gated short convolutions (GConv-1)
   - Grouped query attention variants (SA-3)
   - Differential variants of input-varying recurrences (Rec-1-Diff)
   - SwiGLUs (GMemless)

2. **Effective Composition Patterns**
   - Strategic featurizer sharing between distant LIVs
   - Feature group sharing between specific LIV types
   - Alternating patterns of different LIV classes

## Applications

STAR has been primarily evaluated on autoregressive language modeling, where it has demonstrated improvements over highly-optimized Transformers and striped hybrid models on the frontier of quality, parameter size, and inference cache.

### Benchmark Performance

STAR-evolved architectures have been evaluated on:

- HellaSwag: Testing commonsense reasoning
- ARC-Easy and ARC-Challenge: Testing scientific reasoning
- Winogrande: Testing pronoun resolution
- PiQA: Testing physical commonsense
- SciQ: Testing scientific knowledge

## Advantages Over Traditional Approaches

### 1. Comprehensive Search Space

Unlike previous approaches that focus on narrow aspects of architecture design, STAR provides a unified framework that encompasses a wide range of computational units and composition patterns.

### 2. Well-Conditioned Optimization

The STAR genome and evolutionary process are designed to ensure that most sampled candidates train without instabilities, making the optimization process more efficient.

### 3. Multi-Objective Optimization

STAR can optimize for multiple objectives simultaneously, enabling the discovery of architectures that balance quality and efficiency metrics.

### 4. Automated Motif Discovery

Through evolutionary optimization, STAR can automatically discover effective architectural motifs that drive performance improvements.

## Conclusion

Synthesis of Tailored Architectures (STAR) represents a significant advancement in automated architecture optimization. By combining a novel search space based on linear input-varying systems with evolutionary algorithms, STAR enables the systematic discovery of architectures that push the quality-efficiency frontier beyond what is possible with manual design.

The framework's ability to optimize for multiple objectives simultaneously and its well-conditioned search space make it a powerful tool for discovering architectures tailored to specific requirements, whether prioritizing predictive quality, parameter efficiency, or inference efficiency.

In production systems like Liquid Foundation Models, STAR's principles are extended with additional innovations:

1. **Dynamic Computation**: Incorporating Liquid Time-Constant (LTC) equations for adaptive routing between blocks, creating architectures that are both optimized through evolution and adaptive during execution.

2. **Multi-Path Processing**: Enabling parallel computational paths with learned weight mixers, enhancing the model's ability to capture diverse patterns and representations.

3. **Specialized Primitives**: Integrating advanced computational units like liquid S4 blocks alongside traditional LIVs, expanding the range of patterns the model can efficiently process.

4. **Hardware Optimization**: Employing kernel fusion, memory layout optimizations, and hardware-specific acceleration to maximize computational efficiency and scalability.

These extensions create a comprehensive architecture that represents the state-of-the-art in efficient, high-quality sequence modeling, demonstrating how the theoretical foundations of STAR can be realized in practical, production-ready systems.

## References

1. Thomas, A. W., Parnichkun, R., Amini, A., Massaroli, S., & Poli, M. (2024). STAR: Synthesis of Tailored Architectures. Liquid AI.
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems.
3. Shazeer, N. (2020). GLU variants improve Transformer. arXiv preprint arXiv:2002.05202.
4. Poli, M., Thomas, A. W., Nguyen, E., Ponnusamy, P., Deiseroth, B., Kersting, K., Suzuki, T., Hie, B., Ermon, S., Ré, C., et al. (2024). Mechanistic design and scaling of hybrid architectures. arXiv preprint arXiv:2403.17844.
5. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). RoFormer: Enhanced Transformer with rotary position embedding. Neurocomputing.
6. Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., & Ré, C. (2023). Zoology: Measuring and improving recall in efficient language models. arXiv preprint arXiv:2312.04927.
7. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces.
8. Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated linear attention transformers with hardware-efficient training.
9. So, D., Mańke, W., Liu, H., Dai, Z., Shazeer, N., & Le, Q. V. (2021). Searching for efficient transformers for language modeling. Advances in neural information processing systems, 34, 6010-6022.
10. Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150.
11. Poli, M., Massaroli, S., Nguyen, E., Fu, D. Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S., & Ré, C. (2023). Hyena hierarchy: Towards larger convolutional language models.
---


## Liquid AI Implementation Details

*Note: The following implementation details represent specific architectural choices made by Liquid AI that extend beyond the published research paper.*


## Implementation Details and Practical Answers for Liquid Foundation Models

This section provides concrete answers to the key implementation questions for building a Liquid Foundation Model (LIV-based architecture) using the STAR framework.

### Operator Instantiation

- **Parameterization and Initialization:** Each LIV operator (attention, recurrence, convolution) is instantiated in code using a specific class definition that encapsulates its unique structure and behavior. For example, attention variants are instantiated using SA-1, SA-2, SA-3 classes, each with their own set of parameters. Initialization typically follows standard practices, such as Glorot or He initialization.
- **Token-Mixing and Channel-Mixing Structures:** Valid choices for these structures are defined by the operator's class and can include dense, low-rank, semi-separable, or structured sparsity for token-mixing; and various channel-mixing strategies depending on the operator. The STAR document outlines the allowed structures for each LIV class.

### Featurizer and Feature Group Sharing

- **Implementation:** Featurizer sharing is implemented by sharing weights between different LIVs, often defined by the LIV Genome. Dimensionality and compatibility constraints are outlined in the Featurizer Genome, ensuring that shared parameters are compatible across LIVs.
- **Strategies for Feature Group Sharing:** The document mentions direct parametrization, reparametrization, and input-dependent parametrization for sharing feature groups. Shared parameters are updated during training through backpropagation.

### Genome Encoding and Decoding

- **Mapping:** The hierarchical genome is mapped to a neural network module graph by translating the integer-valued sequences (backbone, operator, featurizer) into a directed acyclic graph (DAG) representation of the computation. This process involves creating a neural network module for each LIV and connecting them according to the encoded genome.
- **Translation to PyTorch/TF:** The genome sequence is decoded into a sequence of computational units and their connections, which are then implemented using PyTorch or TensorFlow operations.

### Mutation and Repair

- **Mutation Operations:** Allowed mutation operations are tailored to each genome segment (backbone, operator, featurizer) and ensure that the resulting configuration remains valid. Invalid configurations are detected through constraints checks and repaired by re-sampling or re-initializing the affected segment.
- **Diversity Maintenance:** Diversity is maintained through the evolutionary process, ensuring a wide exploration of the search space without sacrificing trainability. This is achieved through techniques such as tournament selection and non-dominated sorting.

### Training Protocols

- **Optimizer Settings:** Recommended settings include AdamW optimizer with peak learning rate of 0.0008, as specified in the STAR document.
- **Learning Rate Schedules and Regularization:** These are typically tuned based on the specific objectives and can include learning rate decays, weight decay, and dropout regularization.
- **Unstable/Degenerate Architecture Detection:** Such architectures are detected during the training phase through performance metrics and heuristics, then filtered out of the population.

### Evaluation and Selection

- **Metrics:** Multi-objective optimization uses metrics such as perplexity, parameter count, and cache size. Trade-offs are managed through Pareto front analysis.
- **Solution Diversity Enforcement:** Diversity is enforced through the evolutionary process, ensuring a non-dominated set of solutions that balance multiple objectives.

### Scalability and Hardware

- **Constraints for Large Models:** There may be architectural constraints for scaling to 1B+ parameters, such as memory constraints and computational requirements.
- **Hardware-Specific Optimizations:** Kernel fusion, memory layout optimizations, and other techniques are crucial for efficient inference and training on hardware platforms like GPUs.

### Reference Implementations

- **Canonical Code Examples:** Canonical code examples for each LIV class and composition strategy can be found in the STAR codebase, following best practices for integration with deep learning frameworks.
- **Custom Layers Integration:** Custom layers are integrated using standard interfaces provided by PyTorch or TensorFlow, ensuring compatibility with existing models and architectures.

---

This comprehensive breakdown addresses the open questions regarding LIV-based architectures based on the STAR framework, providing insights into their instantiation, optimization, and application. It is intended as a practical reference for building a Liquid Foundation Model using the STAR framework.