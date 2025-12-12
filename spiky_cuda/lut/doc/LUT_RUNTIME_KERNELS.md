# LUT Runtime CUDA Kernels Documentation

This document provides a comprehensive overview of all CUDA kernels used in the LUT (Look-Up Table) runtime system. Each kernel is documented with its launch configuration, input/output data shapes, and functional description.

## Table of Contents

1. [Non-Sequential Forward Pass Kernels](#non-sequential-forward-pass-kernels)
2. [Non-Sequential Backward Pass Kernels](#non-sequential-backward-pass-kernels)
3. [Sequential Forward Pass Kernels](#sequential-forward-pass-kernels)
4. [Sequential Backward Pass Kernels](#sequential-backward-pass-kernels)
5. [Utility Kernels](#utility-kernels)

---

## Non-Sequential Forward Pass Kernels

### 1. `check_detectors_logic`

**Purpose**: Computes lookup indices for detectors by comparing anchor pair differences in the input vector.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size × n_inputs]` - Input activations
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs (AnchorsPair structs)
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector

**Outputs**:
- `w_lookup_indices`: `[batch_size × n_detectors]` - Lookup table indices (int32)
- `w_min_anchor_deltas`: `[batch_size × n_detectors]` - Minimum anchor delta values (float)
- `w_min_anchor_delta_indices`: `[batch_size × n_detectors]` - Indices of minimum delta anchors (int32)

**Description**: Each thread processes one detector for one batch item. For each anchor pair, it computes the difference between anchor values, forms a bit representation (1 if delta > 0, 0 otherwise), and tracks the anchor pair with the minimum absolute delta. The bit representation forms the lookup index into the LUT table.

---

### 2. `check_detectors_eval_logic`

**Purpose**: Computes lookup indices for detectors by comparing anchor pair differences in the input vector (eval mode, no gradient tracking).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size × n_inputs]` - Input activations
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs (AnchorsPair structs)
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector

**Outputs**:
- `w_lookup_indices`: `[batch_size × n_detectors]` - Lookup table indices (int32)

**Description**: Each thread processes one detector for one batch item. For each anchor pair, it computes the difference between anchor values and forms a bit representation (1 if delta > 0, 0 otherwise). The bit representation forms the lookup index into the LUT table. This is the eval mode version that skips tracking minimum anchor deltas and indices (not needed for inference).

---

### 3. `fill_outputs_non_seq_sparse_logic`

**Purpose**: Accumulates outputs from lookup indices using sparse connectivity information (sparse connectivity mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `n_detectors`: Number of detectors
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks for tiling
- `n_outputs_per_block`: Number of outputs per block
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size × n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one detector and one output block. Uses sparse connectivity information to find the forward synapse group for the lookup neuron, then iterates through connected outputs to accumulate weights. Handles both integer and floating point arithmetic modes.

---

### 4. `fill_outputs_non_seq_fc_logic`

**Purpose**: Accumulates outputs from lookup indices (fully connected mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_outputs × n_detector_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_detectors × n_lookup_neurons_per_detector × n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `n_detector_blocks`: Number of detector blocks for tiling
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `backward_group_size`: Size of backward groups
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size × n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one output and one detector block. Directly accesses weights using the lookup index and accumulates to outputs. Handles both integer and floating point arithmetic modes.

---

## Non-Sequential Backward Pass Kernels

### 5. `propagate_through_detectors_non_seq_sparse_logic`

**Purpose**: Propagates gradients through detectors to input gradients using sparse connectivity information.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × n_outputs]` - Output gradients
- `r_weights`: `[n_weights]` - Weight tensor
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `r_min_anchor_deltas`: `[batch_size × n_detectors]` - Minimum anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size × n_detectors]` - Min delta anchor indices
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `n_inputs`: Number of input dimensions
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size × n_inputs]` - Input gradients (accumulated)

**Description**: Each thread processes one detector and one output block. Uses sparse connectivity information to gather output gradients, computes the gradient difference between main and alternative lookup neurons, and propagates gradients to input anchors using the uncertainty function gradient.

---

### 6. `propagate_through_detectors_non_seq_fc_logic`

**Purpose**: Propagates gradients through detectors to input gradients (fully connected mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × n_outputs]` - Output gradients
- `r_weights`: `[n_detectors × n_lookup_neurons_per_detector × n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `r_min_anchor_deltas`: `[batch_size × n_detectors]` - Minimum anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size × n_detectors]` - Min delta anchor indices
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `n_inputs`: Number of input dimensions
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size × n_inputs]` - Input gradients (accumulated)

**Description**: Each thread processes one detector and one output block. Directly accesses weights and output gradients, computes the gradient difference between main and alternative lookup neurons, and propagates gradients to input anchors using the uncertainty function gradient.

---

### 6. `gather_w_gradients_non_seq_sparse_logic`

**Purpose**: Accumulates weight gradients using sparse connectivity information.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `external_lr`: External learning rate. When `external_lr >= 0` (internal mode), gradients are applied directly to weights (`r_weights`) and the learning rate is multiplied by `-external_lr`. When `external_lr < 0` (external mode), gradients are accumulated in `w_weights_gradients` for later use and the synapse meta learning rate is used directly.
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated). When `external_lr >= 0`, this points to `r_weights` for direct weight updates. When `external_lr < 0`, this is the separate gradient buffer.

**Description**: Each thread processes one detector and one output block. Uses sparse connectivity information to find connected outputs, then accumulates `output_gradient × lr` to the corresponding weight gradient locations. The learning rate `lr` is computed as: if `external_lr >= 0`, then `lr = first_synapse_meta_lr × (-external_lr)`, otherwise `lr = first_synapse_meta_lr` (or synapse-specific learning rate from `r_synapse_metas`).

---

### 8. `gather_w_gradients_non_seq_fc_logic`

**Purpose**: Accumulates weight gradients (fully connected mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size × n_detectors]` - Lookup indices
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `external_lr`: External learning rate. When `external_lr >= 0` (internal mode), gradients are applied directly to weights (`r_weights`) and the learning rate is multiplied by `-external_lr`. When `external_lr < 0` (external mode), gradients are accumulated in `w_weights_gradients` for later use and the synapse meta learning rate is used directly.
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_detectors × n_lookup_neurons_per_detector × n_outputs]` - Weight gradients (accumulated). When `external_lr >= 0`, this points to `r_weights` for direct weight updates. When `external_lr < 0`, this is the separate gradient buffer.

**Description**: Each thread processes one detector and one output block. Directly accesses output gradients and accumulates `output_gradient × lr` to the corresponding weight gradient location (determined by detector index and lookup index). The learning rate `lr` is computed as: if `external_lr >= 0`, then `lr = first_synapse_meta_lr × (-external_lr)`, otherwise `lr = first_synapse_meta_lr`.

---

## Sequential Forward Pass Kernels

### 8. `check_detectors_seq_logic`

**Purpose**: Computes lookup indices for detectors in sequence processing mode (Q and K detectors).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size × sequence_length`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size × sequence_length × n_inputs]` - Input activations
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup table indices (int32)
- `w_min_anchor_deltas`: `[batch_size × sequence_length × n_detectors]` - Minimum anchor delta values (float)
- `w_min_anchor_delta_indices`: `[batch_size × sequence_length × n_detectors]` - Indices of minimum delta anchors (int32)

**Description**: Each thread processes one detector for one batch item at one timestep. Similar to `check_detectors_logic` but handles sequence dimension through `blockIdx.y`.

---

### 9. `check_detectors_seq_eval_logic`

**Purpose**: Computes lookup indices for detectors in sequence processing mode (Q and K detectors) for eval mode (no gradient tracking).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size × sequence_length`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size × sequence_length × n_inputs]` - Input activations
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup table indices (int32)

**Description**: Each thread processes one detector for one batch item at one timestep. Similar to `check_detectors_seq_logic` but skips tracking minimum anchor deltas and indices (not needed for inference). This is the eval mode version.

---

### 10. `check_positional_embeddings_logic`

**Purpose**: Computes lookup indices for positional embeddings in sequence processing mode.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `sequence_length - 1`
- Shared Memory: None

**Inputs**:
- `r_positional_embeddings`: `[(sequence_length - 1) × n_detectors × positional_dim]` - Positional embeddings
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup table indices (int32)
- `w_min_deltas`: `[(sequence_length - 1) × n_detectors]` - Minimum delta values (float)
- `w_min_delta_indices`: `[(sequence_length - 1) × n_detectors]` - Indices of minimum deltas (int32)

**Description**: Each thread processes one detector for one timestep pair (i, i+1). Computes lookup indices for positional embeddings by comparing anchor pair differences in the positional embedding vector.

---

### 11. `check_positional_embeddings_eval_logic`

**Purpose**: Computes lookup indices for positional embeddings in sequence processing mode for eval mode (no gradient tracking).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `sequence_length - 1`
- Shared Memory: None

**Inputs**:
- `r_positional_embeddings`: `[(sequence_length - 1) × n_detectors × positional_dim]` - Positional embeddings
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup table indices (int32)

**Description**: Each thread processes one detector for one timestep pair (i, i+1). Computes lookup indices for positional embeddings by comparing anchor pair differences in the positional embedding vector. This is the eval mode version that skips tracking minimum deltas and delta indices (not needed for inference).

---

### 12. `fill_outputs_fully_connected_seq_logic`

**Purpose**: Accumulates outputs from lookup indices for fully connected sequential mode.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n_detectors × n_outputs / blockDim.x)`
- `gridDim.y`: `batch_size × (sequence_length - 1)`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_detectors × 2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup> × n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector = `2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup>`
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size × sequence_length × n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one detector and one output for one timestep pair (i, j) where i < j. For each output position j, accumulates weights over all previous timesteps i from 0 to j-1. Computes concatenated indices from Q, K, and PE lookup indices, then directly accesses weights and accumulates to outputs.

---

### 13. `fill_outputs_sparse_seq_logic`

**Purpose**: Accumulates outputs from lookup indices using sparse connectivity information for sequential mode.

**Launch Grid**:
- `blockDim.x`: `TILE × TILE` (typically 16×16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles × n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size × n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `sequence_length`: Length of sequence
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size × sequence_length × n_outputs]` - Output activations (accumulated in-place)

**Description**: Processes attention pairs (i, j) in a tiled manner where i < j. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Uses sparse connectivity information to find forward synapse groups and accumulate weights for connected outputs. Processes sequence pairs in tiles of size `TILE × TILE` (typically 16×16) for efficient memory access.

---

## Sequential Backward Pass Kernels

### 14. `propagate_through_detectors_seq_sparse_logic`

**Purpose**: Propagates gradients through detectors and positional embeddings for sequential processing using sparse connectivity information.

**Launch Grid**:
- `blockDim.x`: `TILE × TILE` (typically 16×16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles × n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size × n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × sequence_length × n_outputs]` - Output gradients
- `r_weights`: `[n_weights]` - Weight tensor
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_min_anchor_deltas`: `[batch_size × sequence_length × n_detectors]` - Min anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size × sequence_length × n_detectors]` - Min delta indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `r_positional_min_deltas`: `[(sequence_length - 1) × n_detectors]` - PE min deltas
- `r_positional_min_delta_indices`: `[(sequence_length - 1) × n_detectors]` - PE min delta indices
- `n_detectors`: Number of detectors
- `n_total_tiles`: Total number of tiles = `ceil(sequence_length / TILE)²`
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `n_inputs`: Number of input dimensions
- `positional_dim`: Dimension of positional encoding
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size × sequence_length × n_inputs]` - Input gradients (accumulated)
- `w_positional_embeddings_gradients`: `[(sequence_length - 1) × n_detectors × positional_dim]` - Positional embedding gradients (accumulated)

**Description**: Processes attention pairs (i, j) in a tiled manner where i < j. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Uses sparse connectivity information to gather output gradients, computes gradient differences between main and alternative concatenated neurons (flipping Q, K, or PE), and propagates gradients to input anchors (for Q and K) and to positional embeddings (for PE) using the uncertainty function gradient.

---

### 15. `propagate_through_detectors_seq_fc_logic`

**Purpose**: Propagates gradients through detectors and positional embeddings for fully connected sequential processing, computing all gradients directly without sparse connectivity lookups.

**Launch Grid**:
- `blockDim.x`: `TILE × TILE` (typically 16×16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles × n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size × n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × sequence_length × n_outputs]` - Output gradients
- `r_weights`: `[n_detectors × 2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup> × n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_min_anchor_deltas`: `[batch_size × sequence_length × n_detectors]` - Min anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size × sequence_length × n_detectors]` - Min delta indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `r_positional_min_deltas`: `[(sequence_length - 1) × n_detectors]` - PE min deltas
- `r_positional_min_delta_indices`: `[(sequence_length - 1) × n_detectors]` - PE min delta indices
- `n_detectors`: Number of detectors
- `n_total_tiles`: Total number of tiles = `ceil(sequence_length / TILE)²`
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `r_detectors`: `[n_detectors × n_anchors_per_detector]` - Anchor pairs
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector = `2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup>`
- `n_inputs`: Number of input dimensions
- `positional_dim`: Dimension of positional encoding
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size × sequence_length × n_inputs]` - Input gradients (accumulated)
- `w_positional_embeddings_gradients`: `[(sequence_length - 1) × n_detectors × positional_dim]` - Positional embedding gradients (accumulated)

**Description**: Processes attention pairs (i, j) in a tiled manner where i < j. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Directly accesses weights and output gradients (using vectorized reads when available), computes gradient differences between main and alternative concatenated neurons (flipping Q, K, or PE), and propagates gradients to input anchors (for Q and K) and to positional embeddings (for PE) using the uncertainty function gradient. More efficient than the sparse version as it avoids sparse connectivity lookups.

---

### 16. `gather_w_gradients_seq_sparse_logic`

**Purpose**: Accumulates weight gradients from sequential processing using sparse connectivity information.

**Launch Grid**:
- `blockDim.x`: `TILE × TILE` (typically 16×16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles × n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size × n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × sequence_length × n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `n_detectors`: Number of detectors
- `n_total_tiles`: Total number of tiles = `ceil(sequence_length / TILE)²`
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `positional_dim`: Dimension of positional encoding
- `external_lr`: External learning rate. When `external_lr >= 0` (internal mode), gradients are applied directly to weights (`r_weights`) and the learning rate is multiplied by `-external_lr`. When `external_lr < 0` (external mode), gradients are accumulated in `w_weights_gradients` for later use and the synapse meta learning rate is used directly.
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated). When `external_lr >= 0`, this points to `r_weights` for direct weight updates. When `external_lr < 0`, this is the separate gradient buffer.

**Description**: Processes attention pairs (i, j) in a tiled manner where i < j. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Uses sparse connectivity information to find forward synapse groups, then accumulates `output_gradient × lr` to the corresponding weight gradient locations. The learning rate `lr` is computed as: if `external_lr >= 0`, then `lr = first_synapse_meta_lr × (-external_lr)` (or synapse-specific learning rate from `r_synapse_metas` multiplied by `-external_lr`), otherwise `lr = first_synapse_meta_lr` (or synapse-specific learning rate from `r_synapse_metas`).

---

### 17. `gather_w_gradients_seq_fc_logic`

**Purpose**: Accumulates weight gradients for fully connected sequential processing.

**Launch Grid**:
- `blockDim.x`: `TILE × TILE` (typically 16×16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles × n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size × n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size × sequence_length × n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size × sequence_length × n_detectors]` - Q and K lookup indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) × n_detectors]` - PE lookup indices
- `n_detectors`: Number of detectors
- `n_total_tiles`: Total number of tiles = `ceil(sequence_length / TILE)²`
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector = `2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup>`
- `positional_dim`: Dimension of positional encoding
- `external_lr`: External learning rate. When `external_lr >= 0` (internal mode), gradients are applied directly to weights (`r_weights`) and the learning rate is multiplied by `-external_lr`. When `external_lr < 0` (external mode), gradients are accumulated in `w_weights_gradients` for later use and the synapse meta learning rate is used directly.
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_detectors × 2<sup>2 × N<sub>c</sub> + N<sub>pe</sub></sup> × n_outputs]` - Weight gradients (accumulated). When `external_lr >= 0`, this points to `r_weights` for direct weight updates. When `external_lr < 0`, this is the separate gradient buffer.

**Description**: Processes attention pairs (i, j) in a tiled manner where i < j. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Directly accesses output gradients (using vectorized reads when available) and accumulates `output_gradient × lr` to the corresponding weight gradient location (determined by concatenated index and output index). The learning rate `lr` is computed as: if `external_lr >= 0`, then `lr = first_synapse_meta_lr × (-external_lr)`, otherwise `lr = first_synapse_meta_lr`.

---

## Utility Kernels

### 18. `convert_integers_to_floats_logic`

**Purpose**: Converts integer accumulations back to floating point (when using integer arithmetic mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 1024 threads)
- `gridDim.x`: `ceil(n / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `buffer`: `[batch_size × n]` - Integer values (converted in-place)
- `n`: Number of elements to convert
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `buffer`: `[batch_size × n]` - Floating point values (converted in-place)

**Description**: Each thread converts one integer value to floating point by dividing by the denominator and rescaling factor. Only active when `INTEGERS_INSTEAD_OF_FLOATS` is defined.

---

## Helper Functions

The kernels use several helper functions that are defined in `lut_runtime_kernels_logic.proto`:

- `find_forward_synapse_group`: Finds the forward synapse group ID for a given lookup neuron and output block index.
- `accumulate_grad_sum_sparse`: Accumulates gradient sums using sparse connectivity information.
- `accumulate_grad_sum_fc`: Accumulates gradient sums for fully connected mode.
- `accumulate_pair_of_gradient_sums_fc`: Accumulates pairs of gradient sums (main and alternative) for fully connected mode.
- `gather_weight_gradients_sparse`: Gathers weight gradients using sparse connectivity information.
- `gather_weight_gradients_fc`: Gathers weight gradients for fully connected mode.
- `propagate_through_detector`: Propagates gradients through a single detector using the uncertainty function gradient.
- `update_positional_gradients`: Updates positional embedding gradients.

---

## Implementation Notes

- All kernels support both integer and floating point arithmetic modes (controlled by `INTEGERS_INSTEAD_OF_FLOATS`).
- CUDA kernels use atomic operations when `ATOMIC` is defined for thread-safe updates.
- Vectorized memory accesses (using `float4` and `uint4`) are used when available and when data is properly aligned.
- Sequence processing kernels handle the sequence dimension through tiled computation or `blockIdx.y` encoding.
- Sparse connectivity kernels use `ForwardSynapseGroups` to efficiently access only connected outputs.
- Fully connected kernels use direct indexing for better performance when all connections exist.
