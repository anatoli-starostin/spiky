# LUT Runtime CUDA Kernels Documentation

This document provides a comprehensive overview of all CUDA kernels used in the LUT (Look-Up Table) runtime system. Each kernel is documented with its launch configuration, input/output data shapes, and functional description.

## Table of Contents

1. [Forward Pass Kernels](#forward-pass-kernels)
2. [Backward Pass Kernels](#backward-pass-kernels)
3. [Sequence Processing Kernels](#sequence-processing-kernels)
4. [Utility Kernels](#utility-kernels)

---

## Forward Pass Kernels

### 1. `check_detectors_logic`

**Purpose**: Computes lookup indices for detectors by comparing anchor pair differences in the input vector.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size * n_inputs]` - Input activations
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs (AnchorsPair structs)
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector

**Outputs**:
- `w_lookup_indices`: `[batch_size * n_detectors]` - Lookup table indices (int32)
- `w_min_anchor_deltas`: `[batch_size * n_detectors]` - Minimum anchor delta values (float)
- `w_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Indices of minimum delta anchors (int32)

**Description**: Each thread processes one detector for one batch item. For each anchor pair, it computes the difference between anchor values, forms a bit representation (1 if delta > 0, 0 otherwise), and tracks the anchor pair with the minimum absolute delta. The bit representation forms the lookup index into the LUT table.

---

### 2. `fire_detectors_logic`

**Purpose**: Similar to `check_detectors_logic`, but also generates firing events for sparse connectivity mode.

**Launch Grid**:
- `blockDim.x`: Power of 2 (typically 256-512 threads, rounded up)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: `blockDim.x * sizeof(uint32_t)` (for atomic reduction)

**Inputs**:
- Same as `check_detectors_logic`
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_detectors * n_lookup_neurons_per_detector]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron
- `synapse_group_size`: Size of synapse groups
- `lut_data`: Pointer to LUT data structure
- `device`: Device ID (-1 for CPU)

**Outputs**:
- Same as `check_detectors_logic`
- `rw_firings_buffer`: `[max_firings]` - Sparse firing events (Firing structs)
- `rw_firings_counter_ptr`: Pointer to firing counter (uint64)

**Description**: Performs the same detector checking as `check_detectors_logic`, but additionally generates firing events for sparse connectivity. Uses shared memory reduction for efficient atomic counter updates when running on GPU.

---

### 3. `fill_outputs_by_forward_groups_logic`

**Purpose**: Accumulates outputs from sparse firing events by processing synapse groups.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(max_firings / blockDim.x)`
- `gridDim.y`: 1
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `first_synapse_id`: First synapse ID offset
- `r_firings_buffer`: `[max_firings]` - Firing events from forward pass
- `r_firings_counter_ptr`: Pointer to firing counter
- `max_firings`: Maximum number of firings
- `n_lookup_neurons`: Total number of lookup neurons
- `n_outputs`: Number of output dimensions
- `lut_data`: Pointer to LUT data structure
- `int_rescaler`: Integer rescaling factor (if using integer arithmetic)

**Outputs**:
- `w_output`: `[batch_size * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one firing event. For each firing, it loads the corresponding synapse group, iterates through target neurons and weights, and accumulates weighted contributions to the output. Uses vectorized loads (uint4/float4) for efficiency.

---

### 4. `fill_outputs_fully_connected_logic`

**Purpose**: Computes outputs for fully connected mode by directly accessing weight tables.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_outputs * n_detector_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_detectors * n_lookup_neurons_per_detector * n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices from detectors
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `n_detector_blocks`: Number of detector blocks for tiling
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `n_detectors_per_block`: Number of detectors per block
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one output dimension for one detector block. It iterates through detectors in the block, reads lookup indices, accesses corresponding weights from the weight table, and accumulates the sum. Uses vectorized loads (int4) for efficient lookup index reading.

---

## Backward Pass Kernels

### 5. `fire_detectors_by_lookup_indices_logic`

**Purpose**: Generates firing events for both main and alternative lookup indices (for gradient computation).

**Launch Grid**:
- `blockDim.x`: Power of 2 (typically 256-512 threads, rounded up)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: `blockDim.x * sizeof(uint32_t)` (for atomic reduction)

**Inputs**:
- `n_detectors`: Number of detectors
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices from forward pass
- `r_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Indices of minimum delta anchors
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_detectors * n_lookup_neurons_per_detector]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron
- `synapse_group_size`: Size of synapse groups
- `lut_data`: Pointer to LUT data structure
- `device`: Device ID

**Outputs**:
- `rw_firings_buffer`: `[max_firings]` - Firing events (main and alternative)
- `rw_firings_counter_ptr`: Pointer to firing counter

**Description**: For each detector, generates firing events for both the main lookup index and the alternative (flipped bit) lookup index. Main firings have payload 1.0, alternative firings have payload -1.0. Uses shared memory reduction for efficient counter updates.

---

### 6. `gather_gradients_logic`

**Purpose**: Gathers gradients for lookup neurons and accumulates weight gradients from output gradients.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(max_firings / blockDim.x)`
- `gridDim.y`: 1
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `first_synapse_id`: First synapse ID offset
- `r_firings`: `[max_firings]` - Firing events
- `r_firings_counter_ptr`: Pointer to firing counter
- `max_firings`: Maximum number of firings
- `r_output_gradients`: `[batch_size * n_outputs]` - Output gradients
- `n_lookup_neurons`: Total number of lookup neurons
- `n_outputs`: Number of output dimensions
- `lut_data`: Pointer to LUT data structure
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `external_lr`: External learning rate
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_before_detectors_gradients`: `[batch_size * n_lookup_neurons]` - Gradients before detectors (accumulated)
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated, if not in internal mode)

**Description**: Each thread processes one firing event. For each synapse group in the firing, it computes the gradient contribution by multiplying weights with output gradients. Accumulates gradients to lookup neurons and optionally to weights (depending on gradient policy). Handles both sparse connectivity and internal/external gradient modes.

---

### 7. `gather_x_gradients_fully_connected_logic`

**Purpose**: Gathers input gradients for fully connected mode.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors * n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_detectors * n_lookup_neurons_per_detector * n_outputs]` - Weight tensor
- `r_output_gradients`: `[batch_size * n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices
- `r_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Min delta anchor indices (optional)
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks for tiling
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `lr`: Learning rate
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_before_detectors_gradients`: `[batch_size * n_detectors * n_lookup_neurons_per_detector]` - Gradients before detectors (accumulated)

**Description**: Each thread processes one detector and one output block. It computes the gradient by summing `weight * output_gradient` over all outputs in the block, then accumulates to the appropriate lookup neuron (using main or alternative index based on `r_min_anchor_delta_indices`).

---

### 8. `gather_w_gradients_fully_connected_logic`

**Purpose**: Accumulates weight gradients for fully connected mode.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors * n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size * n_outputs]` - Output gradients
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `lr`: Learning rate
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_detectors * n_lookup_neurons_per_detector * n_outputs]` - Weight gradients (accumulated)

**Description**: Each thread processes one detector and one output block. For each output in the block, it accumulates `output_gradient * lr` to the corresponding weight gradient location (determined by detector index and lookup index).

---

### 9. `propagate_through_detectors_logic`

**Purpose**: Propagates gradients through detectors to input gradients using the gradient of the up() function.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices
- `r_min_anchor_deltas`: `[batch_size * n_detectors]` - Minimum anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Min delta anchor indices
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `rw_before_detectors_gradients`: `[batch_size * n_detectors * n_lookup_neurons_per_detector]` - Gradients before detectors
- `n_inputs`: Number of input dimensions
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size * n_inputs]` - Input gradients (accumulated)
- `rw_before_detectors_gradients`: Cleared for main and alternative lookup neurons (in-place)

**Description**: Each thread processes one detector. It computes the gradient difference between the main and alternative lookup neurons, applies the gradient of the up() function (0.5 * sign(delta) / (1 + |delta|)^2), and propagates to the input anchors. Clears the gradient buffers after processing.

---

## Sequence Processing Kernels

### 10. `check_detectors_for_sequence_logic`

**Purpose**: Computes lookup indices for detectors across a sequence of timesteps.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(sequence_length * n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size * sequence_length * n_inputs]` - Input activations across sequence
- `n_inputs`: Number of input dimensions
- `sequence_length`: Length of sequence
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector

**Outputs**:
- `w_lookup_indices`: `[batch_size * sequence_length * n_detectors]` - Lookup indices
- `w_min_anchor_deltas`: `[batch_size * sequence_length * n_detectors]` - Min anchor deltas
- `w_min_anchor_delta_indices`: `[batch_size * sequence_length * n_detectors]` - Min delta anchor indices

**Description**: Each thread processes one detector at one timestep. Similar to `check_detectors_logic` but handles the sequence dimension. Thread index encodes both timestep and detector index.

---

### 11. `check_positional_embeddings_logic`

**Purpose**: Computes lookup indices for positional embeddings.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil((sequence_length - 1) * n_detectors / blockDim.x)`
- `gridDim.y`: 1 (positional embeddings are not batched)
- Shared Memory: None

**Inputs**:
- `sequence_length`: Length of sequence
- `r_positional_embeddings`: `[(sequence_length - 1) * n_detectors * positional_dim]` - Positional embeddings
- `n_detectors`: Number of detectors
- `positional_dim`: Dimension of positional encoding

**Outputs**:
- `w_positional_lookup_indices`: `[(sequence_length - 1) * n_detectors]` - Positional lookup indices
- `w_positional_min_deltas`: `[(sequence_length - 1) * n_detectors]` - Min positional deltas
- `w_positional_min_delta_indices`: `[(sequence_length - 1) * n_detectors]` - Min delta dimension indices

**Description**: Each thread processes one detector at one relative position (timestep difference). Forms a bit representation based on sign of positional embedding values and tracks the dimension with minimum absolute value.

---

### 12. `fill_after_detectors_firing_stat_logic`

**Purpose**: Builds firing statistics for attention mechanism by concatenating Q, K, and PE lookup indices.

**Launch Grid**:
- `blockDim.x`: `TILE * TILE` (typically 16x16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles * n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_lookup_indices`: `[batch_size * sequence_length * n_detectors]` - Q and K lookup indices
- `r_min_anchor_deltas`: `[batch_size * sequence_length * n_detectors]` - Min anchor deltas (optional)
- `r_min_anchor_delta_indices`: `[batch_size * sequence_length * n_detectors]` - Min delta indices (optional)
- `r_positional_lookup_indices`: `[(sequence_length - 1) * n_detectors]` - PE lookup indices
- `r_positional_min_deltas`: `[(sequence_length - 1) * n_detectors]` - PE min deltas (optional)
- `r_positional_min_delta_indices`: `[(sequence_length - 1) * n_detectors]` - PE min delta indices (optional)
- `n_total_tiles`: Total number of tiles in the attention matrix
- `sequence_length`: Length of sequence
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `n_lookup_neurons`: Total number of lookup neurons
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector

**Outputs**:
- `w_firing_stat`: `[batch_size * sequence_length * n_lookup_neurons]` - Firing statistics (accumulated)

**Description**: Processes attention pairs (i, j) where i < j in a tiled manner. For each valid pair, concatenates Q (from timestep j), K (from timestep i), and PE (from relative position j-i-1) to form a concatenated lookup index. Increments firing statistics for the main concatenated neuron and optionally for alternative neurons (flipping Q, K, or PE based on minimum deltas).

---

### 13. `densify_firing_stat_logic`

**Purpose**: Converts dense firing statistics into sparse firing events.

**Launch Grid**:
- `blockDim.x`: Power of 2 (typically 256-512 threads, rounded up)
- `gridDim.x`: `ceil(n_lookup_neurons / blockDim.x)`
- `gridDim.y`: `batch_size * sequence_length`
- Shared Memory: `blockDim.x * 2 * sizeof(uint32_t)` (for dual counter reduction)

**Inputs**:
- `rw_firing_stat`: `[batch_size * sequence_length * n_lookup_neurons]` - Firing statistics
- `n_lookup_neurons`: Total number of lookup neurons
- `sequence_length`: Length of sequence
- `device`: Device ID

**Outputs**:
- `rw_firings_buffer`: `[max_firings]` - Main firing events (NeuronShiftFiring structs)
- `rw_firings_counter_ptr`: Pointer to main firing counter
- `rw_firings_buffer_alternative`: `[max_firings]` - Alternative firing events (optional)
- `rw_firings_counter_ptr_alternative`: Pointer to alternative firing counter (optional)
- `rw_firing_stat`: Cleared after processing (in-place)

**Description**: Each thread processes one lookup neuron at one timestep. If firing_stat >= 1.0, creates a main firing event. If 0 < firing_stat < 1.0, creates an alternative firing event (for gradient computation). Uses shared memory reduction for efficient dual counter updates.

---

### 14. `fill_outputs_by_sparse_firings_logic`

**Purpose**: Accumulates outputs from sparse firing events for sequence processing.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(max_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `first_synapse_id`: First synapse ID offset
- `r_firings_buffer`: `[max_firings]` - Sparse firing events (NeuronShiftFiring)
- `r_firings_counter_ptr`: Pointer to firing counter
- `max_firings`: Maximum number of firings
- `n_lookup_neurons`: Total number of lookup neurons
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks
- `sequence_length`: Length of sequence
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron (optional, nullptr for fully connected)
- `n_outputs_per_block`: Number of outputs per block
- `lut_data`: Pointer to LUT data structure
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size * sequence_length * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one firing event for one output block. For sparse connectivity, processes synapse groups. For fully connected mode, directly accesses weights and accumulates to outputs. Handles sequence dimension through firing.shift field.

---

### 15. `gather_x_gradients_for_sequence_logic`

**Purpose**: Gathers input gradients from sparse firings for sequence processing.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_sparse_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `r_output_gradients`: `[batch_size * sequence_length * n_outputs]` - Output gradients
- `r_sparse_firings`: `[n_sparse_firings]` - Sparse firing events
- `n_sparse_firings`: Number of sparse firings
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `sequence_length`: Length of sequence
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron (optional)
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_before_detectors_gradients`: `[batch_size * sequence_length * n_detectors * n_lookup_neurons_per_detector]` - Gradients before detectors (accumulated)

**Description**: Each thread processes one firing event for one output block. Computes gradient by summing `weight * output_gradient` over outputs in the block, then accumulates to the lookup neuron (accounting for sequence shift via firing.shift).

---

### 16. `gather_w_gradients_for_sequence_logic`

**Purpose**: Accumulates weight gradients from sparse firings for sequence processing.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_sparse_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size * sequence_length * n_outputs]` - Output gradients
- `r_sparse_firings`: `[n_sparse_firings]` - Sparse firing events
- `n_sparse_firings`: Number of sparse firings
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `sequence_length`: Length of sequence
- `n_output_blocks`: Number of output blocks
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron (optional)
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `external_lr`: External learning rate
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated)

**Description**: Each thread processes one firing event for one output block. For each output in the block, accumulates `output_gradient * firing.payload * lr` to the corresponding weight gradient location.

---

### 17. `propagate_through_detectors_for_sequence_logic`

**Purpose**: Propagates gradients through detectors and positional embeddings for sequence processing.

**Launch Grid**:
- `blockDim.x`: `TILE * TILE` (typically 16x16 = 256 threads per tile)
- `gridDim.x`: `n_total_tiles * n_detectors` (one block per tile per detector)
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_lookup_indices`: `[batch_size * sequence_length * n_detectors]` - Q and K lookup indices
- `r_min_anchor_deltas`: `[batch_size * sequence_length * n_detectors]` - Min anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size * sequence_length * n_detectors]` - Min delta indices
- `r_positional_lookup_indices`: `[(sequence_length - 1) * n_detectors]` - PE lookup indices
- `r_positional_min_deltas`: `[(sequence_length - 1) * n_detectors]` - PE min deltas
- `r_positional_min_delta_indices`: `[(sequence_length - 1) * n_detectors]` - PE min delta indices
- `n_detectors`: Number of detectors
- `n_total_tiles`: Total number of tiles
- `sequence_length`: Length of sequence
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `w_before_detectors_gradients`: `[batch_size * sequence_length * n_detectors * n_lookup_neurons_per_detector]` - Gradients before detectors
- `n_inputs`: Number of input dimensions
- `positional_dim`: Dimension of positional encoding
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size * sequence_length * n_inputs]` - Input gradients (accumulated)
- `w_positional_embeddings_gradients`: `[(sequence_length - 1) * n_detectors * positional_dim]` - Positional embedding gradients (accumulated)
- `w_before_detectors_gradients`: Cleared for processed neurons (in-place)

**Description**: Processes attention pairs (i, j) in a tiled manner. For each valid pair, computes gradient differences between main and alternative concatenated neurons (flipping Q, K, or PE). Propagates gradients to input anchors (for Q and K) and to positional embeddings (for PE) using the up() function gradient. Clears gradient buffers after processing.

---

### 18. `cleanup_x_gradients_for_sequence_logic`

**Purpose**: Clears gradient buffers for processed sparse firing neurons.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_sparse_firings / blockDim.x)`
- `gridDim.y`: 1
- Shared Memory: None

**Inputs**:
- `r_sparse_firings`: `[n_sparse_firings]` - Sparse firing events
- `n_sparse_firings`: Number of sparse firings
- `n_detectors`: Number of detectors
- `sequence_length`: Length of sequence
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector

**Outputs**:
- `w_before_detectors_gradients`: `[batch_size * sequence_length * n_detectors * n_lookup_neurons_per_detector]` - Cleared for processed neurons (in-place)

**Description**: Each thread processes one sparse firing event and clears the corresponding gradient buffer entry. Used for cleanup after gradient propagation.

---

## Utility Kernels

### 19. `convert_integers_to_floats_logic`

**Purpose**: Converts integer accumulations back to floating point (when using integer arithmetic mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `buffer`: `[batch_size * n]` - Integer buffer (in-place conversion)
- `n`: Number of elements per batch item
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `buffer`: `[batch_size * n]` - Floating point values (converted in-place)

**Description**: Each thread converts one integer value to floating point by dividing by the denominator and rescaling factor. Only active when `INTEGERS_INSTEAD_OF_FLOATS` is defined.

---

## Helper Functions

### `process_single_synapse_group`

**Purpose**: Processes a single synapse group, loading targets and weights, and accumulating to outputs.

**Not a kernel**: This is a helper function called from within kernels.

**Inputs**:
- `group_id`: Synapse group ID
- `lut_data`: Pointer to LUT data structure
- `first_synapse_id`: First synapse ID offset
- `r_weights`: `[n_weights]` - Weight tensor
- `n_lookup_neurons`: Total number of lookup neurons
- `multiplier`: Multiplier for weights (typically 1.0 or firing.payload)
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[n_outputs]` - Output activations (accumulated in-place)

**Description**: Loads synapse group metadata, iterates through target neurons and weights using vectorized loads, and accumulates weighted contributions to outputs. Handles both integer and floating point arithmetic modes.

