# LUT Runtime CUDA Kernels Documentation

This document provides a comprehensive overview of all CUDA kernels used in the LUT (Look-Up Table) runtime system. Each kernel is documented with its launch configuration, input/output data shapes, and functional description.

## Table of Contents

1. [Forward Pass Kernels](#forward-pass-kernels)
2. [Backward Pass Kernels](#backward-pass-kernels)
3. [Sequence Processing Kernels - Forward](#sequence-processing-kernels---forward)
4. [Sequence Processing Kernels - Backward](#sequence-processing-kernels---backward)
5. [Utility Kernels](#utility-kernels)

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
- `r_input`: `[batch_size * n_inputs]` - Input activations
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures containing synapse connectivity info for each lookup neuron
- `synapse_group_size`: Size of synapse groups
- `lut_data`: Pointer to LUT data structure
- `device`: Device ID

**Outputs**:
- `w_lookup_indices`: `[batch_size * n_detectors]` - Lookup table indices (int32)
- `w_min_anchor_deltas`: `[batch_size * n_detectors]` - Minimum anchor delta values (float)
- `w_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Indices of minimum delta anchors (int32)
- `rw_firings_buffer`: `[max_firings]` - Firing events (Firing structs)
- `rw_firings_counter_ptr`: Pointer to firing counter

**Description**: Each thread processes one detector for one batch item. Computes lookup indices similar to `check_detectors_logic`, and also generates firing events for sparse connectivity mode. Uses shared memory reduction for efficient counter updates.

---

### 3. `fill_outputs_by_forward_groups_logic`

**Purpose**: Accumulates outputs from firing events using forward synapse groups (sparse connectivity mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `r_firings`: `[n_firings]` - Firing events
- `r_firings_counter_ptr`: Pointer to firing counter
- `n_outputs`: Number of output dimensions
- `n_output_blocks`: Number of output blocks for tiling
- `n_outputs_per_block`: Number of outputs per block
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one firing event for one output block. For sparse connectivity, processes synapse groups using the `process_single_synapse_group` helper function. Handles both integer and floating point arithmetic modes.

---

### 4. `fill_outputs_fully_connected_logic`

**Purpose**: Accumulates outputs from lookup indices (fully connected mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors * n_output_blocks / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_detectors * n_lookup_neurons_per_detector * n_outputs]` - Weight tensor
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices
- `n_outputs`: Number of output dimensions
- `n_detectors`: Number of detectors
- `n_output_blocks`: Number of output blocks for tiling
- `n_outputs_per_block`: Number of outputs per block
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_output`: `[batch_size * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one detector and one output block. Directly accesses weights using the lookup index and accumulates to outputs. Handles both integer and floating point arithmetic modes.

---

## Backward Pass Kernels

### 5. `fire_detectors_by_lookup_indices_logic`

**Purpose**: Generates firing events from lookup indices for backward pass (sparse connectivity mode).

**Launch Grid**:
- `blockDim.x`: Power of 2 (typically 256-512 threads, rounded up)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: `blockDim.x * sizeof(uint32_t)` (for atomic reduction)

**Inputs**:
- `r_lookup_indices`: `[batch_size * n_detectors]` - Lookup indices
- `r_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Min delta anchor indices
- `n_detectors`: Number of detectors
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `synapse_group_size`: Size of synapse groups
- `lut_data`: Pointer to LUT data structure
- `device`: Device ID

**Outputs**:
- `rw_firings_buffer`: `[max_firings]` - Main firing events (Firing structs)
- `rw_firings_counter_ptr`: Pointer to main firing counter
- `rw_firings_buffer_alternative`: `[max_firings]` - Alternative firing events
- `rw_firings_counter_ptr_alternative`: Pointer to alternative firing counter

**Description**: Each thread processes one detector for one batch item. Generates both main and alternative firing events based on lookup indices and minimum anchor delta indices. Uses shared memory reduction for efficient counter updates.

---

### 6. `gather_gradients_logic`

**Purpose**: Gathers input and weight gradients from firing events (sparse connectivity mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_output_gradients`: `[batch_size * n_outputs]` - Output gradients
- `r_firings`: `[n_firings]` - Firing events
- `r_firings_counter_ptr`: Pointer to firing counter
- `w_before_detectors_gradients`: `[batch_size * 2 * n_detectors]` - Gradients before detectors (accumulated)
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated, optional)
- `n_lookup_neurons`: Total number of lookup neurons
- `n_detectors`: Number of detectors
- `n_outputs`: Number of output dimensions
- `r_lookup_neuron_synapses_infos`: `[n_lookup_neurons]` - Array of `NoDelaysIndexedSynapsesInfo` structures
- `r_synapse_metas`: `[n_synapse_metas]` - Synapse metadata
- `first_synapse_id`: First synapse ID offset
- `lut_data`: Pointer to LUT data structure
- `first_synapse_meta_lr`: Learning rate for first synapse meta
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_before_detectors_gradients`: `[batch_size * 2 * n_detectors]` - Gradients before detectors (accumulated)
- `w_weights_gradients`: `[n_weights]` - Weight gradients (accumulated, if not in internal mode)

**Description**: Each thread processes one firing event for one output block. For each synapse group in the firing, it computes the gradient contribution by multiplying weights with output gradients. Accumulates gradients to lookup neurons (main and alternative) and optionally to weights (depending on gradient policy). Handles both sparse connectivity and internal/external gradient modes.

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
- `w_before_detectors_gradients`: `[batch_size * 2 * n_detectors]` - Gradients before detectors (accumulated)

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
- `r_min_anchor_deltas`: `[batch_size * n_detectors]` - Minimum anchor deltas
- `r_min_anchor_delta_indices`: `[batch_size * n_detectors]` - Min delta anchor indices
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `rw_before_detectors_gradients`: `[batch_size * 2 * n_detectors]` - Gradients before detectors
- `n_inputs`: Number of input dimensions
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size * n_inputs]` - Input gradients (accumulated)
- `rw_before_detectors_gradients`: Cleared for main and alternative lookup neurons (in-place)

**Description**: Each thread processes one detector. It computes the gradient difference between the main and alternative lookup neurons, applies the gradient of the up() function (0.5 * sign(delta) / (1 + |delta|)^2), and propagates to the input anchors. Clears the gradient buffers after processing.

---

## Sequence Processing Kernels - Forward

### 10. `check_detectors_for_sequence_logic`

**Purpose**: Computes lookup indices for detectors in sequence processing mode (Q and K detectors).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `batch_size * sequence_length`
- Shared Memory: None

**Inputs**:
- `r_input`: `[batch_size * sequence_length * n_inputs]` - Input activations
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_inputs`: Number of input dimensions
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[batch_size * sequence_length * n_detectors]` - Q and K lookup table indices (int32)
- `w_min_anchor_deltas`: `[batch_size * sequence_length * n_detectors]` - Minimum anchor delta values (float)
- `w_min_anchor_delta_indices`: `[batch_size * sequence_length * n_detectors]` - Indices of minimum delta anchors (int32)

**Description**: Each thread processes one detector for one batch item at one timestep. Similar to `check_detectors_logic` but handles sequence dimension through `blockIdx.y`.

---

### 11. `check_positional_embeddings_logic`

**Purpose**: Computes lookup indices for positional embeddings in sequence processing mode.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_detectors / blockDim.x)`
- `gridDim.y`: `sequence_length - 1`
- Shared Memory: None

**Inputs**:
- `r_positional_embeddings`: `[(sequence_length - 1) * n_detectors * positional_dim]` - Positional embeddings
- `r_detectors`: `[n_detectors * n_anchors_per_detector]` - Anchor pairs
- `n_detectors`: Number of detectors
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `sequence_length`: Length of sequence

**Outputs**:
- `w_lookup_indices`: `[(sequence_length - 1) * n_detectors]` - PE lookup table indices (int32)
- `w_min_deltas`: `[(sequence_length - 1) * n_detectors]` - Minimum delta values (float)
- `w_min_delta_indices`: `[(sequence_length - 1) * n_detectors]` - Indices of minimum deltas (int32)

**Description**: Each thread processes one detector for one timestep pair (i, i+1). Computes lookup indices for positional embeddings by comparing anchor pair differences in the positional embedding vector.

---

### 12. `fill_after_detectors_firing_stat_logic`

**Purpose**: Fills firing statistics by processing attention pairs (i, j) in a tiled manner.

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
- `n_anchors_per_detector`: Number of anchor pairs per detector
- `positional_dim`: Dimension of positional encoding
- `n_lookup_neurons_per_detector`: Number of lookup neurons per detector
- `rw_firings_buffer`: `[max_firings]` - Main firing events buffer
- `rw_firings_counter_ptr`: Pointer to main firing counter
- `rw_firings_buffer_alternative`: `[max_firings]` - Alternative firing events buffer
- `rw_firings_counter_ptr_alternative`: Pointer to alternative firing counter
- `device`: Device ID

**Outputs**:
- `rw_firings_buffer`: `[max_firings]` - Main firing events (NeuronShiftFiring structs)
- `rw_firings_counter_ptr`: Updated firing counter
- `rw_firings_buffer_alternative`: `[max_firings]` - Alternative firing events (optional)
- `rw_firings_counter_ptr_alternative`: Updated alternative firing counter

**Description**: Processes attention pairs (i, j) in a tiled manner. For each valid pair, computes concatenated indices from Q, K, and PE lookup indices. Generates main firing events (when Q/K/PE combinations are valid) and alternative firing events (when flipping Q, K, or PE would change the result). Uses tiling for efficient processing of sequence pairs.

---

### 13. `densify_firing_stat_cpu_logic` and `densify_firing_stat_cuda_logic`

**Purpose**: Converts dense firing statistics into sparse firing events. Split into separate CPU and CUDA kernels for optimal performance on each platform.

**Launch Grid**:
- **CPU version** (`densify_firing_stat_cpu_logic`):
  - `blockDim.x`: Configurable (typically 256-512 threads)
  - `gridDim.x`: `ceil(n_lookup_neurons / blockDim.x)`
  - `gridDim.y`: `batch_size * sequence_length`
  - Shared Memory: None
- **CUDA version** (`densify_firing_stat_cuda_logic`):
  - `blockDim.x`: Power of 2 (typically 256-512 threads, rounded up)
  - `gridDim.x`: `ceil(n_quads / (blockDim.x / 4))` (processes 4 neurons per thread)
  - `gridDim.y`: `batch_size * sequence_length`
  - Shared Memory: `blockDim.x * 2 * sizeof(uint32_t)` (for dual counter reduction)

**Inputs**:
- `rw_firing_stat`: `[batch_size * sequence_length * n_lookup_neurons]` - Firing statistics
  - CPU version: `EXTERNAL_REAL_DT*` (scalar access)
  - CUDA version: `float4*` (vectorized access, 4 neurons per element)
- `n_lookup_neurons`: Total number of lookup neurons
- `n_quads`: Number of quads (4-neuron groups) - CUDA version only, equals `n_lookup_neurons / 4`
- `sequence_length`: Length of sequence

**Outputs**:
- `rw_firings_buffer`: `[max_firings]` - Main firing events (NeuronShiftFiring structs)
- `rw_firings_counter_ptr`: Pointer to main firing counter
- `rw_firings_buffer_alternative`: `[max_firings]` - Alternative firing events (optional)
- `rw_firings_counter_ptr_alternative`: Pointer to alternative firing counter (optional)
- `rw_firing_stat`: Cleared after processing (in-place)

**Description**: 
- **CPU version**: Each thread processes one lookup neuron at one timestep sequentially. If firing_stat >= 1.0, creates a main firing event. If 0 < firing_stat < 1.0, creates an alternative firing event (for gradient computation).
- **CUDA version**: Each thread processes 4 neurons at once (quads) using `float4` vectorized loads. Uses warp-level prefix scans (`__shfl_up_sync`) for efficient dual counter updates. Out-of-bounds quads are treated as zeros to maintain warp coherence. Uses shared memory reduction for efficient dual counter updates.

---

### 14. `fill_outputs_by_sparse_firings_logic`

**Purpose**: Accumulates outputs from sparse firing events for sequence processing.

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n_sparse_firings / blockDim.x)`
- `gridDim.y`: `n_output_blocks`
- Shared Memory: None

**Inputs**:
- `r_weights`: `[n_weights]` - Weight tensor
- `r_sparse_firings`: `[n_sparse_firings]` - Sparse firing events (NeuronShiftFiring structs)
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
- `w_output`: `[batch_size * sequence_length * n_outputs]` - Output activations (accumulated in-place)

**Description**: Each thread processes one firing event for one output block. For sparse connectivity, processes synapse groups. For fully connected mode, directly accesses weights and accumulates to outputs. Handles sequence dimension through firing.shift field.

---

## Sequence Processing Kernels - Backward

### 15. `gather_x_gradients_for_sequence_logic`

**Purpose**: Gathers input gradients from sparse firings for sequence processing using a hash table.

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
- `rw_before_detectors_gradients`: `[batch_size * 6 * n_detectors * sequence_length * (sequence_length - 1)]` - Hash table of `GradientHashInfo` entries for gradients before detectors
- `gradient_hash_width`: Width of the hash table per batch = `6 × n_detectors × sequence_length × (sequence_length - 1)`
- `is_alternative`: Boolean flag indicating whether processing alternative firings (true) or main firings (false)

**Outputs**:
- `rw_before_detectors_gradients`: `[batch_size * 6 * n_detectors * sequence_length * (sequence_length - 1)]` - Hash table of `GradientHashInfo` entries containing gradients before detectors (accumulated in-place)

**Description**: Each thread processes one firing event for one output block. Computes gradient by summing `weight * output_gradient` over outputs in the block, then inserts/updates the gradient in a hash table keyed by `(neuron_id, timestep)` with `firing_id` as a filter.

**Hash Table Details**:
- Each `GradientHashInfo` entry contains: `neuron_id` (stored as `neuron_id + 1` to distinguish empty slots), `timestep`, `firing_id` (positive for main firings, negative for alternative firings), and `gradient_value` (the accumulated gradient).
- Hash function: `(neuron_id + timestep) % (6 × n_detectors × sequence_length × (sequence_length - 1))`
- Collision resolution: Linear probing
- Empty slots are identified by `neuron_id == 0` (since stored values are `neuron_id + 1`)
- On CUDA path: Uses `atomicCAS` for thread-safe slot claiming and `atomicAdd` for gradient accumulation
- On CPU path: Direct memory writes (non-thread-safe, single-threaded execution)
- The hash table is cleared at the beginning of the backward pass using `memsetAsync` (CUDA) or `memset` (CPU)

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

**Purpose**: Propagates gradients through detectors and positional embeddings for sequence processing using hash table lookups.

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
- `w_before_detectors_gradients`: `[batch_size * 6 * n_detectors * sequence_length * (sequence_length - 1)]` - Hash table of `GradientHashInfo` entries containing gradients before detectors
- `gradient_hash_width`: Width of the hash table per batch = `6 × n_detectors × sequence_length × (sequence_length - 1)`
- `n_inputs`: Number of input dimensions
- `positional_dim`: Dimension of positional encoding
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `w_input_gradients`: `[batch_size * sequence_length * n_inputs]` - Input gradients (accumulated)
- `w_positional_embeddings_gradients`: `[(sequence_length - 1) * n_detectors * positional_dim]` - Positional embedding gradients (accumulated)

**Description**: Processes attention pairs (i, j) in a tiled manner. For each valid pair, looks up gradients from the hash table using `(neuron_id, timestep)` keys for both main and alternative concatenated neurons (flipping Q, K, or PE). Computes gradient differences and propagates gradients to input anchors (for Q and K) and to positional embeddings (for PE) using the up() function gradient.

**Hash Table Lookup**:
- Uses the same hash function and linear probing as in `gather_x_gradients_for_sequence_logic`
- Looks up gradient values for both main and alternative neurons
- If an entry is not found (empty slot), the gradient value is treated as zero
- The hash table is cleared at the beginning of the backward pass using `memsetAsync` (CUDA) or `memset` (CPU)

---

## Utility Kernels

### 18. `convert_integers_to_floats_logic`

**Purpose**: Converts integer accumulations back to floating point (when using integer arithmetic mode).

**Launch Grid**:
- `blockDim.x`: Configurable (typically 256-512 threads)
- `gridDim.x`: `ceil(n / blockDim.x)`
- `gridDim.y`: `batch_size`
- Shared Memory: None

**Inputs**:
- `buffer`: `[batch_size * n]` - Integer values (converted in-place)
- `n`: Number of elements to convert
- `int_rescaler`: Integer rescaling factor

**Outputs**:
- `buffer`: `[batch_size * n]` - Floating point values (converted in-place)

**Description**: Each thread converts one integer value to floating point by dividing by the denominator and rescaling factor. Only active when `INTEGERS_INSTEAD_OF_FLOATS` is defined.

---

## Data Structures

### `GradientHashInfo`

Structure used for hash table entries in sequence backward pass:

```c++
typedef struct alignas(8) {
    NeuronIndex_t neuron_id;    // Stored as neuron_id + 1 (0 indicates empty slot)
    uint32_t timestep;          // Timestep (firing.shift)
    int32_t firing_id;          // Positive for main firings, negative for alternative
    SUMMATION32_DT gradient_value;  // Accumulated gradient value
} GradientHashInfo;
```

- Size: 16 bytes (aligned to 8 bytes)
- Hash key construction: `(neuron_id + 1) | (timestep << 32)` (little-endian layout)
- Hash index: `(neuron_id + timestep) % (6 × n_detectors × sequence_length × (sequence_length - 1))`

---

## Notes

- All kernels support both integer and floating point arithmetic modes (controlled by `INTEGERS_INSTEAD_OF_FLOATS`).
- CUDA kernels use atomic operations when `ATOMIC` is defined for thread-safe updates.
- The hash table implementation uses linear probing for collision resolution.
- Sequence processing kernels handle the sequence dimension through `blockIdx.y` or `firing.shift` fields.
- The `densify_firing_stat` kernels are split into CPU and CUDA versions for optimal performance on each platform.
