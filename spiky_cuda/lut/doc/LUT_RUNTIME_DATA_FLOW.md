# LUT Runtime Data Flow

This document describes the data flow for both non-concatenated (single timestep) and concatenated (sequence processing) modes in the LUT runtime system.

## Visual Data Flow Diagrams

### Non-Concatenated Mode - Forward Pass

```mermaid
flowchart TD
    A[Input<br/>batch_size × n_inputs] --> B{Connectivity<br/>Mode?}
    B -->|Sparse| C[fire_detectors<br/>Kernel]
    B -->|Fully Connected| D[check_detectors<br/>Kernel]
    
    C --> E[Lookup Indices<br/>Min Deltas<br/>batch_size × n_detectors]
    C --> F[Firing Events<br/>max_firings]
    
    D --> E
    
    E --> G{Mode?}
    F --> H[fill_outputs_by_forward_groups<br/>Kernel]
    G -->|Fully Connected| I[fill_outputs_fully_connected<br/>Kernel]
    
    H --> J[Output<br/>batch_size × n_outputs]
    I --> J
    
    J --> K{Integer<br/>Mode?}
    K -->|Yes| L[convert_integers_to_floats<br/>Kernel]
    K -->|No| M[Final Output]
    L --> M
```

### Non-Concatenated Mode - Backward Pass

```mermaid
flowchart TD
    A[Output Gradients<br/>batch_size × n_outputs] --> B{Connectivity<br/>Mode?}
    
    B -->|Sparse| C[fire_detectors_by_lookup_indices<br/>Kernel]
    C --> D[Firing Events<br/>Main + Alternative<br/>max_firings]
    D --> E[gather_gradients<br/>Kernel]
    
    B -->|Fully Connected| F[gather_x_gradients_fully_connected<br/>Main - Stream 0]
    B -->|Fully Connected| G[gather_x_gradients_fully_connected<br/>Alternative - Stream 1]
    B -->|Fully Connected| H[gather_w_gradients_fully_connected<br/>Stream 2]
    
    E --> I[Before Detectors Gradients<br/>batch_size × n_lookup_neurons]
    E --> J[Weight Gradients<br/>n_weights]
    
    F --> I
    G --> I
    H --> J
    
    I --> K[propagate_through_detectors<br/>Kernel]
    K --> L[Input Gradients<br/>batch_size × n_inputs]
    
    L --> M{Integer<br/>Mode?}
    M -->|Yes| N[convert_integers_to_floats<br/>Kernel]
    M -->|No| O[Final Input Gradients]
    N --> O
```

### Concatenated Mode - Forward Pass

```mermaid
flowchart TD
    A[Input<br/>batch_size × seq_len × n_inputs] --> B[check_detectors_for_sequence<br/>Stream 0]
    C[Positional Embeddings<br/>seq_len-1 × n_detectors × pos_dim] --> D[check_positional_embeddings<br/>Stream 1]
    
    B --> E[Q/K Lookup Indices<br/>batch_size × seq_len × n_detectors]
    B --> F[Q/K Min Deltas & Indices]
    
    D --> G[PE Lookup Indices<br/>seq_len-1 × n_detectors]
    D --> H[PE Min Deltas & Indices]
    
    E --> I[fill_after_detectors_firing_stat<br/>Kernel<br/>Tiled Processing]
    F --> I
    G --> I
    H --> I
    
    I --> J[Firing Statistics<br/>batch_size × seq_len × n_lookup_neurons]
    
    J --> K[densify_firing_stat<br/>Kernel]
    
    K --> L[Main Firing Events<br/>n_firings]
    K --> M[Alternative Firing Events<br/>n_alternative_firings]
    
    L --> N[fill_outputs_by_sparse_firings<br/>Kernel]
    M --> N
    
    N --> O[Output<br/>batch_size × seq_len × n_outputs]
    
    O --> P{Integer<br/>Mode?}
    P -->|Yes| Q[convert_integers_to_floats<br/>Kernel]
    P -->|No| R[Final Output]
    Q --> R
```

### Concatenated Mode - Backward Pass

```mermaid
flowchart TD
    A[Output Gradients<br/>batch_size × seq_len × n_outputs] --> B[gather_x_gradients_for_sequence<br/>Main - Stream 0]
    A --> C[gather_w_gradients_for_sequence<br/>Stream 1]
    A --> D[gather_x_gradients_for_sequence<br/>Alternative - Stream 2]
    
    E[Main Firing Events<br/>n_firings] --> B
    E --> C
    F[Alternative Firing Events<br/>n_alternative_firings] --> D
    
    B --> G[Before Detectors Gradients<br/>batch_size × seq_len ×<br/>n_detectors × n_lookup_neurons_per_detector]
    C --> H[Weight Gradients<br/>n_weights]
    D --> G
    
    G --> I[propagate_through_detectors_for_sequence<br/>Kernel<br/>Tiled Processing]
    
    I --> J[Input Gradients<br/>batch_size × seq_len × n_inputs]
    I --> K[Positional Embedding Gradients<br/>seq_len-1 × n_detectors × pos_dim]
    
    G --> L[cleanup_x_gradients_for_sequence<br/>Main - Stream 0]
    G --> M[cleanup_x_gradients_for_sequence<br/>Alternative - Stream 1]
    
    L --> N[Cleared Gradients]
    M --> N
    
    J --> O{Integer<br/>Mode?}
    O -->|Yes| P[convert_integers_to_floats<br/>Kernel]
    O -->|No| Q[Final Input Gradients]
    P --> Q
```

### Concatenated Mode - Attention Pair Processing

```mermaid
flowchart LR
    A[Timestep i<br/>Input] --> B[K Lookup Index]
    C[Timestep j<br/>Input] --> D[Q Lookup Index]
    E[Relative Position<br/>j-i-1] --> F[PE Lookup Index]
    
    B --> G[Concatenate<br/>Q, K, PE]
    D --> G
    F --> G
    
    G --> H[Concatenated Index<br/>Q << n_anchors+pos_dim<br/>K << pos_dim<br/>PE]
    
    H --> I[Lookup Neuron<br/>Weight Table]
    I --> J[Output at<br/>Timestep j]
    
    style G fill:#e1f5ff
    style H fill:#fff4e1
    style I fill:#e8f5e9
```

## Non-Concatenated Mode (`sequence_length == 1`)

### Forward Pass

```
Input [batch_size × n_inputs]
  ↓
[check_detectors / fire_detectors]
  ↓
Lookup Indices [batch_size × n_detectors]
Min Anchor Deltas [batch_size × n_detectors]
Min Anchor Delta Indices [batch_size × n_detectors]
  ↓
┌─────────────────────────────────────┐
│  Sparse Connectivity Mode           │
│  (if lookup_neuron_synapses_infos)  │
└─────────────────────────────────────┘
  ↓
Firing Events [max_firings]
  ↓
[fill_outputs_by_forward_groups]
  ↓
Output [batch_size × n_outputs]

┌─────────────────────────────────────┐
│  Fully Connected Mode               │
│  (if !lookup_neuron_synapses_infos) │
└─────────────────────────────────────┘
  ↓
[fill_outputs_fully_connected]
  ↓
Output [batch_size × n_outputs]
```

**Kernel Sequence:**
1. **`check_detectors`** (fully connected) or **`fire_detectors`** (sparse)
   - Computes lookup indices from input anchor comparisons
   - Outputs: `lookup_indices`, `min_anchor_deltas`, `min_anchor_delta_indices`

2. **`fill_outputs_by_forward_groups`** (sparse) or **`fill_outputs_fully_connected`** (fully connected)
   - Accumulates outputs from lookup table weights
   - Sparse: processes firing events through synapse groups
   - Fully connected: directly accesses weight tables using lookup indices

3. **`convert_integers_to_floats`** (if using integer arithmetic)
   - Converts accumulated integer values to floating point

---

### Backward Pass

```
Output Gradients [batch_size × n_outputs]
  ↓
┌─────────────────────────────────────┐
│  Sparse Connectivity Mode           │
└─────────────────────────────────────┘
  ↓
[fire_detectors_by_lookup_indices]
  ↓
Firing Events (main + alternative) [max_firings]
  ↓
[gather_gradients]
  ↓
Before Detectors Gradients [batch_size × n_lookup_neurons]
Weight Gradients [n_weights] (optional)

┌─────────────────────────────────────┐
│  Fully Connected Mode               │
└─────────────────────────────────────┘
  ↓
[gather_x_gradients_fully_connected] (main)
[gather_x_gradients_fully_connected] (alternative, parallel)
[gather_w_gradients_fully_connected] (parallel)
  ↓
Before Detectors Gradients [batch_size × n_detectors × n_lookup_neurons_per_detector]
Weight Gradients [n_weights] (optional)
  ↓
[propagate_through_detectors]
  ↓
Input Gradients [batch_size × n_inputs]
```

**Kernel Sequence:**
1. **Gradient Gathering:**
   - **Sparse**: `fire_detectors_by_lookup_indices` → `gather_gradients`
     - Generates firing events for main and alternative lookup indices
     - Accumulates gradients from output gradients through synapse groups
   - **Fully Connected**: Three parallel kernels
     - `gather_x_gradients_fully_connected` (main lookup index)
     - `gather_x_gradients_fully_connected` (alternative lookup index, stream 1)
     - `gather_w_gradients_fully_connected` (weight gradients, stream 2)

2. **`propagate_through_detectors`**
   - Computes gradient difference between main and alternative lookup neurons
   - Applies up() function gradient: `0.5 * sign(delta) / (1 + |delta|)^2`
   - Propagates to input anchors (adds to anchor1, subtracts from anchor2)
   - Clears gradient buffers

3. **`convert_integers_to_floats`** (if using integer arithmetic)
   - Converts integer gradients to floating point

---

## Concatenated Mode (`sequence_length > 1`)

### Forward Pass

```
Input [batch_size × sequence_length × n_inputs]
Positional Embeddings [(sequence_length - 1) × n_detectors × positional_dim]
  ↓
[check_detectors_for_sequence] (stream 0)
[check_positional_embeddings] (stream 1, parallel)
  ↓
Q/K Lookup Indices [batch_size × sequence_length × n_detectors]
PE Lookup Indices [(sequence_length - 1) × n_detectors]
Min Deltas & Indices (for Q, K, PE)
  ↓
[fill_after_detectors_firing_stat]
  ↓
Firing Statistics [batch_size × sequence_length × n_lookup_neurons]
  ↓
[densify_firing_stat]
  ↓
Sparse Firing Events [n_firings]
Alternative Firing Events [n_alternative_firings] (optional)
  ↓
[fill_outputs_by_sparse_firings]
  ↓
Output [batch_size × sequence_length × n_outputs]
```

**Kernel Sequence:**
1. **`check_detectors_for_sequence`** (stream 0)
   - Computes Q and K lookup indices for all timesteps
   - Outputs: `lookup_indices`, `min_anchor_deltas`, `min_anchor_delta_indices`

2. **`check_positional_embeddings`** (stream 1, parallel)
   - Computes PE lookup indices for relative positions
   - Outputs: `positional_lookup_indices`, `positional_min_deltas`, `positional_min_delta_indices`

3. **`fill_after_detectors_firing_stat`**
   - Processes attention pairs (i, j) where i < j in tiled manner
   - Concatenates Q (from timestep j), K (from timestep i), PE (from j-i-1)
   - Forms concatenated lookup index: `(Q << (n_anchors + positional_dim)) | (K << positional_dim) | PE`
   - Increments firing statistics for main and alternative concatenated neurons

4. **`densify_firing_stat`**
   - Converts dense firing statistics to sparse firing events
   - Main firings: `firing_stat >= 1.0`
   - Alternative firings: `0 < firing_stat < 1.0` (for gradient computation)

5. **`fill_outputs_by_sparse_firings`**
   - Processes sparse firing events
   - Accumulates outputs using weights from lookup neurons
   - Handles sequence dimension through `firing.shift` field

6. **`convert_integers_to_floats`** (if using integer arithmetic)

---

### Backward Pass

```
Output Gradients [batch_size × sequence_length × n_outputs]
Sparse Firing Events [n_firings]
Alternative Firing Events [n_alternative_firings]
  ↓
[gather_x_gradients_for_sequence] (main, stream 0)
[gather_w_gradients_for_sequence] (stream 1, parallel)
[gather_x_gradients_for_sequence] (alternative, stream 2, parallel)
  ↓
Before Detectors Gradients [batch_size × sequence_length × n_detectors × n_lookup_neurons_per_detector]
Weight Gradients [n_weights] (optional)
  ↓
[propagate_through_detectors_for_sequence]
  ↓
Input Gradients [batch_size × sequence_length × n_inputs]
Positional Embedding Gradients [(sequence_length - 1) × n_detectors × positional_dim]
  ↓
[cleanup_x_gradients_for_sequence] (main + alternative, parallel)
```

**Kernel Sequence:**
1. **Gradient Gathering (three parallel streams):**
   - **Stream 0**: `gather_x_gradients_for_sequence` (main firings)
     - Accumulates gradients to lookup neurons from main firing events
   - **Stream 1**: `gather_w_gradients_for_sequence` (main firings)
     - Accumulates weight gradients from main firing events
   - **Stream 2**: `gather_x_gradients_for_sequence` (alternative firings)
     - Accumulates gradients to lookup neurons from alternative firing events

2. **`propagate_through_detectors_for_sequence`**
   - Processes attention pairs (i, j) in tiled manner
   - For each pair, computes gradient differences between:
     - Main concatenated neuron: `(Q, K, PE)`
     - Alternative concatenated neurons: `(Q', K, PE)`, `(Q, K', PE)`, `(Q, K, PE')`
   - Propagates gradients to:
     - Input anchors (for Q and K components)
     - Positional embeddings (for PE component)
   - Uses up() function gradient for each component

3. **`cleanup_x_gradients_for_sequence`** (two parallel streams)
   - Clears gradient buffers for processed neurons
   - Stream 0: main firings
   - Stream 1: alternative firings

4. **`convert_integers_to_floats`** (if using integer arithmetic)

---

## Key Differences

### Non-Concatenated Mode
- **Single timestep**: `sequence_length == 1`
- **Simple lookup**: Direct mapping from input to output via lookup table
- **Two connectivity modes**: Sparse (via synapse groups) or Fully Connected (direct weight access)
- **Backward**: Single gradient path through detectors

### Concatenated Mode
- **Multiple timesteps**: `sequence_length > 1`
- **Attention mechanism**: Processes all pairs (i, j) where i < j
- **Three components**: Q (query from timestep j), K (key from timestep i), PE (positional encoding)
- **Concatenated lookup**: `(Q << (n_anchors + positional_dim)) | (K << positional_dim) | PE`
- **Tiled processing**: Uses TILE×TILE blocks for efficient attention matrix processing
- **Backward**: Three gradient paths (Q, K, PE) propagated separately
- **Parallel streams**: Uses multiple CUDA streams for concurrent processing

---

## Data Structures

### Firing Events
- **Non-concatenated**: `Firing { batch_index, payload, data_id }`
- **Concatenated**: `NeuronShiftFiring { batch_index, payload, neuron_id, shift }`
  - `shift` field encodes the timestep for sequence processing

### Lookup Indices
- **Non-concatenated**: Single lookup index per detector
- **Concatenated**: 
  - Q/K indices: `[batch_size × sequence_length × n_detectors]`
  - PE indices: `[(sequence_length - 1) × n_detectors]`
  - Concatenated index: `(Q << (n_anchors + positional_dim)) | (K << positional_dim) | PE`

### Gradient Buffers
- **Before detectors**: Gradients accumulated at lookup neurons before propagation through detectors
- **After propagation**: Gradients propagated to inputs/positional embeddings via up() function

