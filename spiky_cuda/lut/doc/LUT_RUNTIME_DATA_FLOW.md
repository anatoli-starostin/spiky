# LUT Runtime Data Flow

This document describes the data flow for both non-sequential (single timestep) and sequential (sequence processing) modes in the LUT runtime system.

## Non-Sequential Mode - Forward Pass

### Fully Connected Mode

```mermaid
flowchart TD
    A["Input<br/>[B × I]"] --> B(check_detectors)
    
    B --> C["Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    
    C --> F(fill_outputs_fully_connected)
    
    F --> G["Output: Output<br/>[B × O]"]
    
    style B fill:#81c784,color:#000000
    style F fill:#81c784,color:#000000
    style C fill:#ffffff,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
```

### Sparse Connectivity Mode

```mermaid
flowchart TD
    A["Input<br/>[B × I]"] --> B(fire_detectors)
    
    B --> C["Output: Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Output: Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Output: Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    B --> F["Firing Events<br/>[B × N<sub>t</sub> × max_fw_groups]"]
    
    F --> G(fill_outputs_by_forward_groups)
    
    G --> H["Output: Output<br/>[B × O]"]
    
    style B fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style C fill:#ffffff,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
```

## Non-Sequential Mode - Backward Pass

### Fully Connected Mode

```mermaid
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B("gather_x_gradients_fully_connected<br/>Main - Stream 0")
    A --> C("gather_x_gradients_fully_connected<br/>Alternative - Stream 1")
    A --> D("gather_w_gradients_fully_connected<br/>Stream 2")
    
    B --> E["Before Detectors Gradients<br/>[B × N<sub>t</sub> × (1 << N<sub>c</sub>)]"]
    C --> E
    D --> F["Output: Weight Gradients<br/>[n_weights]"]
    
    E --> G(propagate_through_detectors)
    G --> H["Output: Input Gradients<br/>[B × I]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style F fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
```

### Sparse Connectivity Mode

```mermaid
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B(fire_detectors_by_lookup_indices)
    
    B --> C["Firing Events<br/>Main&nbsp;+&nbsp;Alternative<br/>[B&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;max_fw_groups&nbsp;×&nbsp;2]"]
    
    C --> D(gather_gradients)
    
    D --> E["Before Detectors Gradients<br/>[B × N<sub>t</sub> × (1 << N<sub>c</sub>)]"]
    D --> F["Output: Weight Gradients<br/>[n_weights]"]
    
    E --> G(propagate_through_detectors)
    G --> H["Output: Input Gradients<br/>[B × I]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style F fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
```

## Sequential Mode - Forward Pass

```mermaid
flowchart TD
    A["Input<br/>[B × S × I]"] --> B("check_detectors_for_sequence<br/>Stream 0")
    C["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> D("check_positional_embeddings<br/>Stream 1")
    
    B --> E["Output: Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"]
    B --> F["Output: Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"]
    B --> G["Output: Q/K Min Anchor Delta Indices<br/>[B × S × N<sub>t</sub>]"]
    
    D --> H["Output: PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"]
    D --> I["Output: PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"]
    D --> J["Output: PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"]
    
    E --> K("fill_after_detectors_firing_stat<br/>(&nbsp;processes&nbsp;B&nbsp;×&nbsp;S&nbsp;×&nbsp;(&nbsp;S&minus;1&nbsp;)&nbsp;pairs&nbsp;with&nbsp;tiles&nbsp;)")
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L["Firing Statistics<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))]"]
    
    L --> M(densify_firing_stat)
    
    M --> N["Output: Main Firing Events<br/>[B × N<sub>t</sub> × S × (S-1) / 2]"]
    M --> O["Output: Alternative Firing Events<br/>[B × N<sub>t</sub> × S × (S-1)]"]
    
    N --> P(fill_outputs_by_sparse_firings)
    
    P --> Q["Output: Output<br/>[B × S × O]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style K fill:#81c784,color:#000000
    style M fill:#81c784,color:#000000
    style P fill:#81c784,color:#000000
    style E fill:#ffffff,color:#000000
    style F fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style I fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style N fill:#ffffff,color:#000000
    style O fill:#ffffff,color:#000000
    style Q fill:#ffffff,color:#000000
```

## Sequential Mode - Backward Pass

```mermaid
flowchart TD
    A["Output Gradients<br/>[B × S × O]"] --> B("gather_x_gradients_for_sequence<br/>Main - Stream 0")
    A --> C("gather_w_gradients_for_sequence<br/>Stream 1")
    A --> D("gather_x_gradients_for_sequence<br/>Alternative - Stream 2")
    
    E["Main Firing Events<br/>[B × N<sub>t</sub> × S × (S-1) / 2]"] --> B
    E --> C
    F["Alternative Firing Events<br/>[B × N<sub>t</sub> × S × (S-1)]"] --> D
    
    B --> G["Before Detectors Gradients<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))]"]
    D --> G
    
    G --> I("propagate_through_detectors_for_sequence<br/>(processes B×S×(S-1) pairs with tiles)")
    
    I --> J["Output: Input Gradients<br/>[B × S × I]"]
    I --> K["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    
    C --> H["Output: Weight Gradients<br/>[n_weights]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style I fill:#81c784,color:#000000
    style H fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style K fill:#ffffff,color:#000000
```
