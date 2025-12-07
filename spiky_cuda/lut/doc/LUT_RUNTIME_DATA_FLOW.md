# LUT Runtime Data Flow

## Notation

- **B**: Batch size
- **I**: Number of input dimensions
- **O**: Number of output dimensions
- **S**: Sequence length
- **N<sub>t</sub>**: Number of detectors
- **N<sub>c</sub>**: Number of anchor pairs per detector (determines lookup table size: 2<sup>N<sub>c</sub></sup>)
- **N<sub>pe</sub>**: Positional embedding dimension

## Non-Sequential Mode - Forward Pass

### Fully Connected Case

```mermaid
flowchart TD
    A["Input<br/>[B × I]"] --> B(check_detectors)
    D1["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> B
    
    B --> C["Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    
    W1["Weights<br/>[N<sub>t</sub> × (1 << N<sub>c</sub>) × O]"] --> F(fill_outputs_fully_connected)
    C --> F
    
    F --> G["Output: Output<br/>[B × O]"]
    
    style B fill:#81c784,color:#000000
    style F fill:#81c784,color:#000000
    style C fill:#ffffff,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style D1 fill:#000000,color:#ffffff
    style W1 fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
flowchart TD
    A["Input<br/>[B × I]"] --> B(fire_detectors)
    D2["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> B
    SC1["Sparse connectivity info"] --> B
    
    B --> C["Output: Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Output: Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Output: Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    B --> F["Firing Events<br/>[B × N<sub>t</sub> × max_fw_groups]"]
    
    W2["Weights<br/>[N<sub>t</sub> × (1 << N<sub>c</sub>) × O]"] --> G(fill_outputs_by_forward_groups)
    SC1 --> G
    F --> G
    
    G --> H["Output: Output<br/>[B × O]"]
    
    style B fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style C fill:#ffffff,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style D2 fill:#000000,color:#ffffff
    style SC1 fill:#000000,color:#ffffff
    style W2 fill:#000000,color:#ffffff
    style F fill:#000000,color:#ffffff
```

## Non-Sequential Mode - Backward Pass

### Fully Connected Case

```mermaid
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B("gather_x_gradients_fully_connected<br/>Main - Stream 0")
    A --> C("gather_x_gradients_fully_connected<br/>Alternative - Stream 1")
    A --> D("gather_w_gradients_fully_connected<br/>Stream 2")
    
    W3["Weights<br/>[N<sub>t</sub> × (1 << N<sub>c</sub>) × O]"] --> B
    W3 --> C
    
    L1["Lookup Indices<br/>[B × N<sub>t</sub>]"] --> B
    L1 --> C
    L1 --> D
    
    M2["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"] --> C
    
    B --> E["Before Detectors Gradients<br/>[B × N<sub>t</sub> × (1 << N<sub>c</sub>)]"]
    C --> E
    D --> F["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;N<sub>c</sub>)&nbsp;×&nbsp;O]"]
    
    D3["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> G(propagate_through_detectors)
    M1["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"] --> G
    M2 --> G
    E --> G
    G --> H["Output: Input Gradients<br/>[B × I]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style F fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W3 fill:#000000,color:#ffffff
    style L1 fill:#000000,color:#ffffff
    style M1 fill:#000000,color:#ffffff
    style M2 fill:#000000,color:#ffffff
    style D3 fill:#000000,color:#ffffff
    style E fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B(fire_detectors_by_lookup_indices)
    M9["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"] --> B
    SC2["Sparse connectivity info"] --> B
    L5["Lookup Indices<br/>[B × N<sub>t</sub>]"] --> B
    
    B --> C["Firing Events<br/>Main&nbsp;+&nbsp;Alternative<br/>[B&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;max_fw_groups&nbsp;×&nbsp;2]"]
    
    W4["Weights<br/>[N<sub>t</sub> × (1 << N<sub>c</sub>) × O]"] --> D(gather_gradients)
    SC2 --> D
    C --> D
    
    D --> E["Before Detectors Gradients<br/>[B × N<sub>t</sub> × (1 << N<sub>c</sub>)]"]
    D --> F["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;N<sub>c</sub>)&nbsp;×&nbsp;O]"]
    
    D4["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> G(propagate_through_detectors)
    M3["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"] --> G
    M9 --> G
    L5 --> G
    E --> G
    G --> H["Output: Input Gradients<br/>[B × I]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style G fill:#81c784,color:#000000
    style F fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W4 fill:#000000,color:#ffffff
    style L5 fill:#000000,color:#ffffff
    style M3 fill:#000000,color:#ffffff
    style M9 fill:#000000,color:#ffffff
    style D4 fill:#000000,color:#ffffff
    style SC2 fill:#000000,color:#ffffff
    style E fill:#000000,color:#ffffff
    style C fill:#000000,color:#ffffff
```

## Sequential Mode - Forward Pass

```mermaid
flowchart TD
    A["Input<br/>[B × S × I]"] --> B("check_detectors_for_sequence<br/>Stream 0")
    D5["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> B
    C["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> D("check_positional_embeddings<br/>Stream 1")
    
    B --> E["Output: Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"]
    B --> F["Output: Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"]
    B --> G["Output: Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"]
    
    D --> H["Output: PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"]
    D --> I["Output: PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"]
    D --> J["Output: PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"]
    
    E --> K("fill_after_detectors_firing_stat<br/>(processes&nbsp;B&nbsp;×&nbsp;S&nbsp;×&nbsp;(S&minus;1)&nbsp;pairs&nbsp;with&nbsp;tiles)")
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L["Firing Statistics<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))]"]
    
    L --> M(densify_firing_stat)
    
    M --> N["Output: Main Firing Events<br/>[B × N<sub>t</sub> × S × (S-1) / 2]"]
    M --> O["Output: Alternative Firing Events<br/>[B × N<sub>t</sub> × S × (S-1)]"]
    
    W5["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))&nbsp;×&nbsp;O]&nbsp;"] --> P(fill_outputs_by_sparse_firings)
    SC3["Sparse connectivity info<br/>(if not fully connected)"] --> P
    N --> P
    
    P --> Q["Output: Output<br/>[B × S × O]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style K fill:#81c784,color:#000000
    style M fill:#81c784,color:#000000
    style P fill:#81c784,color:#000000
    style A fill:#000000,color:#ffffff
    style C fill:#000000,color:#ffffff
    style D5 fill:#000000,color:#ffffff
    style SC3 fill:#000000,color:#ffffff
    style W5 fill:#000000,color:#ffffff
    style L fill:#000000,color:#ffffff
    style N fill:#000000,color:#ffffff
    style O fill:#000000,color:#ffffff
    style E fill:#ffffff,color:#000000
    style F fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style I fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
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
    
    L3["Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"] --> I
    L4["PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    M5["Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"] --> I
    M6["Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"] --> I
    M7["PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"] --> I
    M8["PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    
    W6["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))&nbsp;×&nbsp;O]&nbsp;"] --> B
    W6 --> D
    SC4["Sparse connectivity info<br/>(if not fully connected)"] --> B
    SC4 --> C
    SC4 --> D
    
    B --> G["Before Detectors Gradients<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))]"]
    D --> G
    
    D6["Anchors<br/>[N<sub>t</sub> × 2N<sub>c</sub>]"] --> I("propagate_through_detectors_for_sequence<br/>(processes B×S×(S-1) pairs with tiles)")
    G --> I
    
    I --> J["Output: Input Gradients<br/>[B × S × I]"]
    I --> K["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    
    C --> H["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;(1&nbsp;<<&nbsp;(2N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub>))&nbsp;×&nbsp;O]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style I fill:#81c784,color:#000000
    style H fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style K fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W6 fill:#000000,color:#ffffff
    style E fill:#000000,color:#ffffff
    style F fill:#000000,color:#ffffff
    style L3 fill:#000000,color:#ffffff
    style L4 fill:#000000,color:#ffffff
    style M5 fill:#000000,color:#ffffff
    style M6 fill:#000000,color:#ffffff
    style M7 fill:#000000,color:#ffffff
    style M8 fill:#000000,color:#ffffff
    style D6 fill:#000000,color:#ffffff
    style SC4 fill:#000000,color:#ffffff
    style G fill:#000000,color:#ffffff
    style E fill:#000000,color:#ffffff
    style F fill:#000000,color:#ffffff
```
