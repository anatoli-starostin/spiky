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
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Input<br/>[B × I]"] --> B(check_detectors)
    D1["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    
    B --> C["Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    
    W1["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> F(fill_outputs_fully_connected)
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
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Input<br/>[B × I]"] --> B(check_detectors)
    D2["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    
    B --> C["Output: Lookup Indices<br/>[B × N<sub>t</sub>]"]
    B --> D["Output: Min Anchor Deltas<br/>[B × N<sub>t</sub>]"]
    B --> E["Output: Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"]
    
    W2["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> F(fill_outputs_non_seq_sparse)
    SC1["Sparse connectivity info"] --> F
    C --> F
    
    F --> G["Output: Output<br/>[B × O]"]
    
    style B fill:#81c784,color:#000000
    style F fill:#81c784,color:#000000
    style C fill:#ffffff,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style D2 fill:#000000,color:#ffffff
    style SC1 fill:#000000,color:#ffffff
    style W2 fill:#000000,color:#ffffff
```

## Non-Sequential Mode - Backward Pass

### Fully Connected Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B("propagate_through_detectors_non_seq_fc<br/>Stream 0")
    A --> C("gather_w_gradients_non_seq_fc")
    
    W3["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B
    
    L1["Lookup Indices<br/>[B × N<sub>t</sub>]"] --> B
    L1 --> C
    
    M1["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"] --> B
    M2["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"] --> B
    
    D3["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    
    B --> D["Output: Input Gradients<br/>[B × I]"]
    C --> E["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>N<sub>c</sub></sup>&nbsp;×&nbsp;O]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W3 fill:#000000,color:#ffffff
    style L1 fill:#000000,color:#ffffff
    style M1 fill:#000000,color:#ffffff
    style M2 fill:#000000,color:#ffffff
    style D3 fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Output Gradients<br/>[B × O]"] --> B("propagate_through_detectors_non_seq_sparse<br/>Stream 0")
    A --> C("gather_w_gradients_non_seq_sparse")
    
    W4["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B
    
    L5["Lookup Indices<br/>[B × N<sub>t</sub>]"] --> B
    L5 --> C
    
    M3["Min Anchor Deltas<br/>[B × N<sub>t</sub>]"] --> B
    M9["Min Anchor Delta Indices<br/>[B × N<sub>t</sub>]"] --> B
    
    D4["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    
    SC2["Sparse connectivity info"] --> B
    SC2 --> C
    
    B --> D["Output: Input Gradients<br/>[B × I]"]
    C --> E["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>N<sub>c</sub></sup>&nbsp;×&nbsp;O]"]
    
    style B fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style D fill:#ffffff,color:#000000
    style E fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W4 fill:#000000,color:#ffffff
    style L5 fill:#000000,color:#ffffff
    style M3 fill:#000000,color:#ffffff
    style M9 fill:#000000,color:#ffffff
    style D4 fill:#000000,color:#ffffff
    style SC2 fill:#000000,color:#ffffff
```

## Sequential Mode - Forward Pass

### Fully Connected Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Input<br/>[B × S × I]"] --> B("check_detectors_seq<br/>Stream 0")
    D5["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    C["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> D("check_positional_embeddings<br/>Stream 1")
    
    B --> E["Output: Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"]
    B --> F["Output: Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"]
    B --> G["Output: Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"]
    
    D --> H["Output: PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"]
    D --> I["Output: PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"]
    D --> J["Output: PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"]
    
    E --> K("fill_outputs_fully_connected_seq<br/>(processes all pairs i,j where i < j)")
    H --> K
    
    W5["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]&nbsp;"] --> K
    
    K --> Q["Output: Output<br/>[B × S × O]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style K fill:#81c784,color:#000000
    style A fill:#000000,color:#ffffff
    style C fill:#000000,color:#ffffff
    style D5 fill:#000000,color:#ffffff
    style W5 fill:#000000,color:#ffffff
    style E fill:#ffffff,color:#000000
    style F fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style I fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style Q fill:#ffffff,color:#000000
```

### Sparse Connectivity Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Input<br/>[B × S × I]"] --> B("check_detectors_seq<br/>Stream 0")
    D5["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B
    C["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> D("check_positional_embeddings<br/>Stream 1")
    
    B --> E["Output: Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"]
    B --> F["Output: Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"]
    B --> G["Output: Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"]
    
    D --> H["Output: PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"]
    D --> I["Output: PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"]
    D --> J["Output: PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"]
    
    E --> K("fill_outputs_sparse_seq<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    H --> K
    
    W5["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]&nbsp;"] --> K
    SC3["Sparse connectivity info"] --> K
    
    K --> Q["Output: Output<br/>[B × S × O]"]
    
    style B fill:#81c784,color:#000000
    style D fill:#81c784,color:#000000
    style K fill:#81c784,color:#000000
    style A fill:#000000,color:#ffffff
    style C fill:#000000,color:#ffffff
    style D5 fill:#000000,color:#ffffff
    style SC3 fill:#000000,color:#ffffff
    style W5 fill:#000000,color:#ffffff
    style E fill:#ffffff,color:#000000
    style F fill:#ffffff,color:#000000
    style G fill:#ffffff,color:#000000
    style H fill:#ffffff,color:#000000
    style I fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style Q fill:#ffffff,color:#000000
```

## Sequential Mode - Backward Pass

### Fully Connected Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Output Gradients<br/>[B × S × O]"] --> I("propagate_through_detectors_seq_fc<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    
    W6["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]&nbsp;"] --> I
    
    L3["Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"] --> I
    L4["PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    M5["Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"] --> I
    M6["Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"] --> I
    M7["PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"] --> I
    M8["PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    
    D6["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> I
    
    I --> J["Output: Input Gradients<br/>[B × S × I]"]
    I --> K["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    I --> H["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]"]
    
    style I fill:#81c784,color:#000000
    style H fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style K fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W6 fill:#000000,color:#ffffff
    style L3 fill:#000000,color:#ffffff
    style L4 fill:#000000,color:#ffffff
    style M5 fill:#000000,color:#ffffff
    style M6 fill:#000000,color:#ffffff
    style M7 fill:#000000,color:#ffffff
    style M8 fill:#000000,color:#ffffff
    style D6 fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A["Output Gradients<br/>[B × S × O]"] --> I("propagate_through_detectors_seq_sparse<br/>Stream 0<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    A --> C("gather_w_gradients_seq_sparse<br/>Stream 1<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    
    W6["Weights<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]&nbsp;"] --> I
    W6 --> C
    
    L3["Q/K Lookup Indices<br/>[B × S × N<sub>t</sub>]"] --> I
    L3 --> C
    L4["PE Lookup Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    L4 --> C
    M5["Q/K Min Anchor Deltas<br/>[B × S × N<sub>t</sub>]"] --> I
    M6["Q/K Min Anchor Delta Indices<br/>[B&nbsp;×&nbsp;S&nbsp;×&nbsp;N<sub>t</sub>]"] --> I
    M7["PE Min Deltas<br/>[(S-1) × N<sub>t</sub>]"] --> I
    M8["PE Min Delta Indices<br/>[(S-1) × N<sub>t</sub>]"] --> I
    
    SC4["Sparse connectivity info"] --> I
    SC4 --> C
    
    D6["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> I
    
    I --> J["Output: Input Gradients<br/>[B × S × I]"]
    I --> K["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    
    C --> H["Output: Weight Gradients<br/>[N<sub>t</sub>&nbsp;×&nbsp;2<sup>2&nbsp;×&nbsp;N<sub>c</sub>&nbsp;+&nbsp;N<sub>pe</sub></sup>&nbsp;×&nbsp;O]"]
    
    style I fill:#81c784,color:#000000
    style C fill:#81c784,color:#000000
    style H fill:#ffffff,color:#000000
    style J fill:#ffffff,color:#000000
    style K fill:#ffffff,color:#000000
    style A fill:#000000,color:#ffffff
    style W6 fill:#000000,color:#ffffff
    style L3 fill:#000000,color:#ffffff
    style L4 fill:#000000,color:#ffffff
    style M5 fill:#000000,color:#ffffff
    style M6 fill:#000000,color:#ffffff
    style M7 fill:#000000,color:#ffffff
    style M8 fill:#000000,color:#ffffff
    style D6 fill:#000000,color:#ffffff
    style SC4 fill:#000000,color:#ffffff
```

## Product Mode - Forward Pass

Product mode processes two separate input sequences (`input_1` and `input_2`) together, computing lookup indices from pairs (i, j) where i comes from `input_1` and j comes from `input_2`. This is used for attention-like mechanisms.

### Fully Connected Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A1["Input 1<br/>[B × S × I<sub>1</sub>]"] --> B1("fill_outputs_product_fc<br/>(processes pairs i,j where i < j)")
    A2["Input 2<br/>[B × S × I<sub>2</sub>]"] --> B1
    D7["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B1
    PE1["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> B1
    
    W7["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B1
    
    B1 --> G1["Output: Output<br/>[B × S × O]"]
    
    style B1 fill:#81c784,color:#000000
    style G1 fill:#ffffff,color:#000000
    style A1 fill:#000000,color:#ffffff
    style A2 fill:#000000,color:#ffffff
    style D7 fill:#000000,color:#ffffff
    style PE1 fill:#000000,color:#ffffff
    style W7 fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A3["Input 1<br/>[B × S × I<sub>1</sub>]"] --> B2("fill_outputs_product_sparse<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    A4["Input 2<br/>[B × S × I<sub>2</sub>]"] --> B2
    D8["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B2
    PE2["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> B2
    
    W8["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B2
    SC5["Sparse connectivity info"] --> B2
    
    B2 --> G2["Output: Output<br/>[B × S × O]"]
    
    style B2 fill:#81c784,color:#000000
    style G2 fill:#ffffff,color:#000000
    style A3 fill:#000000,color:#ffffff
    style A4 fill:#000000,color:#ffffff
    style D8 fill:#000000,color:#ffffff
    style PE2 fill:#000000,color:#ffffff
    style SC5 fill:#000000,color:#ffffff
    style W8 fill:#000000,color:#ffffff
```

## Product Mode - Backward Pass

### Fully Connected Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A5["Output Gradients<br/>[B × S × O]"] --> B3("propagate_backward_product_fc<br/>Stream 0<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    A5 --> C1("gather_w_gradients_product_fc<br/>Stream 1<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    
    W9["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B3
    
    D9["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B3
    IN1["Input 1<br/>[B × S × I<sub>1</sub>]"] --> B3
    IN2["Input 2<br/>[B × S × I<sub>2</sub>]"] --> B3
    PE3["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> B3
    
    B3 --> D1["Output: Input Gradients 1<br/>[B × S × I<sub>1</sub>]"]
    B3 --> D2["Output: Input Gradients 2<br/>[B × S × I<sub>2</sub>]"]
    B3 --> PE4["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    
    C1 --> E1["Output: Weight Gradients<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"]
    
    style B3 fill:#81c784,color:#000000
    style C1 fill:#81c784,color:#000000
    style D1 fill:#ffffff,color:#000000
    style D2 fill:#ffffff,color:#000000
    style E1 fill:#ffffff,color:#000000
    style PE4 fill:#ffffff,color:#000000
    style A5 fill:#000000,color:#ffffff
    style W9 fill:#000000,color:#ffffff
    style D9 fill:#000000,color:#ffffff
    style IN1 fill:#000000,color:#ffffff
    style IN2 fill:#000000,color:#ffffff
    style PE3 fill:#000000,color:#ffffff
```

### Sparse Connectivity Case

```mermaid
%%{init: { "flowchart": { "defaultRenderer": "elk" } }}%%
flowchart TD
    A6["Output Gradients<br/>[B × S × O]"] --> B4("propagate_backward_product_sparse<br/>Stream 0<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    A6 --> C2("gather_w_gradients_product_sparse<br/>Stream 1<br/>(processes B×S×(S-1)/2 pairs with tiles)")
    
    W10["Weights<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"] --> B4
    
    D10["Anchors<br/>[N<sub>t</sub> × 2 × N<sub>c</sub>]"] --> B4
    IN3["Input 1<br/>[B × S × I<sub>1</sub>]"] --> B4
    IN4["Input 2<br/>[B × S × I<sub>2</sub>]"] --> B4
    PE5["Positional Embeddings<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"] --> B4
    
    SC6["Sparse connectivity info"] --> B4
    SC6 --> C2
    
    B4 --> D3["Output: Input Gradients 1<br/>[B × S × I<sub>1</sub>]"]
    B4 --> D4["Output: Input Gradients 2<br/>[B × S × I<sub>2</sub>]"]
    B4 --> PE6["Output: Positional Embedding Gradients<br/>[(S-1) × N<sub>t</sub> × N<sub>pe</sub>]"]
    
    C2 --> E2["Output: Weight Gradients<br/>[N<sub>t</sub> × 2<sup>N<sub>c</sub></sup> × O]"]
    
    style B4 fill:#81c784,color:#000000
    style C2 fill:#81c784,color:#000000
    style D3 fill:#ffffff,color:#000000
    style D4 fill:#ffffff,color:#000000
    style E2 fill:#ffffff,color:#000000
    style PE6 fill:#ffffff,color:#000000
    style A6 fill:#000000,color:#ffffff
    style W10 fill:#000000,color:#ffffff
    style D10 fill:#000000,color:#ffffff
    style IN3 fill:#000000,color:#ffffff
    style IN4 fill:#000000,color:#ffffff
    style PE5 fill:#000000,color:#ffffff
    style SC6 fill:#000000,color:#ffffff
```
