# LUT Architecture Diagrams

## Project Architecture Overview

This diagram shows the complete architecture of the spiky project, including both Python and C++/CUDA layers.

```mermaid
graph TB
    subgraph "Python Layer"
        subgraph "LUT Module"
            LUT_LAYER["LUTLayer<br/>(PyTorch Modules)<br/>• LUTLayerBasic<br/>• Conv2DLUTLayer<br/>• LUTTransformer<br/>• Backpropagation Support"]
        end
        
        subgraph "SPNet Module"
            SPNET_PY["SpikingNet<br/>(Izhikevich Model)<br/>• Neuron Dynamics<br/>• STDP Learning<br/>• No Backpropagation"]
        end
        
        subgraph "Utils Module"
            SYNAPSE_GROWTH_PY["Synapse Growth Engine<br/>(Python Interface)<br/>• Connection Generation<br/>• Spatial Sampling<br/>• Growth Policies"]
            OTHER_UTILS["Other Utilities<br/>• torch_utils<br/>• chunk_of_connections<br/>• visual_helpers"]
        end
    end

    subgraph "C++/CUDA Layer"
        LUT_IMPL["LUT CUDA Implementation<br/>• LUTDataManager (F/I variants)<br/>• LUTRuntimeContext<br/>• Forward/Backward Kernels<br/>• Structure Management"]
        
        SPNET_IMPL["SPNet CUDA Implementation<br/>• SPNetDataManager (F/I variants)<br/>• SPNetRuntimeContext<br/>• Neuron Dynamics Kernels<br/>• STDP Computation"]
        
        CONN_MGR["Connections Manager<br/>(Reusable Component)<br/>• Sparse Connectivity<br/>• Synapse Storage<br/>• Group Organization<br/>• Delay Management"]
        
        SYNAPSE_GROWTH_CUDA["Synapse Growth CUDA<br/>• SynapseGrowthLowLevelEngine<br/>• GPU-Accelerated Growth<br/>• Connection Generation<br/>• Spatial Sampling"]
        
        CUDA_KERNEL_UTILS["CUDA Kernel Utilities<br/>• Firing Buffer<br/>• Spike Storage<br/>• Dense to Sparse Conversion<br/>• Kernels Logic Preprocessor"]
    end

    %% Python to CUDA connections
    LUT_LAYER -->|Pybind11| LUT_IMPL
    SPNET_PY -->|Pybind11| SPNET_IMPL
    SYNAPSE_GROWTH_PY -->|Pybind11| SYNAPSE_GROWTH_CUDA

    %% Shared infrastructure connections
    LUT_IMPL --> CONN_MGR
    SPNET_IMPL --> CONN_MGR
    SYNAPSE_GROWTH_CUDA --> CONN_MGR

    %% Styling
    classDef pythonLUT fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000000
    classDef pythonSPNet fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000000
    classDef pythonUtils fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000000
    classDef cudaLUT fill:#e1f5ff,stroke:#01579b,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    classDef cudaSPNet fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    classDef sharedInfra fill:#f3e5f5,stroke:#4a148c,stroke-width:4px,color:#000000
    classDef synapseGrowth fill:#fce4ec,stroke:#880e4f,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    classDef cudaUtils fill:#fff9c4,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 5,color:#000000

    class LUT_LAYER pythonLUT
    class SPNET_PY pythonSPNet
    class SYNAPSE_GROWTH_PY,OTHER_UTILS pythonUtils
    class LUT_IMPL cudaLUT
    class SPNET_IMPL cudaSPNet
    class CONN_MGR sharedInfra
    class SYNAPSE_GROWTH_CUDA synapseGrowth
    class CUDA_KERNEL_UTILS cudaUtils
```

## Component Descriptions

### Python Layer

**LUT Module**: Full-fledged PyTorch modules implementing different versions of LUT networks with backpropagation support. Includes basic layers, convolutional and projection layers, and transformer like architecture. Supports gradient computation and integration with PyTorch's autograd system.

**SPNet Module**: Implementation of spiking network model from polychronization paper (Izhikevitch, 2003). Handles neuron dynamics using the Izhikevich model, implements STDP (Spike-Timing-Dependent Plasticity) learning, but does not support backpropagation.

**Utils Module**: Collection of useful utilities. The **Synapse Growth Engine** is particularly important - it provides connection generation capabilities used by both LUT and SPNet modules. Supports various growth policies (convolutional, random rectangles and others) and spatial sampling strategies.

### C++/CUDA Layer

**LUT CUDA Implementation**: CUDA implementation of LUT networks. Includes data managers, runtime context, and CUDA kernels for forward and backward pass computations. Manages structure, compilation, and GPU execution.

**SPNet CUDA Implementation**: CUDA implementation of Izhkevitch spiking network model. Includes data managers, runtime context, and CUDA kernels for neuron dynamics, spike detection, and STDP weight updates.

**Connections Manager**: Reusable component responsible for effective sparse connectivity handling. Used by both LUT and SPNet implementations. Manages synapse storage, organizes connections into groups for efficient GPU access, handles delays, and provides indexed access patterns. This shared infrastructure enables efficient sparse neural network operations.

**Synapse Growth CUDA**: GPU-accelerated implementation of synapse growth. The core idea is placing neurons in 3D space and allowing them to "grow" synapses using spatial search. `SynapseGrowthLowLevelEngine` performs connection generation and spatial sampling directly on GPU, enabling efficient spatial queries and connection generation based on neuron positions and growth policies. Provides high-performance connection generation for both LUT and SPNet modules.

**CUDA Kernel Utilities**: Supporting utilities for CUDA kernel operations. Includes firing buffer for spike storage and event management, spike storage for recording spike events, dense to sparse conversion utilities for tensor operations, and kernels logic preprocessor for processing proto-defined kernel logic files.
