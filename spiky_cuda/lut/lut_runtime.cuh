#pragma once
#include "lut.h"
#include "../misc/firing_buffer.h"

#define LUT_RUNTIME_KERNELS_TPB 1024
static_assert((LUT_RUNTIME_KERNELS_TPB % 2) == 0, "LUT_RUNTIME_KERNELS_TPB must be even");
// Macro to calculate optimal threads per block (min of TPB and variable size)
#define LUT_RUNTIME_KERNELS_TPB_OPT(var) (((var) < LUT_RUNTIME_KERNELS_TPB) ? (var) : LUT_RUNTIME_KERNELS_TPB)
// Macro to calculate number of blocks needed for a given variable
#define LUT_RUNTIME_NUM_BLOCKS(var) (((var) + LUT_RUNTIME_KERNELS_TPB_OPT(var) - 1) / LUT_RUNTIME_KERNELS_TPB_OPT(var))
#define TILE 32

#define LUT_RUNTIME_CONTEXT_CLASS PFX(LUTRuntimeContext)
class LUT_RUNTIME_CONTEXT_CLASS {
private:
    uint8_t *lut_data;
    int device;

    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t n_detectors;
    uint32_t n_anchors_per_detector;
    uint32_t n_lookup_neurons;
    uint32_t positional_dim;
    uint32_t forward_group_size;
    uint32_t backward_group_size;
    REAL_DT first_synapse_meta_lr;

    uint32_t batch_size;
    uint32_t sequence_length;

    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler;
    #endif

    BaseSynapseMeta *base_synapse_metas;
    IndexedSynapsesInfo *lookup_neuron_synapses_infos;

    uint32_t max_forward_groups_per_neuron;
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    uint64_t n_weights;
    double int_rescaler;
    #endif

    NeuronDataId_t first_synapse_id;
public:
    // base constructor
    LUT_RUNTIME_CONTEXT_CLASS(
        uint8_t *lut_data,
        int device,
        uint32_t n_inputs,
        uint32_t n_outputs,
        uint32_t n_detectors,
        uint32_t n_anchors_per_detector,
        uint32_t n_lookup_neurons,
        uint32_t sequence_length,
        uint32_t positional_dim,
        uint32_t forward_group_size,
        uint32_t backward_group_size,
        uint32_t max_forward_groups_per_neuron,
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        uint64_t n_weights,
        double int_rescaler,
        #endif
        #ifdef ENABLE_PROFILING
        SimpleProfiler& profiler,
        #endif
        BaseSynapseMeta *base_synapse_metas,
        IndexedSynapsesInfo *lookup_neuron_synapses_infos,
        NeuronDataId_t first_synapse_id
    );

    ~LUT_RUNTIME_CONTEXT_CLASS();

    // base methods

    void forward_step(
        EXTERNAL_REAL_DT *r_weights,
        uint32_t batch_size,
        EXTERNAL_REAL_DT *r_input,
        AnchorsPair *r_detectors,
        EXTERNAL_REAL_DT *w_output,
        int32_t *w_lookup_indices,
        EXTERNAL_REAL_DT *w_min_anchor_deltas,
        int32_t *w_min_anchor_delta_indices,
        int64_t *w_sparse_firing_buffer  // Can be nullptr
        #ifndef NO_CUDA
        , cudaStream_t *cuda_streams
        #endif
    );

    void backward_backprop(
        EXTERNAL_REAL_DT *r_weights,
        uint32_t batch_size,
        // external gradients
        EXTERNAL_REAL_DT *r_output_gradients,
        // data from forward pass
        EXTERNAL_REAL_DT *r_input,
        AnchorsPair *r_detectors,
        int32_t *r_lookup_indices,
        EXTERNAL_REAL_DT *r_min_anchor_deltas,
        int32_t *r_min_anchor_delta_indices,
        // gradients that we need to calculate
        SUMMATION32_DT *w_before_detectors_gradients,
        EXTERNAL_REAL_DT *w_input_gradients,
        int64_t *w_sparse_firing_buffer_ptr,  // Can be nullptr
        EXTERNAL_REAL_DT external_lr,
        EXTERNAL_REAL_DT *w_weights_gradients  // Can be nullptr when external_lr != 0.0
        #ifndef NO_CUDA
        , cudaStream_t *cuda_streams
        #endif
    );

    void forward_step_concat(
        EXTERNAL_REAL_DT *r_weights,
        EXTERNAL_REAL_DT *r_positional_embeddings,
        uint32_t batch_size,
        EXTERNAL_REAL_DT *r_input,
        AnchorsPair *r_detectors,
        EXTERNAL_REAL_DT *w_output,
        int32_t *w_lookup_indices,
        EXTERNAL_REAL_DT *w_min_anchor_deltas,
        int32_t *w_min_anchor_delta_indices,
        int32_t *w_positional_lookup_indices,
        EXTERNAL_REAL_DT *w_positional_min_deltas,
        int32_t *w_positional_min_delta_indices,
        int64_t *w_sparse_firing_buffer,
        int64_t *w_sparse_firing_buffer_alternative,
        EXTERNAL_REAL_DT *w_firing_stat
        #ifndef NO_CUDA
        , cudaStream_t *cuda_streams
        #endif
    );

    void backward_backprop_concat(
        EXTERNAL_REAL_DT *r_weights,
        EXTERNAL_REAL_DT *r_positional_embeddings,
        uint32_t batch_size,
        // external gradients
        EXTERNAL_REAL_DT *r_output_gradients,
        // data from forward pass
        EXTERNAL_REAL_DT *r_input,
        AnchorsPair *r_detectors,
        int32_t *r_lookup_indices,
        EXTERNAL_REAL_DT *r_min_anchor_deltas,
        int32_t *r_min_anchor_delta_indices,
        int32_t *r_positional_lookup_indices,
        EXTERNAL_REAL_DT *r_positional_min_deltas,
        int32_t *r_positional_min_delta_indices,
        SUMMATION32_DT *w_before_detectors_gradients,
        NeuronShiftFiring *r_sparse_firings,
        uint32_t n_sparse_firings,
        NeuronShiftFiring *r_sparse_firing_alternatives,
        uint32_t n_sparse_firing_alternatives,
        EXTERNAL_REAL_DT *w_input_gradients,
        EXTERNAL_REAL_DT *w_positional_embeddings_gradients,
        EXTERNAL_REAL_DT external_lr,
        EXTERNAL_REAL_DT *w_weights_gradients  // Can be nullptr when external_lr != 0.0
        #ifndef NO_CUDA
        , cudaStream_t *cuda_streams
        #endif
    );
};

