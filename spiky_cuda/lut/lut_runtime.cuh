#pragma once
#include "lut.h"
#include "../misc/firing_buffer.h"

#define LUT_RUNTIME_KERNELS_TPB 1024
static_assert((LUT_RUNTIME_KERNELS_TPB % 2) == 0, "LUT_RUNTIME_KERNELS_TPB must be even");
// Macro to calculate optimal threads per block (min of TPB and variable size)
#define LUT_RUNTIME_KERNELS_TPB_OPT(var) ((var) < LUT_RUNTIME_KERNELS_TPB ? (var) : LUT_RUNTIME_KERNELS_TPB)
// Macro to calculate number of blocks needed for a given variable
#define LUT_RUNTIME_NUM_BLOCKS(var) (((var) + LUT_RUNTIME_KERNELS_TPB_OPT(var) - 1) / LUT_RUNTIME_KERNELS_TPB_OPT(var))

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
    uint32_t synapse_group_size;
    REAL_DT first_synapse_meta_lr;

    uint32_t batch_size;
    uint32_t sequence_length;

    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler;
    #endif

    BaseSynapseMeta *base_synapse_metas;
    IndexedSynapsesInfo *lookup_neuron_synapses_infos;

    FiringBuffer *firing_buffer;
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
        uint32_t synapse_group_size,
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

    void _ensure_firing_buffer_size(uint64_t max_groups_to_fire);

    // base methods

    void forward_step(
        EXTERNAL_REAL_DT *weights,
        uint32_t batch_size,
        EXTERNAL_REAL_DT *input,
        AnchorsPair *detectors,
        EXTERNAL_REAL_DT *target_output,
        int32_t *target_lookup_indices,
        EXTERNAL_REAL_DT *target_min_anchor_deltas,
        int32_t *target_min_anchor_delta_indices
    );

    void backward_backprop(
        EXTERNAL_REAL_DT *weights,
        uint32_t batch_size,
        // external gradients
        EXTERNAL_REAL_DT *output_gradients,
        // data from forward pass
        EXTERNAL_REAL_DT *input,
        AnchorsPair *detectors,
        int32_t *lookup_indices,
        EXTERNAL_REAL_DT *min_anchor_deltas,
        int32_t *min_anchor_delta_indices,
        // gradients that we need to calculate
        SUMMATION32_DT *before_detectors_gradients,
        EXTERNAL_REAL_DT *target_input_gradients,
        EXTERNAL_REAL_DT *target_weights_gradients
    );

    void forward_step_concat(
        EXTERNAL_REAL_DT *weights,
        EXTERNAL_REAL_DT *positional_embeddings,
        uint32_t batch_size,
        EXTERNAL_REAL_DT *input,
        AnchorsPair *detectors,
        EXTERNAL_REAL_DT *target_output,
        int32_t *target_lookup_indices,
        EXTERNAL_REAL_DT *target_min_anchor_deltas,
        int32_t *target_min_anchor_delta_indices,
        int32_t *target_positional_lookup_indices,
        EXTERNAL_REAL_DT *target_positional_min_deltas,
        int32_t *target_positional_min_delta_indices,
        int32_t *target_sparse_firing_buffer,
        EXTERNAL_REAL_DT *target_firing_stat
    );

    void backward_backprop_concat(
        EXTERNAL_REAL_DT *weights,
        EXTERNAL_REAL_DT *positional_embeddings,
        uint32_t batch_size,
        // external gradients
        EXTERNAL_REAL_DT *output_gradients,
        // data from forward pass
        EXTERNAL_REAL_DT *input,
        AnchorsPair *detectors,
        int32_t *lookup_indices,
        EXTERNAL_REAL_DT *min_anchor_deltas,
        int32_t *min_anchor_delta_indices,
        int32_t *positional_lookup_indices,
        EXTERNAL_REAL_DT *positional_min_deltas,
        int32_t *positional_min_delta_indices,
        // forward statistics from forward pass
        int32_t *sparse_firing_buffer,
        // gradients that we need to calculate
        SUMMATION32_DT *before_detectors_gradients,
        EXTERNAL_REAL_DT *target_input_gradients,
        EXTERNAL_REAL_DT *target_weights_gradients,
        EXTERNAL_REAL_DT *target_positional_embeddings_gradients
    );
};

