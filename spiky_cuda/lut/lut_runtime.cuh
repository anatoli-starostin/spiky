#pragma once
#include "lut.h"
#include "../misc/firing_buffer.h"

#define LUT_RUNTIME_KERNELS_TPB 1024
static_assert((LUT_RUNTIME_KERNELS_TPB % 2) == 0, "LUT_RUNTIME_KERNELS_TPB must be even");

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
};

